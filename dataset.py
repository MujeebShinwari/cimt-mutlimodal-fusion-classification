"""Data loading: patient metadata (JSON), VascX biomarkers (Excel) and fundus images.

Metadata format, one entry per patient:
    "2491006": {"gender": 0, "thickness": 0.8, "label": 0, "group": 1,
                "True_age": 63, "age": 0.6849, "right_eye": "2491006_R.png",
                "left_eye": "2491006_L.png"}
"age" is min-max scaled to [0, 1]; "group" 1/2/3 marks train/val/test.
The biomarker workbook has one row per image, an image-id column and one
column per entry of config.BIOMARKER_COLUMNS.
"""

import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torchvision import transforms

import config


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def load_metadata(json_path: Path = config.METADATA_JSON_PATH) -> dict:
    if not json_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {json_path}")
    with open(json_path, "r") as f:
        metadata = json.load(f)
    print(f"Loaded metadata for {len(metadata)} patients from {json_path.name}")
    return metadata


def _find_header_row(raw_df: pd.DataFrame, max_rows_to_scan: int = 10) -> int:
    for row_idx in range(min(max_rows_to_scan, len(raw_df))):
        row_values = [_normalize(v) for v in raw_df.iloc[row_idx].tolist()]
        if any(candidate.replace(" ", "") in row_values for candidate in config.IMAGE_ID_COLUMN_CANDIDATES):
            return row_idx
    return 0


def load_biomarkers(xlsx_path: Path = config.BIOMARKERS_XLSX_PATH,
                    sheet_name=config.BIOMARKERS_SHEET_NAME) -> pd.DataFrame:
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Biomarker workbook not found: {xlsx_path}")

    sheet = sheet_name if sheet_name is not None else 0
    raw = pd.read_excel(xlsx_path, sheet_name=sheet, header=None)
    header_row = _find_header_row(raw)
    df = pd.read_excel(xlsx_path, sheet_name=sheet, header=header_row)

    normalized_to_actual = {_normalize(c): c for c in df.columns}

    id_col = None
    for candidate in config.IMAGE_ID_COLUMN_CANDIDATES:
        if _normalize(candidate) in normalized_to_actual:
            id_col = normalized_to_actual[_normalize(candidate)]
            break
    if id_col is None:
        raise ValueError(
            f"No image-id column found in {xlsx_path.name}; expected one of "
            f"{config.IMAGE_ID_COLUMN_CANDIDATES}. Columns present: {list(df.columns)}"
        )

    resolved_columns, missing_columns = {}, []
    for wanted in config.BIOMARKER_COLUMNS:
        key = _normalize(wanted)
        if key in normalized_to_actual:
            resolved_columns[wanted] = normalized_to_actual[key]
        else:
            missing_columns.append(wanted)
    if missing_columns:
        print(f"WARNING: {len(missing_columns)} biomarker columns not found and treated as "
              f"missing: {missing_columns}")

    out = pd.DataFrame(index=df[id_col].astype(str).map(lambda s: _normalize(Path(s).stem)))
    for wanted, actual in resolved_columns.items():
        out[wanted] = pd.to_numeric(df[actual].values, errors="coerce")
    for wanted in missing_columns:
        out[wanted] = np.nan

    out = out[config.BIOMARKER_COLUMNS]
    out = out[~out.index.duplicated(keep="first")]
    print(f"Loaded biomarkers for {len(out)} images "
          f"({len(config.BIOMARKER_COLUMNS) - len(missing_columns)}/{len(config.BIOMARKER_COLUMNS)} columns found)")
    return out


def build_sample_table(metadata: dict, biomarkers_df: pd.DataFrame) -> pd.DataFrame:
    """One row per image with label, demographics, group and biomarker values."""
    rows = []
    for patient_id, entry in metadata.items():
        for eye_key in ("left_eye", "right_eye"):
            filename = entry.get(eye_key)
            if not filename:
                continue
            norm_id = _normalize(Path(filename).stem)

            row = {
                "patient_id": patient_id,
                "eye": "L" if eye_key == "left_eye" else "R",
                "filename": filename,
                "image_path": str(config.IMAGES_DIR / filename),
                "label": int(entry["label"]),
                "thickness": float(entry["thickness"]),
                "true_age": float(entry.get("True_age", entry.get("true_age", np.nan))),
                "age": float(entry["age"]),
                "gender": float(entry["gender"]),
                "group": int(entry.get("group", -1)),
            }

            if norm_id in biomarkers_df.index:
                biomarker_values = biomarkers_df.loc[norm_id]
                if isinstance(biomarker_values, pd.DataFrame):
                    biomarker_values = biomarker_values.iloc[0]
                for col in config.BIOMARKER_COLUMNS:
                    row[col] = biomarker_values[col]
            else:
                for col in config.BIOMARKER_COLUMNS:
                    row[col] = np.nan
            rows.append(row)

    table = pd.DataFrame(rows)

    n_missing_image = (~table["image_path"].map(lambda p: Path(p).exists())).sum()
    if n_missing_image > 0:
        print(f"WARNING: {n_missing_image}/{len(table)} image files not found under {config.IMAGES_DIR}")
    n_missing_biomarkers = table[config.BIOMARKER_COLUMNS].isna().all(axis=1).sum()
    if n_missing_biomarkers > 0:
        print(f"WARNING: {n_missing_biomarkers}/{len(table)} images have no biomarker row (values will be imputed)")
    n_missing_demo = table[config.DEMOGRAPHIC_FEATURES].isna().any(axis=1).sum()
    if n_missing_demo > 0:
        print(f"WARNING: {n_missing_demo}/{len(table)} images are missing age or gender")

    print(f"Built sample table with {len(table)} images ({table['patient_id'].nunique()} patients)")
    return table


def split_sample_table(table: pd.DataFrame) -> dict:
    if config.SPLIT_MODE == "predefined_group":
        splits = {}
        for group_value, split_name in config.GROUP_TO_SPLIT.items():
            splits[split_name] = table[table["group"] == group_value].reset_index(drop=True)
        for name, df in splits.items():
            if len(df) == 0:
                print(f"WARNING: the '{name}' split is empty; check config.GROUP_TO_SPLIT")
        return splits

    if config.SPLIT_MODE == "random":
        patients = table[["patient_id", "label"]].drop_duplicates("patient_id")
        train_ids, temp_ids = train_test_split(
            patients, test_size=1 - config.RANDOM_SPLIT_RATIOS["train"],
            stratify=patients["label"], random_state=config.RANDOM_SEED,
        )
        relative_val = config.RANDOM_SPLIT_RATIOS["val"] / (
            config.RANDOM_SPLIT_RATIOS["val"] + config.RANDOM_SPLIT_RATIOS["test"]
        )
        val_ids, test_ids = train_test_split(
            temp_ids, test_size=1 - relative_val,
            stratify=temp_ids["label"], random_state=config.RANDOM_SEED,
        )
        return {
            "train": table[table["patient_id"].isin(train_ids["patient_id"])].reset_index(drop=True),
            "val": table[table["patient_id"].isin(val_ids["patient_id"])].reset_index(drop=True),
            "test": table[table["patient_id"].isin(test_ids["patient_id"])].reset_index(drop=True),
        }

    raise ValueError(f"Unknown SPLIT_MODE: {config.SPLIT_MODE}")


def fit_biomarker_scaler(train_table: pd.DataFrame):
    """Median imputation followed by standardization, fitted on the training split only."""
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    values = train_table[config.BIOMARKER_COLUMNS].values
    scaler.fit(imputer.fit_transform(values))
    with open(config.BIOMARKER_SCALER_PATH, "wb") as f:
        pickle.dump({"imputer": imputer, "scaler": scaler}, f)
    print(f"Fitted biomarker imputer/scaler on {len(train_table)} training images")
    return imputer, scaler


def load_biomarker_scaler():
    if not config.BIOMARKER_SCALER_PATH.exists():
        raise FileNotFoundError(
            f"No fitted biomarker scaler at {config.BIOMARKER_SCALER_PATH}; run training first."
        )
    with open(config.BIOMARKER_SCALER_PATH, "rb") as f:
        d = pickle.load(f)
    return d["imputer"], d["scaler"]


def apply_biomarker_scaler(table: pd.DataFrame, imputer, scaler) -> np.ndarray:
    values = table[config.BIOMARKER_COLUMNS].values
    return scaler.transform(imputer.transform(values)).astype(np.float32)


def build_demographic_array(table: pd.DataFrame) -> np.ndarray:
    values = table[config.DEMOGRAPHIC_FEATURES].values.astype(np.float32)
    if np.isnan(values).any():
        col_medians = np.nanmedian(values, axis=0)
        nan_mask = np.isnan(values)
        for col_idx in range(values.shape[1]):
            values[nan_mask[:, col_idx], col_idx] = col_medians[col_idx]
    return values


def build_transforms(is_train: bool) -> transforms.Compose:
    normalize = transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    if is_train:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])


class CIMTTriModalDataset(Dataset):
    """One item = one fundus image with its demographics, biomarkers and label.

    Images are only read from disk when mode is "image_only".
    """

    def __init__(self, table: pd.DataFrame, biomarker_values: np.ndarray,
                 demographic_values: np.ndarray, is_train: bool, mode: str = None):
        assert len(table) == len(biomarker_values) == len(demographic_values)
        self.table = table.reset_index(drop=True)
        self.biomarker_values = biomarker_values
        self.demographic_values = demographic_values
        self.transform = build_transforms(is_train)
        self.mode = mode or config.MODEL_MODE

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int):
        row = self.table.iloc[idx]
        biomarkers = torch.tensor(self.biomarker_values[idx], dtype=torch.float32)
        demographics = torch.tensor(self.demographic_values[idx], dtype=torch.float32)

        if self.mode == "image_only":
            image = self.transform(Image.open(row["image_path"]).convert("RGB"))
        else:
            image = torch.zeros(3, config.IMG_SIZE, config.IMG_SIZE)

        return {
            "image": image,
            "demographics": demographics,
            "biomarkers": biomarkers,
            "label": int(row["label"]),
            "patient_id": row["patient_id"],
            "eye": row["eye"],
            "true_age": float(row["true_age"]),
        }


def prepare_datasets(mode: str = None, fit_scaler: bool = True):
    """Returns ({"train", "val", "test"} datasets, {"train", "val", "test"} tables)."""
    metadata = load_metadata()
    biomarkers_df = load_biomarkers()
    table = build_sample_table(metadata, biomarkers_df)
    splits = split_sample_table(table)

    if fit_scaler:
        imputer, scaler = fit_biomarker_scaler(splits["train"])
    else:
        imputer, scaler = load_biomarker_scaler()

    datasets = {}
    for split_name, split_table in splits.items():
        datasets[split_name] = CIMTTriModalDataset(
            split_table,
            apply_biomarker_scaler(split_table, imputer, scaler),
            build_demographic_array(split_table),
            is_train=(split_name == "train"),
            mode=mode,
        )
    return datasets, splits
