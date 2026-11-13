# Multimodal fusion of fundus images, vascular biomarkers and demographics for CIMT classification

Code for the paper *Multimodal Fusion of Fundus Imaging, Vascular Biomarkers, and Demographics Improves Subclinical Cardiovascular Risk Classification over Image-Only Deep Learning*.

The task is to detect carotid intima-media thickness (CIMT) thickening (>= 0.9 mm) from retinal fundus images. Two arms are compared:

- **`image_only`**: a RETFound ViT-Large/16 fine-tuned on fundus images (baseline).
- **`fusion`**: late-fusion stacking. Three base learners are trained independently (RETFound on images, an MLP on age and gender, an MLP on 31 VascX biomarkers). A logistic regression over their three predicted probabilities, fitted on the validation split, gives the final prediction.

On the 200-image test set the image-only baseline reaches AUROC 0.8731 and the fusion arm 0.9185.

## Repository structure

| File | Purpose |
|---|---|
| `config.py` | Paths, constants and hyperparameters |
| `dataset.py` | Loading of metadata, biomarkers and images; train/val/test split; biomarker scaling |
| `model.py` | RETFound backbone, demographic and biomarker encoders, base-learner model |
| `train.py` / `test.py` | Training and evaluation of a base learner |
| `fusion.py` | Late-fusion stacking (probability extraction, meta-learner fitting, evaluation) |
| `main.py` | Command-line entry point |
| `utils.py` | Seeding, logging, RETFound weight loading, focal loss, metrics, early stopping |
| `plot_results.py` | ROC/PR curves, confusion matrices, calibration, training curves |
| `ablation_modality_combinations.py` | Ablation over subsets of modalities with DeLong tests |

## Installation

```bash
git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt
```

Python 3.10 and a CUDA-capable GPU are recommended. Training the RETFound branch on a CPU is very slow.

## Data and pretrained weights

The data and weights are not included in this repository.

1. **Dataset.** Download the public CIMT fundus dataset used in the paper ([dataset reference]) and arrange it as:

   ```
   data/
   |-- data_info.json    patient metadata
   |-- biomarkers.xlsx   VascX biomarkers, one row per image
   `-- raw/              fundus images referenced by data_info.json
   ```

   `data_info.json` has one entry per patient:

   ```json
   "2491006": {"gender": 0, "thickness": 0.8, "label": 0, "group": 1,
               "True_age": 63, "age": 0.6849,
               "right_eye": "2491006_R.png", "left_eye": "2491006_L.png"}
   ```

   `age` is min-max scaled to [0, 1] and `group` 1, 2, 3 marks the training, validation and test patients. The biomarkers are computed with [VascX](https://github.com/Eyened/rtnls_vascx_models); the workbook needs an `image_id` column and the 31 columns listed in `config.BIOMARKER_COLUMNS`. If the biomarkers are not in the first sheet of the workbook, set `BIOMARKERS_SHEET_NAME` in `config.py`.

2. **RETFound weights.** Download `RETFound_mae_natureCFP.pth` from the [RETFound repository](https://github.com/rmaphoh/RETFound_MAE) and place it in the project folder.

Paths can also be set with environment variables: `CIMT_DATA_DIR`, `CIMT_OUTPUT_DIR` and `RETFOUND_CHECKPOINT`.

## Running the experiments

```bash
# image-only baseline (run first)
python main.py --stage all --model_mode image_only

# late-fusion stacking (trains the demographic and biomarker models if needed)
python main.py --stage all --model_mode fusion
```

`--stage train` and `--stage test` run the two phases separately.

Outputs are written to `results/`:

- `checkpoints/`: best model of each base learner, the biomarker scaler and the fitted meta-learner
- `logs/`: training and test logs
- `test_results_<mode>.json` and `test_predictions_<mode>.csv`: test metrics and per-image predictions (the fusion file also contains the three base-learner probabilities)
- `val_predictions_fusion.csv`: validation probabilities used to fit the meta-learner
- `fusion_meta_learner_info.json`: meta-learner coefficients and base-learner validation AUROCs

## Figures

```bash
python plot_results.py --model_mode fusion --compare image_only
```

## Modality ablation

```bash
cd results
python ../ablation_modality_combinations.py val_predictions_fusion.csv test_predictions_fusion.csv
```

This fits the meta-learner on every subset of modalities, evaluates it on the test split, and reports the AUROC with a patient-level bootstrap interval and a DeLong test against the full fusion. Results are saved to `ablation_modality_results.csv`.

## Reproducibility notes

- All random seeds are set to 42 (`config.SEED`).
- The test split is fixed by the `group` field in the metadata (1 = train, 2 = validation, 3 = test).
- GPU kernels are not fully deterministic, so retraining can change the metrics slightly. The meta-learner is deterministic given the base-learner probabilities.
- The meta-learner regularization (`META_LEARNER_C = 1.0`) is the scikit-learn default and is not tuned.

## License

MIT, see `LICENSE`.
