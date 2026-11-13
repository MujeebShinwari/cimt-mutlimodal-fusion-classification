"""Command-line entry point.

    python main.py --stage all --model_mode image_only   # image-only baseline
    python main.py --stage all --model_mode fusion       # late-fusion stacking

Running the fusion arm trains the demographic and biomarker models first if
their checkpoints do not exist yet.
"""

import argparse

import config
import test as test_module
import train as train_module

VALID_MODES = ["image_only", "fusion"]


def parse_args():
    parser = argparse.ArgumentParser(description="CIMT classification: image-only baseline vs. multimodal fusion")
    parser.add_argument("--stage", type=str, default="all", choices=["train", "test", "all"],
                        help="train, test an existing checkpoint, or both")
    parser.add_argument("--model_mode", type=str, default=None, choices=VALID_MODES,
                        help="image_only or fusion (default: config.MODEL_MODE)")
    return parser.parse_args()


def main():
    args = parse_args()
    model_mode = args.model_mode or config.MODEL_MODE

    if args.stage in ("train", "all"):
        print(f"\n=== TRAINING (model_mode={model_mode}) ===")
        train_module.run_training(model_mode=model_mode)

    if args.stage in ("test", "all"):
        print(f"\n=== TESTING (model_mode={model_mode}) ===")
        test_module.run_testing(model_mode=model_mode)


if __name__ == "__main__":
    main()
