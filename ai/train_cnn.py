import sys
import os
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ai.supervised import SupervisedAI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the supervised Caro CNN model.")
    parser.add_argument("--logs-dir", default="data/human_logs", help="directory containing JSON game logs")
    parser.add_argument("--validation-split", type=float, default=0.2, help="fraction of usable logs kept for validation")
    parser.add_argument("--rebuild-model", action="store_true", help="start a new 3-channel CNN instead of continuing an existing model")
    args = parser.parse_args()

    print("=== START CNN TRAINING ===")
    ai = SupervisedAI(load_existing=not args.rebuild_model)
    ai.train_model(
        logs_dir=args.logs_dir,
        validation_split=args.validation_split,
        rebuild_model=args.rebuild_model,
    )
    print("=== TRAINING COMPLETE ===")
