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
    parser.add_argument("--model-path", default="data/models/caro_supervised.h5", help="model file to load and save")
    parser.add_argument("--epochs", type=int, default=10, help="maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="training batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Adam learning rate")
    parser.add_argument("--seed", type=int, default=0, help="seed used for deterministic shuffling")
    parser.add_argument("--no-shuffle", action="store_true", help="keep files and training samples in sorted order")
    parser.add_argument("--no-archive", action="store_true", help="leave training logs in place after training")
    parser.add_argument("--no-checkpoint", action="store_true", help="do not save the best epoch checkpoint")
    args = parser.parse_args()

    print("=== START CNN TRAINING ===")
    ai = SupervisedAI(model_path=args.model_path, load_existing=not args.rebuild_model)
    ai.train_model(
        logs_dir=args.logs_dir,
        validation_split=args.validation_split,
        rebuild_model=args.rebuild_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        shuffle=not args.no_shuffle,
        archive=not args.no_archive,
        checkpoint=not args.no_checkpoint,
    )
    print("=== TRAINING COMPLETE ===")
