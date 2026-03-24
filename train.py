import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional


def _configure_env_before_tensorflow() -> None:
    # Must run before any `import tensorflow` (pulled in via `src.data` / `src.model`).
    # Helps on macOS with noisy C++ logs and occasional long stalls during first init.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "2")


_configure_env_before_tensorflow()
print("Loading TensorFlow and project modules (first run can take 1–2 minutes)...", file=sys.stderr, flush=True)

from src.data import (
    build_label_mapping,
    load_training_dataframe,
    make_dataset,
    split_train_val,
)
from src.model import build_model, build_training_callbacks

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 15
ARTIFACTS_DIR = "artifacts"
MODEL_FILE = "butterfly_model.keras"
CLASS_NAMES_FILE = "class_names.json"
TRAIN_CONFIG_FILE = "train_config.json"


def resolve_data_root(user_data_root: Optional[str]) -> str:
    import kagglehub

    if user_data_root:
        if not os.path.exists(user_data_root):
            raise FileNotFoundError(f"Provided data root does not exist: {user_data_root}")
        return user_data_root

    local_csv = os.path.join(os.getcwd(), "Training_set.csv")
    if os.path.exists(local_csv):
        return os.getcwd()

    return kagglehub.dataset_download("phucthaiv02/butterfly-image-classification")


def parse_args():
    parser = argparse.ArgumentParser(description="Train butterfly classifier once and export artifacts.")
    parser.add_argument("--data-root", type=str, default=None, help="Path to dataset root.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    return parser.parse_args()


def main():
    import matplotlib.pyplot as plt

    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    print(f"Using dataset root: {data_root}")

    df = load_training_dataframe(data_root)
    if len(df) == 0:
        raise RuntimeError("No valid images were found. Check dataset path and CSV filenames.")
    class_names, num_classes = build_label_mapping(df)
    train_df, val_df = split_train_val(df)

    train_ds = make_dataset(
        train_df["abs_path"].values,
        train_df["label_idx"].values,
        img_size=IMG_SIZE,
        num_classes=num_classes,
        batch_size=args.batch_size,
        training=True,
    )
    val_ds = make_dataset(
        val_df["abs_path"].values,
        val_df["label_idx"].values,
        img_size=IMG_SIZE,
        num_classes=num_classes,
        batch_size=args.batch_size,
        training=False,
    )

    model = build_model(num_classes=num_classes, img_size=IMG_SIZE)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=build_training_callbacks(),
    )

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    model_path = os.path.join(ARTIFACTS_DIR, MODEL_FILE)
    model.save(model_path)

    class_names_path = os.path.join(ARTIFACTS_DIR, CLASS_NAMES_FILE)
    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    train_config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "img_size": list(IMG_SIZE),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "num_classes": num_classes,
        "model_name": "MobileNetV3Large",
        "preprocess_fn": "tf.keras.applications.mobilenet_v3.preprocess_input",
    }
    train_config_path = os.path.join(ARTIFACTS_DIR, TRAIN_CONFIG_FILE)
    with open(train_config_path, "w", encoding="utf-8") as f:
        json.dump(train_config, f, ensure_ascii=False, indent=2)

    plt.figure(figsize=(8, 5))
    plt.plot(history.history.get("accuracy", []), label="train_accuracy")
    plt.plot(history.history.get("val_accuracy", []), label="val_accuracy")
    plt.title("Accuracy Evolution")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(ARTIFACTS_DIR, "training_accuracy.png")
    plt.savefig(plot_path)
    plt.close()

    print("Training complete. Artifacts:")
    print(f"- Model: {model_path}")
    print(f"- Class names: {class_names_path}")
    print(f"- Config: {train_config_path}")
    print(f"- Plot: {plot_path}")


if __name__ == "__main__":
    main()
