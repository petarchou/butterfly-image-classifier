import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def build_all_files_map(root_dir: str) -> Dict[str, str]:
    file_map: Dict[str, str] = {}
    for current_root, _, files in os.walk(root_dir):
        for file_name in files:
            if file_name.lower().endswith(IMAGE_EXTENSIONS):
                file_map[file_name] = os.path.join(current_root, file_name)
    return file_map


def load_training_dataframe(data_root: str, csv_name: str = "Training_set.csv") -> pd.DataFrame:
    csv_path = os.path.join(data_root, csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find training CSV at: {csv_path}")

    df = pd.read_csv(csv_path)
    if "filename" not in df.columns or "label" not in df.columns:
        raise ValueError("Training CSV must contain 'filename' and 'label' columns.")

    all_files_map = build_all_files_map(data_root)
    df["abs_path"] = df["filename"].apply(lambda x: all_files_map.get(os.path.basename(str(x))))
    df = df.dropna(subset=["abs_path"]).reset_index(drop=True)
    return df


def build_label_mapping(df: pd.DataFrame) -> Tuple[List[str], int]:
    class_names = sorted(df["label"].unique().tolist())
    label_to_index = {label: idx for idx, label in enumerate(class_names)}
    df["label_idx"] = df["label"].map(label_to_index)
    return class_names, len(class_names)


def split_train_val(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 123,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label_idx"],
    )


def make_dataset(
    paths,
    labels,
    img_size: Tuple[int, int],
    num_classes: int,
    batch_size: int,
    training: bool,
    cache_in_memory: bool = True,
    cache_path: Optional[str] = None,
):
    def load_and_preprocess_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32)
        return img, tf.one_hot(label, num_classes)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    if cache_in_memory:
        ds = ds.cache()
    elif cache_path:
        ds = ds.cache(cache_path)

    if training:
        ds = ds.shuffle(1000)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
