# Butterfly Classifier App

This project is split into two independent phases:

1. Training (`train.py`) - run occasionally to produce model artifacts.
2. Inference UI (`app.py`) - run anytime to predict classes from uploaded images.

## Project structure

- `train.py`: trains model and exports artifacts.
- `app.py`: Streamlit web UI for predictions.
- `src/data.py`: dataset loading, path fixing, split, tf.data pipeline.
- `src/model.py`: model definition and training callbacks.
- `src/inference.py`: preprocessing and prediction helpers.
- `artifacts/`: exported files:
  - `butterfly_model.keras`
  - `class_names.json`
  - `train_config.json`
  - `training_accuracy.png`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train once

If `Training_set.csv` is in this folder, training uses local data.
Otherwise it tries to download from KaggleHub.

```bash
python train.py
```

Optional:

```bash
python train.py --data-root /path/to/dataset --epochs 15 --batch-size 32
```

## Run UI for predictions

After artifacts are created:

```bash
streamlit run app.py
```

Then open the shown local URL and upload a butterfly image.

## Notes

- Inference uses the same MobileNetV3 preprocessing as training.
- Label mapping is loaded from `artifacts/class_names.json` to keep predictions consistent.
