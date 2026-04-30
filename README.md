# BirdCLEF+ 2026

Code to train, soup, export, ensemble-tune, and run inference for the
BirdCLEF+ 2026 Pantanal soundscape competition. The deployed system is a
6-model ensemble of timm CNN backbones (EfficientNetV2-S, eca_nfnet_l0,
EfficientNet-B0) under two log-mel configurations, exported to OpenVINO
FP16 and run on Kaggle's CPU-only kernel.

## Layout

    release/
      src/
        data/         dataset, augmentations, mel-spectrogram cache, samplers
        models/       timm-backbone wrapper, SED head, classifier, factory
        training/     trainer, cosine scheduler with warmup
        losses/       focal-BCE with label smoothing
        utils/        macro-AUC metric, seeding, CSV logger
      scripts/
        train.py                       single-fold training
        soup_models.py                 average a 5-fold group's weights
        export_models.py               PyTorch to ONNX to OpenVINO FP16
        tune_soundscape_ensemble.py    pick ensemble weights against labelled soundscapes
        tune_per_class_soundscape.py   per-class variant of the above
        generate_pseudolabels.py       run an ensemble on unlabelled audio
        cache_pseudolabels.py          turn a pseudo-CSV into an HDF5
        merge_labeled_into_pseudo_h5.py append labelled keys to a pseudo HDF5
      kaggle/
        inference_notebook.py          CPU/OpenVINO inference for the kernel
      configs/
        base.yaml                      default training hyperparameters
        data/                          mel configs (cfg A and cfg B)
        model/                         backbone configs

## Data layout assumed by the scripts

Configurable via `configs/data/*.yaml`. The expected layout one level above
the project root is:

    data/
      train_audio/                       labelled Xeno-Canto clips, one folder per species
      train_soundscapes/                 ~10,600 one-minute Pantanal recordings
      train_soundscapes_labels.csv       labels for 66 of the soundscape files
      sample_submission.csv              defines the 234-class column order
      taxonomy.csv                       species code to common name
      cache/
        all_config_a.h5                  pre-computed log-mel for cfg A
        all_config_b.h5                  pre-computed log-mel for cfg B
        soundscapes_config_a.h5          pre-computed log-mel for the 66 labelled soundscapes
        soundscapes_config_b.h5          same, cfg B
      pseudo_labels/                     optional, output of generate_pseudolabels.py

## Quick start

These commands assume your working directory contains both `release/` and
`data/` side by side.

### Train a single fold

    python release/scripts/train.py \
      --config release/configs/base.yaml \
      --config release/configs/model/efficientnet_v2s.yaml \
      --config release/configs/data/default.yaml \
      training.fold=0 \
      training.epochs=150 \
      training.output_dir=runs/effv2s_cfga_fold0

To use the labelled soundscapes for early stopping during training, append:

    soundscape_val.soundscape_dir=data/train_soundscapes \
    soundscape_val.labels_csv=data/train_soundscapes_labels.csv \
    soundscape_val.sample_sub_csv=data/sample_submission.csv \
    data.pseudo_h5=data/cache/soundscapes_config_a.h5

`data.pseudo_h5` concatenates the labelled soundscape windows into the
training set as in-domain examples.

### Soup five folds and export

    python release/scripts/soup_models.py \
      --runs-root runs/ \
      --output-root exports/souped/

    python release/scripts/export_models.py \
      --runs-root exports/souped/

`exports/souped/<name>/model_fp16/model.xml` is the OpenVINO file used at
inference time.

### Tune ensemble weights against the labelled soundscapes

    python release/scripts/tune_soundscape_ensemble.py \
      --exports-dir exports/souped/ \
      --output soundscape_ensemble_weights.json

The output JSON contains per-model solo soundscape AUC, the equal-weight
ensemble AUC, and the optimised weights. Use it to select your top-N
models for the kernel.

### Run inference

`kaggle/inference_notebook.py` is the script form of the live Kaggle kernel
cell. Edit the `MODELS` list and `MODEL_WEIGHTS` dict at the top of the file
to match your top-N selection, then:

    python release/kaggle/inference_notebook.py \
      --test-dir data/test_soundscapes \
      --sample-sub data/sample_submission.csv \
      --models-dir exports/souped/ \
      --output submission.csv

## Where the methodology lives in code

Two files do most of the interesting work:

* `src/training/trainer.py` — the training loop. `_validate_soundscapes`
  computes macro-AUC over the 66 labelled soundscapes at the end of each
  epoch and is used as the early-stopping signal when `soundscape_val.*`
  is configured.
* `scripts/tune_soundscape_ensemble.py` — runs each souped model on the 66
  labelled soundscapes via the same OpenVINO path used at submission time,
  then optimises ensemble weights to maximise soundscape macro-AUC. Use
  this instead of OOF-based ensemble tuning.

## Install

Python 3.10+ is required. From the project root:

    pip install -r release/requirements.txt

The pinned versions are in `requirements.txt`. The Kaggle kernel installs
the OpenVINO wheel from a bundled `wheels/openvino*.whl` so that no
internet access is required at submission time.

## Contact

Kanta Saito — University of Hawai\`i at Manoa — kantas@hawaii.edu
