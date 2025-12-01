# EEG 2025 Challenge

Implementation for the [NeurIPS 2025 EEG Foundation Challenge](https://eeg2025.github.io/).

## Challenges

**Challenge 1**: Predict response time from CCD EEG data (RMSE metric)

**Challenge 2**: Predict externalizing factor from EEG (MAE metric)

Both tasks: regression on 129-channel EEG, 2s windows @ 100Hz.

## Dataset

**HBN-EEG** - ~3,000 participants, auto-downloaded via [EEGDash](https://github.com/eeg2025/eegdash).

## Models

Default: **EEGNeX** (state-of-the-art). Alternatives: EEGNet, EEGConformer (from Braindecode).

## Lighter Features Demonstrated

- **Braindecode integration** - EEGNeX, EEGNet models
- **EEGDash data loading** - automatic download and preprocessing
- **`%` raw references** - shared config values via `vars::`
- **MetricCollection** - MAE, RMSE tracking

## Requirements

```bash
pip install eegdash braindecode mne
```

## Usage

```bash
cd projects/eeg

# Challenge 1
uv run --project ../.. lighter fit configs/challenge1.yaml

# Challenge 2
lighter fit configs/challenge2.yaml

# Full training
lighter fit configs/challenge1.yaml trainer::fast_dev_run=false
```

## Submission

```bash
python -m projects.eeg.submission --export --ckpt1 path/to/c1.ckpt --ckpt2 path/to/c2.ckpt
python -m projects.eeg.submission --zip
```

Submit to [CodaBench](https://www.codabench.org/competitions/9975/).

## References

- [EEG 2025 Challenge](https://eeg2025.github.io/)
- [Braindecode](https://braindecode.org/)
- [EEGDash](https://github.com/eeg2025/eegdash)
