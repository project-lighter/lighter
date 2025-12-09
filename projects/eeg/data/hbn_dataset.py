"""HBN-EEG Dataset loading for EEG 2025 Challenge.

This module provides dataset classes for both challenges using EEGDash.

Challenge 1: Cross-Task Transfer Learning
- Task: Predict response time from Contrast Change Detection (CCD) EEG
- Target: rt_from_stimulus (response time)
- Preprocessing: Window extraction around stimulus events

Challenge 2: Externalizing Factor Prediction
- Task: Predict psychopathology scores from EEG
- Target: externalizing factor (p_factor in dataset)
- Preprocessing: Fixed-length windows with random cropping

Dataset:
- HBN-EEG: ~3000 subjects with EEG recordings
- 129 EEG channels (EGI system)
- Multiple paradigms: CCD, Resting State, etc.

Requirements:
    pip install eegdash braindecode mne
"""

import math
import random
from pathlib import Path

import numpy as np
from braindecode.datasets.base import BaseConcatDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_fixed_length_windows,
    create_windows_from_events,
    preprocess,
)
from eegdash.dataset import EEGChallengeDataset
from eegdash.hbn.windows import (
    add_aux_anchors,
    add_extras_columns,
    annotate_trials_with_target,
    keep_only_recordings_with,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from torch.utils.data import Dataset

# Subjects to exclude (corrupted/incomplete data)
EXCLUDED_SUBJECTS = [
    "NDARWV769JM7",
    "NDARME789TD2",
    "NDARUA442ZVF",
    "NDARJP304NK1",
    "NDARTY128YLU",
    "NDARDW550GU6",
    "NDARLD243KRE",
    "NDARUJ292JXV",
    "NDARBA381JGH",
]

# Default parameters
DEFAULT_SFREQ = 100  # Sampling frequency after resampling
DEFAULT_N_CHANS = 129  # Number of EEG channels


class HBNDatasetChallenge1(Dataset):
    """HBN-EEG Dataset for Challenge 1: Cross-Task Transfer Learning.

    Predicts response time from Contrast Change Detection (CCD) EEG data.

    Args:
        data_dir: Directory for dataset cache.
        release: Data release version ('R5', 'R6', etc.).
        mini: Whether to use mini version for testing.
        split: Data split ('train', 'val', 'test').
        seed: Random seed for reproducibility.
        epoch_len_s: Length of each epoch in seconds.
        shift_after_stim: Time shift after stimulus onset.
        window_len: Window length in seconds.
    """

    def __init__(
        self,
        data_dir: str = "data",
        release: str = "R5",
        mini: bool = False,
        split: str = "train",
        seed: int = 2025,
        epoch_len_s: float = 2.0,
        shift_after_stim: float = 0.5,
        window_len: float = 2.0,
        valid_frac: float = 0.1,
        test_frac: float = 0.1,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.sfreq = DEFAULT_SFREQ

        # Load dataset
        dataset = EEGChallengeDataset(
            task="contrastChangeDetection",
            release=release,
            cache_dir=self.data_dir,
            mini=mini,
        )

        # Preprocessing: annotate trials with response time target
        transformation = [
            Preprocessor(
                annotate_trials_with_target,
                target_field="rt_from_stimulus",
                epoch_length=epoch_len_s,
                require_stimulus=True,
                require_response=True,
                apply_on_array=False,
            ),
            Preprocessor(add_aux_anchors, apply_on_array=False),
        ]
        preprocess(dataset, transformation, n_jobs=1)

        # Create windows around stimulus events
        anchor = "stimulus_anchor"
        dataset = keep_only_recordings_with(anchor, dataset)

        self.windows_dataset = create_windows_from_events(
            dataset,
            mapping={anchor: 0},
            trial_start_offset_samples=int(shift_after_stim * self.sfreq),
            trial_stop_offset_samples=int((shift_after_stim + window_len) * self.sfreq),
            window_size_samples=int(epoch_len_s * self.sfreq),
            window_stride_samples=self.sfreq,
            preload=True,
        )

        # Add extra columns with target values
        self.windows_dataset = add_extras_columns(
            self.windows_dataset,
            dataset,
            desc=anchor,
            keys=(
                "target",
                "rt_from_stimulus",
                "rt_from_trialstart",
                "stimulus_onset",
                "response_onset",
                "correct",
                "response_type",
            ),
        )

        # Split by subject
        meta = self.windows_dataset.get_metadata()
        subjects = [s for s in meta["subject"].unique() if s not in EXCLUDED_SUBJECTS]

        train_subj, valid_test_subj = train_test_split(
            subjects,
            test_size=(valid_frac + test_frac),
            random_state=check_random_state(seed),
            shuffle=True,
        )
        valid_subj, test_subj = train_test_split(
            valid_test_subj,
            test_size=test_frac / (valid_frac + test_frac),
            random_state=check_random_state(seed + 1),
            shuffle=True,
        )

        # Select appropriate split
        split_map = {"train": train_subj, "val": valid_subj, "test": test_subj}
        target_subjects = set(split_map[split])

        subject_split = self.windows_dataset.split("subject")
        split_datasets = [subject_split[s] for s in subject_split if s in target_subjects]

        self.dataset = BaseConcatDataset(split_datasets) if split_datasets else None

        # Cache metadata for efficient access
        self._metadata = self.dataset.get_metadata() if self.dataset else None

    def __len__(self) -> int:
        return len(self.dataset) if self.dataset else 0

    def __getitem__(self, idx: int) -> tuple:
        X, y, window_info = self.dataset[idx]
        # Get target from cached metadata
        target = float(self._metadata.iloc[idx].get("rt_from_stimulus", y))
        # Convert to float32 for MPS compatibility (MPS doesn't support float64)
        X = X.astype(np.float32)
        return X, np.float32(target)


class HBNDatasetChallenge2(Dataset):
    """HBN-EEG Dataset for Challenge 2: Externalizing Factor Prediction.

    Predicts psychopathology scores (externalizing factor) from EEG.

    Args:
        data_dir: Directory for dataset cache.
        releases: List of data release versions.
        task: EEG paradigm to use.
        mini: Whether to use mini version for testing.
        split: Data split ('train', 'val', 'test').
        seed: Random seed for reproducibility.
        window_size_s: Window size in seconds.
        window_stride_s: Window stride in seconds.
        crop_size_s: Crop size for random cropping.
        target_name: Name of target variable in metadata.
    """

    def __init__(
        self,
        data_dir: str = "data",
        releases: list[str] | None = None,
        task: str = "contrastChangeDetection",
        mini: bool = False,
        split: str = "train",
        seed: int = 2025,
        window_size_s: float = 4.0,
        window_stride_s: float = 2.0,
        crop_size_s: float = 2.0,
        target_name: str = "p_factor",
        valid_frac: float = 0.1,
        test_frac: float = 0.1,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.sfreq = DEFAULT_SFREQ
        self.crop_size_samples = int(crop_size_s * self.sfreq)
        self.target_name = target_name
        self.rng = random.Random(seed)

        if releases is None:
            releases = ["R5"]

        # Load datasets from multiple releases
        all_datasets = []
        for release in releases:
            ds = EEGChallengeDataset(
                release=release,
                task=task,
                mini=mini,
                description_fields=[
                    "subject",
                    "session",
                    "run",
                    "task",
                    "age",
                    "gender",
                    "sex",
                    "p_factor",
                ],
                cache_dir=self.data_dir,
            )
            all_datasets.append(ds)

        combined = BaseConcatDataset(all_datasets) if len(all_datasets) > 1 else all_datasets[0]

        # Filter valid recordings
        valid_datasets = []
        for ds in combined.datasets:
            # Check: valid subject, enough samples, correct channels, valid p_factor
            if (
                ds.description.subject not in EXCLUDED_SUBJECTS
                and ds.raw.n_times >= int(window_size_s * self.sfreq)
                and len(ds.raw.ch_names) == DEFAULT_N_CHANS
                and not math.isnan(ds.description.get("p_factor", float("nan")))
            ):
                valid_datasets.append(ds)

        if not valid_datasets:
            self.dataset = None
            self._window_datasets = []
            return

        filtered = BaseConcatDataset(valid_datasets)

        # Create fixed-length windows
        windows_ds = create_fixed_length_windows(
            filtered,
            window_size_samples=int(window_size_s * self.sfreq),
            window_stride_samples=int(window_stride_s * self.sfreq),
            drop_last_window=True,
        )

        # Get unique subjects and split
        all_subjects = list({ds.description.subject for ds in filtered.datasets})
        all_subjects = [s for s in all_subjects if s not in EXCLUDED_SUBJECTS]

        train_subj, valid_test_subj = train_test_split(
            all_subjects,
            test_size=(valid_frac + test_frac),
            random_state=check_random_state(seed),
            shuffle=True,
        )
        valid_subj, test_subj = train_test_split(
            valid_test_subj,
            test_size=test_frac / (valid_frac + test_frac),
            random_state=check_random_state(seed + 1),
            shuffle=True,
        )

        split_map = {"train": train_subj, "val": valid_subj, "test": test_subj}
        target_subjects = set(split_map[split])

        # Filter windows by subject
        self._window_datasets = []
        for wds in windows_ds.datasets:
            if wds.description.subject in target_subjects:
                self._window_datasets.append(wds)

        self.dataset = BaseConcatDataset(self._window_datasets) if self._window_datasets else None

    def __len__(self) -> int:
        if not self._window_datasets:
            return 0
        return sum(len(wds) for wds in self._window_datasets)

    def __getitem__(self, idx: int) -> tuple:
        # Find which window dataset and local index
        cumsum = 0
        for wds in self._window_datasets:
            if idx < cumsum + len(wds):
                local_idx = idx - cumsum
                X, _, crop_inds = wds[local_idx]

                # Get target
                target = float(wds.description.get(self.target_name, 0.0))

                # Random crop
                i_window, i_start, i_stop = crop_inds
                window_len = i_stop - i_start
                if window_len > self.crop_size_samples:
                    start_offset = self.rng.randint(0, window_len - self.crop_size_samples)
                    X = X[:, start_offset : start_offset + self.crop_size_samples]

                # Convert to float32 for MPS compatibility
                X = X.astype(np.float32)
                return X, np.float32(target)
            cumsum += len(wds)

        raise IndexError(f"Index {idx} out of range")


def get_train_val_test_split(
    subjects: list[str],
    valid_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 2025,
) -> tuple[list[str], list[str], list[str]]:
    """Split subjects into train/val/test sets.

    Args:
        subjects: List of subject IDs.
        valid_frac: Fraction for validation.
        test_frac: Fraction for test.
        seed: Random seed.

    Returns:
        Tuple of (train_subjects, val_subjects, test_subjects).
    """
    # Remove excluded subjects
    subjects = [s for s in subjects if s not in EXCLUDED_SUBJECTS]

    train_subj, valid_test_subj = train_test_split(
        subjects,
        test_size=(valid_frac + test_frac),
        random_state=check_random_state(seed),
        shuffle=True,
    )
    valid_subj, test_subj = train_test_split(
        valid_test_subj,
        test_size=test_frac / (valid_frac + test_frac),
        random_state=check_random_state(seed + 1),
        shuffle=True,
    )

    return train_subj, valid_subj, test_subj
