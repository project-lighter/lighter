"""Competition submission template for EEG 2025 Challenge.

This module provides the Submission class required by CodaBench.

Submission format:
- submission.py (this file)
- weights_challenge_1.pt (trained weights for Challenge 1)
- weights_challenge_2.pt (trained weights for Challenge 2)

All files should be in a flat zip archive for submission.

Usage:
    # After training, export weights and create submission:
    python -m projects.eeg.submission --export

    # Test submission locally:
    python -m projects.eeg.submission --test
"""

import argparse
import zipfile
from pathlib import Path

import torch
from braindecode.models import EEGNeX

# Default parameters matching the challenge
DEFAULT_SFREQ = 100
DEFAULT_N_CHANS = 129
DEFAULT_N_TIMES = 200  # 2 seconds at 100 Hz


def resolve_path(name: str = "model_file_name") -> str:
    """Resolve model weight file path across different environments.

    CodaBench may place files in different locations depending on
    the execution environment.

    Args:
        name: Name of the weight file.

    Returns:
        Resolved path to the file.

    Raises:
        FileNotFoundError: If file cannot be found.
    """
    # Check various possible locations
    candidates = [
        Path(f"/app/input/res/{name}"),
        Path(f"/app/input/{name}"),
        Path(name),
        Path(__file__).parent / name,
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    raise FileNotFoundError(f"Could not find {name} in expected locations: {[str(c) for c in candidates]}")


class Submission:
    """Competition submission class for EEG 2025 Challenge.

    This class is instantiated by the CodaBench evaluation system
    and used to load models for both challenges.

    Args:
        SFREQ: Sampling frequency (100 Hz for HBN-EEG).
        DEVICE: Device to load models on ('cuda' or 'cpu').
    """

    def __init__(self, SFREQ: int, DEVICE: str) -> None:
        self.sfreq = SFREQ
        self.device = DEVICE
        self.n_times = int(2 * SFREQ)  # 2 second windows

    def get_model_challenge_1(self) -> torch.nn.Module:
        """Load trained model for Challenge 1 (Response Time Prediction).

        Returns:
            Loaded EEGNeX model ready for inference.
        """
        model = EEGNeX(
            n_chans=DEFAULT_N_CHANS,
            n_outputs=1,
            n_times=self.n_times,
            sfreq=self.sfreq,
        ).to(self.device)

        # Load trained weights
        weights_path = resolve_path("weights_challenge_1.pt")
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)

        # Handle potential 'model.' prefix from wrapper
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        model.eval()

        return model

    def get_model_challenge_2(self) -> torch.nn.Module:
        """Load trained model for Challenge 2 (Externalizing Factor).

        Returns:
            Loaded EEGNeX model ready for inference.
        """
        model = EEGNeX(
            n_chans=DEFAULT_N_CHANS,
            n_outputs=1,
            n_times=self.n_times,
            sfreq=self.sfreq,
        ).to(self.device)

        # Load trained weights
        weights_path = resolve_path("weights_challenge_2.pt")
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)

        # Handle potential 'model.' prefix from wrapper
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        model.eval()

        return model


def export_weights(
    checkpoint_path_1: str | None = None,
    checkpoint_path_2: str | None = None,
    output_dir: str = "submission",
) -> None:
    """Export trained model weights for submission.

    Args:
        checkpoint_path_1: Path to Challenge 1 checkpoint (Lightning format).
        checkpoint_path_2: Path to Challenge 2 checkpoint (Lightning format).
        output_dir: Directory to save exported weights.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for challenge, ckpt_path in [
        (1, checkpoint_path_1),
        (2, checkpoint_path_2),
    ]:
        if ckpt_path is None:
            print(f"Skipping Challenge {challenge} - no checkpoint provided")
            continue

        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            print(f"Warning: Checkpoint not found: {ckpt_path}")
            continue

        # Load Lightning checkpoint
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Extract network weights (remove 'network.' prefix from LighterModule)
        network_state = {}
        for key, value in state_dict.items():
            if key.startswith("network."):
                # Remove 'network.' prefix
                new_key = key[8:]
                # Also handle 'model.' prefix from wrapper
                if new_key.startswith("model."):
                    new_key = new_key[6:]
                network_state[new_key] = value

        # Save weights
        output_file = output_path / f"weights_challenge_{challenge}.pt"
        torch.save(network_state, output_file)
        print(f"Exported Challenge {challenge} weights to: {output_file}")


def create_submission_zip(
    output_dir: str = "submission",
    zip_name: str = "submission.zip",
) -> None:
    """Create submission zip file.

    Args:
        output_dir: Directory containing submission files.
        zip_name: Name of the output zip file.
    """
    output_path = Path(output_dir)
    zip_path = output_path / zip_name

    required_files = [
        "submission.py",
        "weights_challenge_1.pt",
        "weights_challenge_2.pt",
    ]

    # Copy submission.py to output directory
    import shutil

    shutil.copy(__file__, output_path / "submission.py")

    # Create zip
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename in required_files:
            filepath = output_path / filename
            if filepath.exists():
                zf.write(filepath, filename)
                print(f"Added to zip: {filename}")
            else:
                print(f"Warning: Missing file: {filename}")

    print(f"\nCreated submission: {zip_path}")


def test_submission(output_dir: str = "submission") -> None:
    """Test submission locally.

    Args:
        output_dir: Directory containing submission files.
    """
    import sys

    sys.path.insert(0, str(Path(output_dir)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on device: {device}")

    try:
        sub = Submission(SFREQ=DEFAULT_SFREQ, DEVICE=device)

        # Test Challenge 1
        print("\nTesting Challenge 1 model...")
        model1 = sub.get_model_challenge_1()
        x = torch.randn(2, DEFAULT_N_CHANS, DEFAULT_N_TIMES).to(device)
        with torch.no_grad():
            y = model1(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {y.shape}")
        print(f"  Output: {y.squeeze().tolist()}")

        # Test Challenge 2
        print("\nTesting Challenge 2 model...")
        model2 = sub.get_model_challenge_2()
        with torch.no_grad():
            y = model2(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {y.shape}")
        print(f"  Output: {y.squeeze().tolist()}")

        print("\nSubmission test PASSED!")

    except Exception as e:
        print(f"\nSubmission test FAILED: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG 2025 Challenge Submission Tools")
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export trained weights for submission",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test submission locally",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Create submission zip file",
    )
    parser.add_argument(
        "--ckpt1",
        type=str,
        default=None,
        help="Path to Challenge 1 checkpoint",
    )
    parser.add_argument(
        "--ckpt2",
        type=str,
        default=None,
        help="Path to Challenge 2 checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="submission",
        help="Output directory for submission files",
    )

    args = parser.parse_args()

    if args.export:
        export_weights(
            checkpoint_path_1=args.ckpt1,
            checkpoint_path_2=args.ckpt2,
            output_dir=args.output_dir,
        )

    if args.zip:
        create_submission_zip(output_dir=args.output_dir)

    if args.test:
        test_submission(output_dir=args.output_dir)

    if not any([args.export, args.test, args.zip]):
        parser.print_help()
