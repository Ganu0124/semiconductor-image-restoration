import os
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "madesh-2006/semiconductor-real-dataset"
DATASET_DIR = Path("data/real")


def main():
    token = os.getenv("HF_TOKEN")

    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set."
        )

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print("Downloading real semiconductor dataset")
    print("========================================")
    print(f"Repository: {REPO_ID}")
    print(f"Destination: {DATASET_DIR.resolve()}")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(DATASET_DIR),
        token=token,
    )

    print("========================================")
    print("Dataset download completed")
    print("========================================")

    expected_dirs = [
        DATASET_DIR / "train" / "clean",
        DATASET_DIR / "train" / "degraded",
        DATASET_DIR / "val" / "clean",
        DATASET_DIR / "val" / "degraded",
        DATASET_DIR / "test" / "degraded",
    ]

    for directory in expected_dirs:
        if not directory.exists():
            raise RuntimeError(
                f"Expected dataset directory not found: {directory}"
            )

    total_files = sum(
        1 for path in DATASET_DIR.rglob("*.npy")
    )

    print(f"Total .npy files found: {total_files}")

    if total_files == 0:
        raise RuntimeError(
            "Dataset downloaded, but no .npy files were found."
        )

    print("Dataset is ready.")


if __name__ == "__main__":
    main()