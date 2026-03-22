import kagglehub
import logging
import shutil
from pathlib import Path
from typing import Final


DATASET_ID: Final[str] = "purusinghvi/email-spam-classification-dataset"
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def download_dataset(dataset_id: str) -> Path:
    logger.info("Starting dataset download from KaggleHub")
    path = kagglehub.dataset_download(dataset_id)
    logger.info("Download complete: %s", path)
    return Path(path)


def dataset_exists(path: Path) -> bool:
    exists = path.exists() and any(path.iterdir())

    if exists:
        logger.debug("Dataset exists at %s", path)
    else:
        logger.debug("Dataset not found at %s", path)

    return exists


def copy_tree_idempotent(src: Path, dest: Path) -> None:
    logger.info("Copying dataset from %s to %s", src, dest)

    dest.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        target = dest / item.name

        if target.exists():
            logger.debug("Skipping existing item: %s", target)
            continue

        if item.is_dir():
            shutil.copytree(item, target)
            logger.debug("Copied directory: %s", target)
        else:
            shutil.copy2(item, target)
            logger.debug("Copied directory: %s", target)

    logger.info("Dataset copy complete")


def normalize_dataset_filename(data_dir: Path) -> None:
    desired = data_dir / "email-spam-classification-dataset.csv"
    original = data_dir / "combined_data.csv"

    if desired.exists():
        logger.debug("Normalized dataset already exists: %s", desired)
        return

    if original.exists():
        original.rename(desired)
        logger.info("Renamed %s -> %s", original.name, desired.name)
        return

    raise FileNotFoundError("Expected dataset file not found (combined_data.csv)")


def ensure_dataset() -> Path:
    if dataset_exists(DATA_DIR):
        logger.info("Dataset already present at: %s", DATA_DIR)

        normalize_dataset_filename(DATA_DIR)

        return DATA_DIR

    logger.info("Downloading dataset via kagglehub...")
    cache_path: Path = download_dataset(DATASET_ID)

    logger.info("Copying dataset into project data directory...")
    copy_tree_idempotent(cache_path, DATA_DIR)

    normalize_dataset_filename(DATA_DIR)

    logger.info("Dataset ready at: %s", DATA_DIR)

    return DATA_DIR


def main() -> None:
    ensure_dataset()


if __name__ == "__main__":
    main()
