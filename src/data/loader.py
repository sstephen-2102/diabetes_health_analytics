"""CSV loading and optional Kaggle acquisition; no raw-data overwrites."""
from pathlib import Path
import csv
import shutil
import hashlib
import pandas as pd
from src.common.exceptions import DataLoadError
from src.common.config import EXPECTED_COLUMNS
from src.data.validation import validate_dataset

def download_uci_dataset(output_path: str, dataset_id: int = 891) -> str:
    """Download a UCI CDC dataset data, verify it against the reference, save as CSV.
    Refuses to overwrite an existing file, returns CSV path."""
    destination = Path(output_path)
    if destination.exists():
        return str(destination)
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=dataset_id)
        data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    except Exception as exc:
        raise DataLoadError(f'UCI acquisition failed: {exc}') from exc

    checks = validate_dataset(data, EXPECTED_COLUMNS, "Diabetes_binary")
    if not checks["is_valid"] or not checks["matches_reference"]:
        raise DataLoadError("Downloaded dataset differs from project reference")

    try:
        destination.parent.mkdir(parents=True,exist_ok=True)
        data.to_csv(destination, index=False)
    except OSError as exc:
        raise DataLoadError(f'Cannot save UCI dataset: {exc}') from exc
    return str(destination)
    
def load_dataset(file_path: str) -> pd.DataFrame:
    """Read a non-empty numeric CSV preserving rows; wrap failures in DataLoadError.

    Schema validation is separate. Result attrs record raw_v1 and source filename.
    """
    try:
        path = Path(file_path)
        if not path.is_file():
            raise ValueError('File does not exist or is not a regular file.')
        with path.open(encoding='utf-8-sig',newline='') as stream:
            header = next(csv.reader(stream))
        if len(header) != len(set(header)):
            raise ValueError('CSV has duplicate column names.')
        data = pd.read_csv(path,on_bad_lines='error')
        if data.empty or data.shape[1] < 2:
            raise ValueError('CSV is empty or has no usable table schema.')
        if not all(pd.api.types.is_numeric_dtype(data[c]) for c in data):
            raise ValueError('Expected numeric survey columns; found text or malformed CSV.')
    except (OSError,ValueError,TypeError,StopIteration,UnicodeError,pd.errors.ParserError) as exc:
        raise DataLoadError(f'Cannot load dataset: {exc}') from exc
    data.attrs.update(dataset_variant='raw_v1',source_file=path.name)
    return data

def download_kaggle_dataset(dataset_id: str, output_directory: str) -> str:
    """Download CSVs with kagglehub; return directory; refuse changed-file overwrites.

    Dependency is imported only on this path. Callers select the required CSV.
    """
    try:
        if not isinstance(dataset_id,str) or dataset_id.count('/') != 1:
            raise ValueError('Expected a Kaggle owner/dataset identifier.')
        import kagglehub
        source = Path(kagglehub.dataset_download(dataset_id))
        files = list(source.rglob('*.csv'))
        if not files:
            raise ValueError('Downloaded dataset contains no CSV files.')
        destination = Path(output_directory)
        if source.resolve() == destination.resolve():
            return str(destination)
        destination.mkdir(parents=True,exist_ok=True)
        for file in files:
            target = destination / file.relative_to(source)
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if hashlib.sha256(file.read_bytes()).digest() != hashlib.sha256(target.read_bytes()).digest():
                    raise ValueError(f'Refusing to overwrite raw file {target.name}.')
            else:
                shutil.copyfile(file,target)
        return str(destination)
    except Exception as exc:
        raise DataLoadError(f'Kaggle acquisition failed: {exc}') from exc
