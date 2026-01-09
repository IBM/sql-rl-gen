import argparse
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional


def download(url: str, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    def reporthook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100.0, downloaded * 100.0 / total_size)
        sys.stdout.write("\rDownloading: {:.1f}%".format(pct))
        sys.stdout.flush()

    print("Downloading:", url)
    urllib.request.urlretrieve(url, str(dst_path), reporthook=reporthook)
    sys.stdout.write("\n")
    sys.stdout.flush()


def safe_extract_zip(zip_path: Path, extract_dir: Path, force: bool) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    extract_dir_resolved = extract_dir.resolve()
    with zipfile.ZipFile(str(zip_path)) as zf:
        members = zf.infolist()
        for m in members:
            target = extract_dir / m.filename
            target_resolved = target.resolve()
            common = os.path.commonpath([str(extract_dir_resolved), str(target_resolved)])
            if common != str(extract_dir_resolved):
                raise RuntimeError("Unsafe path in zip: {}".format(m.filename))
        if force:
            zf.extractall(str(extract_dir))
        else:
            for m in members:
                target = extract_dir / m.filename
                if target.exists():
                    continue
                zf.extract(m, str(extract_dir))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download (optional) and extract Spider dataset zip into data_preprocess/data/"
    )
    parser.add_argument(
        "--zip-path",
        type=str,
        default="data_preprocess/data/spider.zip",
        help="Path to spider.zip",
    )
    parser.add_argument(
        "--extract-dir",
        type=str,
        default="data_preprocess/data",
        help="Directory to extract into (should contain spider/ after extraction)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Optional URL to download spider.zip if it does not exist locally",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing extracted files",
    )
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    extract_dir = Path(args.extract_dir)
    url: Optional[str] = args.url

    if not zip_path.exists():
        if url is None:
            print("ERROR: {} not found.".format(zip_path))
            print("Please download the dataset bundle and place spider.zip at that path, or pass --url.")
            sys.exit(2)
        download(url, zip_path)

    print("Extracting:", zip_path, "->", extract_dir)
    safe_extract_zip(zip_path, extract_dir, force=args.force)

    expected_db_dir = extract_dir / "spider" / "database"
    if not expected_db_dir.exists():
        print("WARNING: Expected directory not found:", expected_db_dir)
    else:
        print("OK: Spider DB directory:", expected_db_dir)


if __name__ == "__main__":
    main()
