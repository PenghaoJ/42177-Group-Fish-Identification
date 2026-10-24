"""Download the two official ImageNet initial checkpoints (not trained FFE weights)."""
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

URLS = {
    "resnet18-f37072fd.pth": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "yolov8n-cls.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-cls.pt",
}
HASHES = {
    "resnet18-f37072fd.pth": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
    "yolov8n-cls.pt": "11fa19f2aea79bc960d680a13f82f22105982b325eb9e17a4a5e1a9f8245980a",
}


def download(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        path = folder / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".download")
            with urlopen(url, timeout=120) as source, temporary.open("wb") as destination:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    destination.write(chunk)
            temporary.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != HASHES[name]:
            raise ValueError(f"Official checkpoint SHA-256 mismatch: {name}")
        print(f"{name}: {path.stat().st_size} bytes, sha256={digest}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".training/pretrained"))
    download(parser.parse_args().output)
