"""Fine-tune one pretrained CNN using training/validation data only.

The test set is intentionally inaccessible to this training entry point.
Every run saves its configuration, epoch history, best weights and validation
predictions. Multiple seeds are separate runs of the same fixed protocol.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, confusion_matrix

from .cnn import build_network, choose_device, letterbox
from .data import LABELS, LIMITATION, read_rows, sha256_file, write_json, write_csv, checked_training_rows, image_path
from .metrics import save_predictions


def _cached_image(args):
    path, size = args
    with Image.open(path) as image:
        return np.asarray(letterbox(image, size)).copy()


def build_cache(root, split, folder, size=224):
    if split not in ("train", "validation"):
        raise ValueError("Training must not read test images")
    rows = read_rows(root, split)
    manifest_sha = sha256_file(root / "manifests" / "ffe3" / f"{split}.csv")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{split}_{size}_{manifest_sha[:12]}.npy"
    if not path.exists():
        temporary = path.with_suffix(".tmp.npy")
        array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8, shape=(len(rows), size, size, 3))
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i, image in enumerate(pool.map(_cached_image, [(image_path(root, r), size) for r in rows])):
                array[i] = image
        array.flush()
        del array
        temporary.replace(path)
        print(f"Built {split} letterbox cache: {len(rows)} images", flush=True)
    return path, rows, manifest_sha


class FishDataset(Dataset):
    def __init__(self, cache, rows, augment=False):
        self.cache_path = str(cache)
        self.rows = rows
        self.array = None
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self.array is None:
            self.array = np.load(self.cache_path, mmap_mode="r")
        image = Image.fromarray(self.array[index])
        if self.augment:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
            if random.random() < 0.5:
                image = ImageOps.flip(image)
            # Keep the complete eye crop; preserve colour cues for freshness.
            image = image.rotate(random.uniform(-12, 12), resample=Image.Resampling.BILINEAR, expand=True)
            image = letterbox(image, self.array.shape[1])
        tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div_(255)
        return tensor, int(self.rows[index]["label_id"])


def worker_seed(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def atomic_save(payload, path):
    temporary = path.with_suffix(".tmp.pt")
    torch.save(payload, temporary)
    temporary.replace(path)


def epoch(model, loader, criterion, device, optimizer=None, warmup=False):
    training = optimizer is not None
    model.train(training)
    if warmup:
        model.base.eval()
        model.head().train()
    total, loss_sum, correct = 0, 0.0, 0
    predictions, labels, probabilities = [], [], []
    started = time.perf_counter()
    with torch.set_grad_enabled(training):
        for step, (x, y) in enumerate(loader, 1):
            x, y = x.to(device), y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            # Blackwell supports bfloat16; gradients remain float32, no scaler needed.
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and training):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            correct += int((logits.argmax(1) == y).sum())
            total += len(y)
            if not training:
                labels.extend(y.cpu().tolist())
                probabilities.extend(logits.detach().softmax(1).cpu().tolist())
                predictions.extend(logits.argmax(1).cpu().tolist())
            if training and step % 50 == 0:
                print(f"  batch {step}/{len(loader)} loss={loss_sum/total:.4f} acc={correct/total:.4f}", flush=True)
    result = {"loss": loss_sum/total, "accuracy": correct/total,
              "seconds": time.perf_counter()-started}
    if not training:
        result.update({"macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
                       "predictions": predictions, "labels": labels, "probabilities": probabilities,
                       "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(3))).tolist()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=["resnet18", "yolov8n_cls"])
    parser.add_argument("--pretrained", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(".training/cache"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42177)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This training protocol requires a bfloat16-capable CUDA GPU or --device cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but unavailable")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit("Run folder already contains an experiment; choose a new run folder")
    root = args.data_root.resolve()
    _, audit = checked_training_rows(root, args.audit)
    caches, rows = {}, {}
    for split in ("train", "validation"):
        caches[split], rows[split], digest = build_cache(root, split, args.cache)
        if digest != audit["manifest_hashes"][split]:
            raise ValueError("The fixed manifest has changed")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(FishDataset(caches["train"], rows["train"], True), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, persistent_workers=args.workers > 0,
                              worker_init_fn=worker_seed, generator=generator)
    val_loader = DataLoader(FishDataset(caches["validation"], rows["validation"]), batch_size=args.batch_size*2,
                            shuffle=False, num_workers=args.workers, persistent_workers=args.workers > 0)
    model, architecture = build_network(args.family, pretrained=args.pretrained)
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    versions = {p: importlib.metadata.version(p) for p in ["torch", "torchvision", "ultralytics", "numpy", "Pillow"]}
    config = {"family": args.family, "seed": args.seed, "epochs_limit": args.epochs,
              "warmup_epochs": args.warmup, "patience": args.patience,
              "image_size": 224, "batch_size": args.batch_size, "device": str(device),
              "backbone_learning_rate": args.learning_rate, "head_learning_rate": args.learning_rate*10,
              "optimiser": "AdamW", "weight_decay": 0.0001, "label_smoothing": 0.05,
              "dropout": 0.2, "augmentation": "Training only: horizontal/vertical flips; +/-12deg expand+letterbox. No colour jitter or random crop.",
              "normalisation": "ImageNet mean/std" if args.family == "resnet18" else "RGB [0,1], matching Ultralytics classification defaults",
              "pretrained_sha256": sha256_file(args.pretrained), "pretrained_file": str(args.pretrained),
              "manifest_hashes": audit["manifest_hashes"], "versions": versions,
              "selection": "Validation macro-F1, then accuracy, then lower validation loss",
              "stopping": "Maximum epochs or patience stale validation epochs, after at least warmup+5 epochs",
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "mixed_precision": "bfloat16 train; float32 validation" if device.type == "cuda" else "float32",
              "cudnn_deterministic": True, "cross_hardware_bitwise_reproducibility": False,
              "limitation": LIMITATION, "labels": LABELS,
              "masking": False, "validation_used_for_gradient_updates": False,
              "test_used_during_training": False,
              "source_sha256": {p.name: sha256_file(p) for p in [Path(__file__), Path(__file__).with_name("cnn.py")]}}
    write_json(out / "config.json", config)
    head_params = list(model.head().parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    for p in backbone_params:
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW(head_params, lr=args.learning_rate*10, weight_decay=0.0001)
    scheduler = None
    history, best_rank, best_epoch, stale = [], (-1., -1., -float("inf")), 0, 0
    start = time.perf_counter()
    print(f"Training {args.family}, seed={args.seed}, device={device}, train={len(rows['train'])}, val={len(rows['validation'])}", flush=True)
    stop_reason = "epoch_limit"
    for e in range(args.epochs):
        warmup = e < args.warmup
        if e == args.warmup:
            for p in backbone_params:
                p.requires_grad_(True)
            optimizer = torch.optim.AdamW([
                {"params": backbone_params, "lr": args.learning_rate},
                {"params": head_params, "lr": args.learning_rate*10},
            ], weight_decay=0.0001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=2, min_lr=1e-6)
        train = epoch(model, train_loader, criterion, device, optimizer, warmup)
        val = epoch(model, val_loader, criterion, device)
        record = {"epoch": e+1, "stage": "head_warmup" if warmup else "fine_tuning",
                  "train_loss": train["loss"], "train_accuracy": train["accuracy"],
                  "validation_loss": val["loss"], "validation_accuracy": val["accuracy"],
                  "validation_macro_f1": val["macro_f1"],
                  "learning_rate": optimizer.param_groups[0]["lr"],
                  "train_seconds": train["seconds"], "validation_seconds": val["seconds"]}
        history.append(record)
        write_csv(out / "history.csv", history)
        rank = (val["macro_f1"], val["accuracy"], -val["loss"])
        if rank > best_rank:
            best_rank, best_epoch, stale = rank, e+1, 0
            checkpoint = {"family": args.family, "architecture": architecture, "state_dict": cpu_state(model),
                          "labels": LABELS, "image_size": 224,
                          "model_version": f"{args.family}-ffe3-seed{args.seed}-epoch{e+1}",
                          "best_epoch": e+1, "validation_macro_f1": val["macro_f1"],
                          "validation_accuracy": val["accuracy"], "training_config": config}
            atomic_save(checkpoint, out / "best.pt")
            save_predictions(out, "validation", rows["validation"], val["predictions"], val["probabilities"])
            write_json(out / "best_validation.json", {k: val[k] for k in ["loss", "accuracy", "macro_f1", "confusion_matrix"]})
        else:
            stale += 1
        if scheduler:
            scheduler.step(val["macro_f1"])
        print(f"{args.family} seed={args.seed} epoch={e+1:02d} train_loss={train['loss']:.4f} train_acc={train['accuracy']:.4f} val_acc={val['accuracy']:.4f} val_f1={val['macro_f1']:.4f} best_epoch={best_epoch} seconds={train['seconds']+val['seconds']:.1f}", flush=True)
        if e >= args.warmup+4 and stale >= args.patience:
            stop_reason = "validation_plateau"
            break
    write_json(out / "summary.json", {"family": args.family, "seed": args.seed, "best_epoch": best_epoch,
               "epochs_run": len(history), "validation_macro_f1": best_rank[0], "validation_accuracy": best_rank[1],
               "stop_reason": stop_reason, "training_wall_seconds": time.perf_counter()-start,
               "checkpoint_sha256": sha256_file(out / "best.pt"), "device": str(device)})
    print(f"COMPLETED {args.family}: best validation F1={best_rank[0]:.5f}, accuracy={best_rank[1]:.5f}", flush=True)


if __name__ == "__main__":
    main()
