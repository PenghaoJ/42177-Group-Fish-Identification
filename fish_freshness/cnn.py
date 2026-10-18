"""FFE eye preprocessing, three-class networks and saved CNN inference."""
import copy
import os
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".cache/ultralytics").resolve()))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torchvision.models import resnet18
from .data import LABELS


def letterbox(image, size=224):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size))
    canvas.paste(image, ((size-image.width)//2, (size-image.height)//2))
    return canvas


def image_tensor(image, size=224):
    array = np.asarray(letterbox(image, size), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255)


class FreshnessNetwork(nn.Module):
    def __init__(self, base, family):
        super().__init__()
        self.base = base
        self.family = family
        mean, std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) if family == "resnet18" else ([0., 0., 0.], [1., 1., 1.])
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        out = self.base((x-self.mean)/self.std)
        # Ultralytics classification returns (probabilities, logits) in eval mode.
        if isinstance(out, tuple):
            out = out[1]
        if out.ndim != 2 or out.shape[1] != len(LABELS):
            raise ValueError(f"Unexpected classifier logits: {out.shape}")
        return out

    def head(self):
        return self.base.fc if self.family == "resnet18" else self.base.model[-1]


def build_network(family, pretrained=None, architecture=None):
    if family == "resnet18":
        base = resnet18(weights=None)
        if pretrained:
            base.load_state_dict(torch.load(pretrained, map_location="cpu", weights_only=True))
        base.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(base.fc.in_features, len(LABELS)))
        architecture = None
    elif family == "yolov8n_cls":
        if pretrained:
            from ultralytics import YOLO
            base = YOLO(str(pretrained), task="classify").model.float()
            head = base.model[-1]
            head.linear = nn.Linear(head.linear.in_features, len(LABELS))
            head.drop.p = 0.2
            base.names = dict(enumerate(LABELS))
            architecture = copy.deepcopy(base.yaml)
            architecture["nc"] = len(LABELS)
        else:
            from ultralytics.nn.tasks import ClassificationModel
            if architecture is None:
                raise ValueError("Saved YOLO architecture is required")
            base = ClassificationModel(cfg=copy.deepcopy(architecture), ch=3, nc=len(LABELS), verbose=False)
            base.model[-1].drop.p = 0.2
    else:
        raise ValueError(family)
    return FreshnessNetwork(base, family), architecture


def choose_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FreshnessCNNPredictor:
    """FishLens adapter; score is a softmax score, not calibrated certainty."""
    def __init__(self, path, device="cpu"):
        self.device = choose_device(device)
        self.checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if self.checkpoint["labels"] != LABELS:
            raise ValueError("Unexpected FFE label order")
        self.model, _ = build_network(self.checkpoint["family"], architecture=self.checkpoint["architecture"])
        self.model.load_state_dict(self.checkpoint["state_dict"])
        self.model.to(self.device).eval()

    def __call__(self, image):
        x = image_tensor(image, self.checkpoint["image_size"]).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = self.model(x).softmax(dim=1).cpu()[0]
        label_id = int(probabilities.argmax())
        return {"label": self.checkpoint["labels"][label_id],
                "score": float(probabilities[label_id]),
                "model_version": self.checkpoint["model_version"]}
