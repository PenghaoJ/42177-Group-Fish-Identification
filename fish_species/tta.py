"""Optional validation-selected four-flip inference, with unchanged weights.

This averages predictions for the original, horizontal, vertical and both-flipped
complete image. It uses four forward passes; latency records include that cost.
"""
import torch
from .cnn import SpeciesCNNPredictor, image_tensor


def batch_probabilities(model, x, mode="single"):
    if mode == "single":
        return model(x).softmax(1)
    if mode != "flip4":
        raise ValueError(mode)
    views = torch.cat([x, x.flip(-1), x.flip(-2), x.flip((-2, -1))])
    return model(views).softmax(1).reshape(4, len(x), 5).mean(0)


class SpeciesTTAPredictor(SpeciesCNNPredictor):
    def __call__(self, image):
        x = image_tensor(image, self.checkpoint["image_size"]).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = batch_probabilities(self.model, x, "flip4").cpu()[0]
        label_id = int(probabilities.argmax())
        return {"label": self.checkpoint["labels"][label_id],
                "score": float(probabilities[label_id]),
                "model_version": self.checkpoint["model_version"]+"-flip4"}
