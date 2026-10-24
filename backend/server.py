"""Local prototype server. Run from project root: python -m backend.server."""
import base64
import hashlib
import binascii
import io
import json
import math
import os
import threading
import time
import uuid
import warnings
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from PIL import Image, ImageOps, UnidentifiedImageError
from PIL.Image import open as open_uploaded_image
from . import adapters, agent, config

ROOT = Path(__file__).resolve().parent.parent
INSTANCE_ID = hashlib.sha256(str(ROOT).casefold().encode()).hexdigest()[:16]
MAX_IMAGE = 8 * 1024 * 1024
MAX_BODY = 12 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000
RESULTS = OrderedDict()
LOCK = threading.Lock()
# Team members can register implementations in this optional local module.
if (ROOT / "backend" / "model_setup.py").exists():
    from . import model_setup  # noqa: F401

def process(task, payload):
    if task not in adapters.LABELS:
        raise ValueError("Unknown task")
    if "model" in payload or "mode" in payload:
        raise ValueError("The model is selected by the service, not by the client")
    model = config.FINAL_MODELS[task]
    mode = "demo" if model is None else "trained"
    if model is not None and model not in adapters.MODELS:
        raise RuntimeError("Invalid final model configuration")
    try:
        raw = base64.b64decode(payload.get("image_base64", ""), validate=True)
    except (binascii.Error, TypeError):
        raise ValueError("Invalid image encoding")
    if not raw or len(raw) > MAX_IMAGE:
        raise ValueError("Choose a JPEG or PNG image up to 8 MB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            # Retain Pillow's decoder before Ultralytics patches Image.open.
            # Invalid uploads must remain HTTP 400, without optional HEIF imports.
            with open_uploaded_image(io.BytesIO(raw), formats=("JPEG", "PNG")) as original:
                if original.format not in ("JPEG", "PNG"):
                    raise ValueError("Only JPEG and PNG images are supported")
                if original.width * original.height > 20_000_000:
                    raise ValueError("Image exceeds the 20 megapixel limit")
                original.load()
                image = ImageOps.exif_transpose(original).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ValueError("Invalid image or image exceeds the 20 megapixel limit")
    start = time.perf_counter()
    output = adapters.predict(task, model, image, mode)
    if not isinstance(output, dict) or output.get("label") not in adapters.LABELS[task]:
        raise RuntimeError("Model adapter returned an invalid label")
    score = output.get("score")
    if score is not None and (type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1):
        raise RuntimeError("Model adapter returned an invalid score")
    if not isinstance(output.get("model_version"), str):
        raise RuntimeError("Model adapter must return model_version")
    result = {"id": uuid.uuid4().hex, "task": task, "model": model, "mode": mode,
              "label": output["label"], "score": score, "model_version": output["model_version"],
              "scientific_name": adapters.SPECIES.get(output["label"]),
              "inference_ms": round((time.perf_counter()-start)*1000, 2),
              "created_at": time.time()}
    with LOCK:
        RESULTS[result["id"]] = result
        while len(RESULTS) > 100:
            RESULTS.popitem(last=False)
    return result

class Handler(BaseHTTPRequestHandler):
    def send(self, status, data, content_type="application/json; charset=utf-8"):
        body = json.dumps(data, allow_nan=False).encode() if isinstance(data, (dict, list)) else data
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; object-src 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/config":
            return self.send(200, {"tasks": {task: {"mode": "demo" if model is None else "trained"}
                                           for task, model in config.FINAL_MODELS.items()}})
        if path == "/api/health":
            return self.send(200, {"status": "ok", "app": "fishlens", "instance_id": INSTANCE_ID,
                                  "pid": os.getpid(), "launch_id": os.environ.get("FISHLENS_LAUNCH_ID"), "agent_connected": False,
                                  "connected_models": [list(k) for k in adapters.PREDICTORS]})
        files = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"), "/styles.css": ("styles.css", "text/css; charset=utf-8")}
        if path in files:
            file, mime = files[path]
            return self.send(200, (ROOT / "frontend" / file).read_bytes(), mime)
        self.send(404, {"error": "Not found"})

    def do_POST(self):
        # Local same-origin application; no cross-origin API access.
        origin = self.headers.get("Origin")
        if origin and origin != "http://" + self.headers.get("Host", ""):
            return self.send(403, {"error": "Cross-origin requests are not allowed"})
        if self.headers.get_content_type() != "application/json":
            return self.send(415, {"error": "Use application/json"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                return self.send(413, {"error": "Request is empty or too large"})
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object")
            path = urlsplit(self.path).path
            if path in ("/api/predict/species", "/api/predict/freshness"):
                return self.send(200, process(path.rsplit("/", 1)[1], payload))
            if path == "/api/explain":
                result_id = payload.get("result_id")
                if not isinstance(result_id, str):
                    raise ValueError("result_id must be a string")
                with LOCK:
                    result = RESULTS.get(result_id)
                if not result or time.time() - result["created_at"] > 3600:
                    return self.send(404, {"error": "Result expired; run the image again"})
                return self.send(200, agent.explain(result))
            self.send(404, {"error": "Not found"})
        except LookupError as exc:
            self.send(503, {"error": str(exc)})
        except (ValueError, UnicodeError) as exc:
            self.send(400, {"error": str(exc)})
        except Exception:
            self.send(500, {"error": "Model or explanation service failed. Check the server adapter."})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8765"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"FishLens local preview: http://127.0.0.1:{port}", flush=True)
    server.serve_forever()
