# Model and explanation integration

## API contract v1

The local server uses JSON requests on the same origin. JPEG/PNG inputs are
base64 encoded without a data URL prefix. Maximum decoded image size: 8 MB;
maximum dimensions: 20 million pixels. EXIF orientation is corrected, metadata
is not passed on, and pixels are converted to RGB. No images are persisted.

`POST /api/predict/species` or `POST /api/predict/freshness`

```json
{"image_base64":"<base64 bytes>"}
```

The public API does not accept model or mode overrides. Configure each final
model in `backend/config.py` using `FINAL_MODELS`. Allowed selections are `svm`,
`resnet18`, `yolov8n_cls`; `None` leaves that task in explicit demo mode. No model
has been selected yet. Tasks may use different final models. Restart the server
when changing this configuration. A selected but unregistered model returns 503.
Demo returns the first label, a null score and no selected model.
`GET /api/config` exposes only each task's mode for the UI banner and button.

Species IDs and scientific names:

| ID | Scientific name |
|---|---|
| horse_mackerel | Trachurus trachurus |
| whiting | Merlangius merlangus |
| haddock | Melanogrammus aeglefinus |
| cod | Gadus morhua |
| hake | Merluccius merluccius |

Freshness IDs: `highly_fresh`, `fresh`, `not_fresh`. Do not infer elapsed storage
days from a prediction. These are labels defined by the FFE study conditions.

Responses contain `id`, `task`, `model`, `mode`, `label`, `score`, `model_version`,
`scientific_name`, `inference_ms`, `created_at`. `score` is nullable; only return
a number in [0,1] when the model provides it. Do not turn arbitrary SVM decision
values into probabilities. Latency excludes image decoding and is not a research
benchmark. Scores do not detect unsupported species reliably.

`POST /api/explain` takes `{"result_id":"<id from prediction>"}`. It retrieves
the result server-side. The last 100 results are held in memory, accessible for
up to one hour; a restart removes them. It returns `provider`, `agent_connected`,
`text`, `location_note`, `safety_note`, `sources`. Currently this is a deterministic
local explanation template, not a connected language model.

`GET /api/health` lists connected task/model pairs. Errors use `{"error":"..."}`:
400 invalid input; 403 cross-origin; 404 missing route/result; 413 request size;
415 content type; 503 model unavailable; 500 adapter failure.

## Connecting a trained model

Create `backend/model_setup.py` and register functions at startup:

```python
from .adapters import register
from .my_resnet import predict_species, predict_freshness

register("species", "resnet18", predict_species)
register("freshness", "resnet18", predict_freshness)
```

Each function receives a PIL RGB image and returns:

```python
return {"label": label_id, "score": score_or_none, "model_version": "resnet18-species-v1"}
```

Deliver the weights, inference code, exact class-index order, preprocessing
(resize/padding/normalisation/masking), dependency versions and an example image
with an expected result. Match training preprocessing, including SVM feature
standardisation and exclusion of padding. Do not rely on ground-truth masks at
inference unless the user can actually supply them. Load weights once; set neural
models to evaluation mode. Protect non-thread-safe model state with a lock.

All three families need separate predictors for the two tasks. Restart the
server after registration. Do not commit datasets, weights, uploads or secrets.

## Connecting an Agent API later

Replace `backend/agent.py` with the chosen provider's server-side integration.
Keep keys in environment variables. Send trusted classification results and
reviewed species records, not browser-supplied claims. Sources must come from
those records and be returned with the explanation. Update the health endpoint
and frontend provider badge when connecting a real provider. Provide a timeout
and local fallback so classification still works if the provider fails.

Do not assert edibility, toxin absence, disease or unverified Sydney beach
occurrence. Use scientific names as lookup keys. If reliable evidence is absent,
say so. Never upgrade uncertain predictions to certain identifications. The
current knowledge directory intentionally contains no fabricated location data.

## Deployment boundary

This is a loopback-only teaching prototype using Python's development HTTP
server. Public deployment requires a production service, authentication if
needed, request/concurrency limits, HTTPS and a reviewed data retention policy.
GitHub is for source collaboration, not for running Python inference on Pages.
