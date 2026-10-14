# FishLens

Local website and API scaffold for the submitted fish species and eye freshness
project. Two independent image tasks, three model families, one shared interface.

## Run locally

On Windows 10/11, download or clone the repository, extract the entire folder
if downloaded as a ZIP, then double-click **Start-FishLens.cmd**. Do not run it
inside the ZIP preview. The launcher:

1. Finds Python 3.11+ or attempts a per-user Python 3.12 install with WinGet.
2. Creates a private `.venv` inside the project.
3. Installs `requirements.txt` on first launch or when the file changes.
4. Starts the local service and opens your default browser after it is ready.

Internet is required for first-time dependency/runtime installation. Subsequent
launches reuse the environment. If Windows policy, missing WinGet or a package
agreement prevents automatic Python installation, the launcher gives a Python
installation link. It does not bypass Windows security warnings or permanently
change PowerShell policy. Python installation, if needed, is per Windows user;
project dependencies remain in `.venv`.

Double-click **Stop-FishLens.cmd** to stop the launcher-managed server. Closing
the browser does not stop the server. Repeated launches reopen the existing
website. Port 8765 is preferred; occupied ports are skipped up to 8785. The
website is only available on the current computer, not the public internet.
Logs are in `.runtime/server.stdout.log` and `.runtime/server.stderr.log`.

If you move the folder to another computer, share source files only: let the
launcher recreate `.venv` and `.runtime`. Git already ignores both directories.
After pulling backend, model or configuration changes, stop and start again.
No GitHub login, Git installation or Codex installation is needed to use a ZIP.

For macOS/Linux (the double-click launcher is Windows-only):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m backend.server
```

Open http://127.0.0.1:8765 when starting manually.

Only Pillow is required. The frontend uses HTML, CSS and JavaScript without a
build step. The Python backend is deliberately small for student integration.

## What works

- Responsive English interface, two task tabs, image preview and drag-and-drop.
- Final model selected per task in backend/config.py, with no user-facing model selector.
- Explicit demo display until each task has its final model configured.
- JPEG/PNG validation, size limits and EXIF orientation correction.
- Prediction endpoints with validated adapter outputs and a model registry.
- Server-side result lookup and local explanation fallback.
- Clear errors for missing models, bad images and expired results.

## What is not connected yet

No trained model, external Agent API or verified location database is included.
Demo output is fixed and never represents a prediction. An image cannot confirm
food safety. The five supported species are the submitted AutoFish subset, not
a validated Sydney fish catalogue. Eye input must already be a clear close-up.

See [integration contract](docs/INTEGRATION.md) before implementing a model.

## Collaboration layout

- `frontend/`: website
- `backend/`: API, model registry and explanation adapter
- `training/`: model training code
- `inference/`: shared inference implementation notes
- `knowledge/`: reviewed information and citations
- `evaluation/`: agreed held-out metrics
- `docs/`: handoff and API contract

Use separate branches and pull requests. The local Git repository is ready for
an eventual GitHub remote; no GitHub repository or collaborator invitations are
created by this scaffold. Do not commit the full course folder.

## Tests

```powershell
python -m unittest discover -s tests -v
```
