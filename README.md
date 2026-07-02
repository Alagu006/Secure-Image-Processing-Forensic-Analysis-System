# Secure Image Processor

A FastAPI-based web application with an 8-stage security scanning pipeline and a 16-operation image processing engine powered by OpenCV, Pillow, and scikit-image.

```
                           ┌─────────────────────────────────────┐
                           │          SECURITY SCANNER           │
  ┌─────────┐              │  ┌──────┐  ┌──────┐  ┌──────────┐ │
  │         │    POST      │  │Magic │  │Header│  │ Entropy  │ │
  │ Upload  │   /upload    │  │Bytes │  │Inspect│  │ Analysis │ │
  │  Image  │─────────────▶│  └──────┘  └──────┘  └──────────┘ │
  │         │              │  ┌──────┐  ┌──────┐  ┌──────────┐ │
  └─────────┘              │  │Poly‑ │  │ EXIF │  │Dimension │ │
      ▲                    │  │glot  │  │Check │  │  Sanity  │ │
      │                    │  └──────┘  └──────┘  └──────────┘ │
      │                    │  ┌──────────────────────────────┐  │
      │                    │  │   Pixel Re-encoding (PIL)    │  │
      │                    │  └──────────────────────────────┘  │
      │                    └─────────────────────────────────────┘
      │                                         │
      │                                    Sanitized
      │                                      Image
      │                                         │
      │                    ┌────────────────────▼────────────────┐
      │                    │         IMAGE PROCESSOR             │
      │                    │  ┌─────────┐  ┌──────┐  ┌────────┐ │
      │                    │  │ OpenCV  │  │Pillow│  │skimage │ │
      │                    │  │ 7 ops   │  │5 ops │  │ 4 ops  │ │
      │                    │  └─────────┘  └──────┘  └────────┘ │
      │                    └─────────────────────────────────────┘
      │                                         │
      │              ┌──────────────────────────▼───────┐
      │              │           DOWNLOAD               │
      └──────────────│  GET /download/{filename}        │
                     │  GET /scan-report/{filename}     │
                     └──────────────────────────────────┘
```

## Features

### Security Pipeline (8 checks)

| Check | What it does |
|-------|-------------|
| Magic Bytes | Reads first 16 bytes, validates against JPEG/PNG/GIF/WEBP signatures |
| File Size | Rejects files over 10 MB |
| Header Inspection | Opens and verifies with Pillow, catches corrupt/malformed images |
| Polyglot Detection | Scans body (after byte 512) for ZIP, PDF, ELF, and other embedded format signatures |
| EXIF Extraction | Extracts metadata keys; flags GPS location data |
| Entropy Analysis | Computes Shannon entropy; flags if > 7.5 (potential hidden encrypted data) |
| Pixel Re-encoding | Fully decompresses and re-saves the image to destroy steganographic payloads |
| Dimension Sanity | Rejects images wider/taller than 10 000 px or smaller than 10 px |

### Image Processing (16 operations)

| Library | Operations |
|---------|-----------|
| **OpenCV** | translate, rotate, reflect, crop, shear, edge_detect, equalize |
| **Pillow** | brightness, contrast, color, sharpness, noise_remove |
| **scikit-image** | grayscale, hsv, segment, blur |

## Setup

### Local

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

The API runs at `http://localhost:8000`. Open `frontend/index.html` in a browser for the UI.

### Docker

```bash
docker compose up
```

Images persist in `./uploads` and `./processed` via bind mounts.

### Tests

```bash
pip install pytest
pytest tests/
```

## API Reference

| Method | Endpoint | Params | Response |
|--------|----------|--------|----------|
| `POST` | `/upload` | `file` (multipart) | `{status, message, data: {original_filename, stored_filename, scan}}` |
| `POST` | `/process` | `?filename=` + JSON body `{operations: [...]}` | `{status, message, data: {original_filename, processed_filename, result}}` |
| `GET` | `/download/{filename}` | path param | File (binary) |
| `GET` | `/scan-report/{filename}` | path param | `{status, message, data: {passed, score, checks, ...}}` |

The `/process` endpoint accepts operations as a list of strings (`"grayscale"`) or dicts (`{"operation": "blur", "params": {"sigma": 2}}`).

## Screenshots

> _Screenshots to be added once the application is running:_
>
> **Upload & Scan Section** — Dark-themed drag-and-drop zone with scan progress animation and the Security Report Card showing score, SAFE/REJECTED badge, and individual check results.
>
> **Processing Section** — Two-panel layout with original / processed image comparison slider; tabbed operation cards (OpenCV / Pillow / scikit-image) and the Pipeline Queue with add/reorder/remove controls.
>
> **Download Section** — Metadata cards (dimensions, operations, file size) with Download and Start Over buttons.

## Deploy to Render (Free Tier)

One-click deploy — no credit card required.

### Step-by-step

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USER/securevision.git
   git push -u origin main
   ```

2. **Connect Render**
   - Go to [dashboard.render.com](https://dashboard.render.com) and click **New + → Web Service**
   - Connect your GitHub account and select the `securevision` repository
   - Render will auto-detect the `render.yaml` blueprint — click **Apply**

3. **Configure (blueprint does it for you)**
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - **Disk**: A 1 GB persistent disk is mounted at `/app/uploads` for uploaded/processed images

4. **Deploy**
   - Click **Deploy** and wait ~3 minutes for the build & deploy
   - Once live, your URL will be `https://securevision.onrender.com` (or whatever you named the service)

5. **Update the frontend**
   - Edit `frontend/config.js` and set:
     ```js
     window.API_BASE = "https://YOUR-APP.onrender.com";
     ```
   - Commit and push — Render will auto-redeploy

### Important notes

| Note | Detail |
|------|--------|
| **Cold starts** | Render's free tier spins down after 15 min of inactivity. The first request after idle takes ~30 s while the service wakes up. |
| **YOLO model** | `yolov8n.pt` (~6 MB) is downloaded automatically on the first `detect_objects` request. Subsequent requests reuse the cached model. |
| **CORS** | Currently set to `allow_origins=["*"]`. Lock this down in `backend/main.py` before production use. |
| **Ephemeral storage** | The `/app/uploads` disk is persisted. All other directories (`__pycache__`, `batch_manifests/`) are ephemeral and reset on each deploy. |

## Project Structure

```
├── backend/
│   ├── main.py         FastAPI app, CORS, routes
│   ├── scanner.py      8-stage security scan pipeline
│   ├── processor.py    16 operations + pipeline chaining
│   ├── models.py       Pydantic response models
│   └── utils.py        Path helpers, constants
├── frontend/
│   └── index.html      Single-page UI (inline CSS + JS)
├── tests/
│   └── test_scanner.py Pytest test suite
├── uploads/            Persisted uploaded images
├── processed/          Persisted processed images
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
