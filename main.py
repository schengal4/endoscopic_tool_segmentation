# main.py  (safe UI tweaks + proxy-friendly helpers)

import os
import torch
import shutil
import uvicorn
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from typing import List, Dict, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request, Response, Query, status
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from monai.bundle import ConfigParser
from monai.data import decollate_batch
from PIL import Image, ImageOps
import numpy as np
from datetime import datetime
import uuid
import tempfile
import zipfile
import logging
import json
from fastapi import Request, Response
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from typing import Optional, List
from fastapi.responses import FileResponse



# ---- logging ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- app ----
app = FastAPI(
    title="MONAI Endoscopic Tool Segmentation API",
    version="1.0.0",
    description="""
API for segmenting surgical tools in endoscopic images.

**Using this API via Swagger (/docs)**
1. Open **/docs** and find **POST /segment**.
2. Click **Try it out**, attach 1+ image files (PNG/JPG).
3. If you ever see **403 Forbidden** responses in Swagger (from an upstream gateway),
   simply refesh the page and hit **Execute** again. These are transient, edge-level blocks.
4. The endpoint returns a ZIP file with the original (re-oriented) images and blended overlays.
   In Swagger, use the **Download file** link in the response section to save the ZIP.
""",
    openapi_tags=[
        {"name": "Segmentation", "description": "Core endpoints for image processing."},
        {"name": "Utilities", "description": "Health check and metadata."},
        {"name": "Examples", "description": "Run the built-in demo set."},
    ],
)


@app.middleware("http")
async def no_store(request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("Cache-Control", "no-store")
    return resp
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
    max_age=86400,
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
app.add_middleware(ProxyHeadersMiddleware)

# ---- paths & device ----
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BASE_DIR = "."
MODELS_DIR = os.path.join(BASE_DIR, "models")
CONFIG_DIR = os.path.join(BASE_DIR, "configs")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "monai_results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")  # optional in dev

# ---- MONAI globals ----
model_config = None
model = None
inferer = None
original_postprocessing = None


def load_model():
    global model_config, model, inferer, original_postprocessing
    cfg_path = os.path.join(CONFIG_DIR, "inference.json")
    model_config = ConfigParser()
    model_config.read_config(cfg_path)
    model_config["bundle_root"] = "."
    model_config["output_dir"] = RESULTS_DIR

    checkpoint = os.path.join(MODELS_DIR, "model.pt")
    model = model_config.get_parsed_content("network").to(device)
    inferer = model_config.get_parsed_content("inferer")

    # Strip SaveImaged from postprocessing to avoid path-side-effects
    original_postprocessing = model_config.get_parsed_content("postprocessing")
    new_transforms = []
    for t in original_postprocessing.transforms:
        if not (hasattr(t, "__class__") and t.__class__.__name__ == "SaveImaged"):
            new_transforms.append(t)
    original_postprocessing.transforms = new_transforms

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    logger.info("Model loaded")

@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request, exc):
    if exc.status_code == 403:
        return JSONResponse(
            status_code=403,
            content={"detail": "Access temporarily forbidden. Please refresh and try again."}
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail if exc.detail else "Unexpected error occurred."}
    )


@app.on_event("startup")
async def startup_event():
    load_model()


def log_processing_event(event_type: str, session_id: str, details: dict):
    audit_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event_type": event_type,
        "session_id": session_id,
        "details": details,
    }
    logger.info(f"AUDIT {json.dumps(audit_entry)}")


def get_results(input_dir: str, output_dir: str) -> Dict[str, str]:
    """
    Runs inference and writes:
      - corrected-orientation original image: {name}.png
      - mask image: {name}_trans.png  (internal; not returned)
      - blended image: {name}_blend.png
    Returns dict with keys like "<stem>_original" and "<stem>_blend".
    """
    global model_config, model, inferer, original_postprocessing

    if model is None:
        load_model()

    model_config["dataset_dir"] = input_dir
    model_config["output_dir"] = output_dir
    os.makedirs(output_dir, exist_ok=True)

    dataloader = model_config.get_parsed_content("dataloader")
    result_paths = {}

    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            orig_path = batch["image_meta_dict"]["filename_or_obj"][0]
            base = os.path.basename(orig_path)
            stem = os.path.splitext(base)[0]

            batch["pred"] = inferer(images, network=model)
            for data_i in decollate_batch(batch):
                # Try configured postprocessing; fall back to manual handling
                try:
                    processed = original_postprocessing(data_i)
                    mask = processed["pred"].detach().cpu().numpy()
                except RuntimeError:
                    raw = data_i["pred"].detach().cpu().numpy()
                    if raw.min() < 0 or raw.max() > 1:
                        raw = torch.sigmoid(torch.from_numpy(raw)).numpy()
                    mask = raw

                # prep original
                orig_img = data_i["image"].detach().cpu().numpy()
                if orig_img.shape[0] in (1, 3):
                    if orig_img.shape[0] == 1:
                        orig_img = orig_img[0]
                    else:
                        orig_img = np.transpose(orig_img, (1, 2, 0))
                orig_img = (orig_img * 255).astype(np.uint8)

                # prep mask (binary)
                if len(mask.shape) > 3:
                    mask = mask[0]
                mask_vis = np.argmax(mask, axis=0) if mask.shape[0] > 1 else mask[0]
                mask_vis = (mask_vis > 0.5).astype(np.float32)
                mask_vis = (mask_vis * 255).astype(np.uint8)

                # Resize & fix orientation (rotate 270°, then mirror)
                mh, mw = mask_vis.shape
                orig_pil = Image.fromarray(orig_img).resize((mw, mh), Image.BILINEAR)
                orig_pil = ImageOps.mirror(orig_pil.transpose(Image.ROTATE_270))
                mask_pil = ImageOps.mirror(Image.fromarray(mask_vis).transpose(Image.ROTATE_270))

                orig_np = np.array(orig_pil)
                mask_np = np.array(mask_pil)

                out_dir = os.path.join(output_dir, stem)
                os.makedirs(out_dir, exist_ok=True)

                orig_file = os.path.join(out_dir, f"{stem}.png")
                mask_file = os.path.join(out_dir, f"{stem}_trans.png")
                blend_file = os.path.join(out_dir, f"{stem}_blend.png")

                orig_pil.save(orig_file)
                mask_pil.save(mask_file)

                # blend (simple red overlay at 0.5 alpha)
                alpha = 0.5
                mask_rgb = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
                mask_rgb[mask_np > 0, 0] = 255
                blended = (orig_np.astype(float) * (1 - alpha) + mask_rgb.astype(float) * alpha).astype(np.uint8)
                Image.fromarray(blended).save(blend_file)

                # Expose only originals and blends
                result_paths[f"{stem}_original"] = orig_file
                result_paths[f"{stem}_blend"] = blend_file

    return result_paths


def clean_old_files(background_tasks: BackgroundTasks):
    def cleanup():
        now = datetime.now().timestamp()
        for root in (UPLOAD_DIR, RESULTS_DIR):
            for d in os.listdir(root):
                path = os.path.join(root, d)
                try:
                    if os.path.isdir(path) and os.path.getmtime(path) < now - 3600:
                        shutil.rmtree(path, ignore_errors=True)
                except Exception:
                    pass

    background_tasks.add_task(cleanup)


def create_zip_from_results(result_paths: Dict[str, str], session_id: str) -> str:
    """
    Create a zip that contains only the *original* and *blend* images for each file.
    Paths are stored inside the zip with a clean structure:
        results/<stem>/<stem>.png
        results/<stem>/<stem>_blend.png
    """
    tmp_dir = tempfile.mkdtemp(prefix=f"zip_{session_id}_")
    zip_path = os.path.join(tmp_dir, f"{session_id}_results.zip")

    # Deduplicate and keep order (originals first, blends second)
    to_add = []
    for k in sorted(result_paths.keys()):
        if k.endswith("_original") or k.endswith("_blend"):
            to_add.append(result_paths[k])

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for abs_path in to_add:
            stemdir = os.path.basename(os.path.dirname(abs_path))
            fname = os.path.basename(abs_path)
            arc = os.path.join("results", stemdir, fname)
            zf.write(abs_path, arc)

    return zip_path

# ----------------- endpoints -----------------


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return HTMLResponse("""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>MONAI Endoscopic Tool Segmentation</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root { --bg:#0b1020; --card:#131a2b; --muted:#8ca3c7; --accent:#5cc8ff; --ok:#5CFF9E; --warn:#ffc857; }
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:#e6eefc;font-family:Inter,ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial}
    .wrap{max-width:880px;margin:40px auto;padding:0 20px}
    .card{background:var(--card);border:1px solid #1f2a44;border-radius:18px;padding:22px;box-shadow:0 10px 30px rgba(0,0,0,.35)}
    h1{margin:0 0 10px;font-weight:700;letter-spacing:.2px}
    p.sub{color:var(--muted);margin:0 0 24px}
    .drop{border:2px dashed #2d3c61;border-radius:16px;padding:24px;text-align:center;transition:.2s background}
    .drop.dragover{background:#0f1730}
    .files{display:flex;flex-wrap:wrap;gap:10px;margin-top:12px}
    .chip{background:#1a2440;border:1px solid #27345a;border-radius:999px;padding:6px 10px;font-size:12px}
    .row{display:flex;gap:12px;flex-wrap:wrap;margin-top:18px}
    button{border:0;border-radius:12px;padding:12px 16px;font-weight:600;cursor:pointer}
    .primary{background:var(--accent)}
    .ghost{background:#243259;color:#e6eefc}
    .status{margin-top:16px;min-height:24px;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;color:var(--muted)}
    .status.ok{color:var(--ok)} .status.warn{color:var(--warn)}
    .footer{margin-top:14px;color:#7d91b6;font-size:12px}
    .thumb{width:88px;height:88px;object-fit:cover;border-radius:10px;border:1px solid #27345a}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Endoscopic Tool Segmentation</h1>
      <p class="sub">Upload one or more images. The app returns a ZIP with originals and overlays.</p>

      <div id="drop" class="drop">
        <p><strong>Drop images</strong> here or <u>click to select</u>.</p>
        <input id="file-input" type="file" accept="image/*" multiple hidden>
        <div id="files" class="files"></div>
      </div>

      <div class="row">
        <button id="process" class="primary">Process</button>
        <button id="example" class="ghost">Run Example</button>
        <button id="clear" class="ghost">Clear</button>
      </div>

      <div id="status" class="status">Ready.</div>
      <div class="footer">API: <code>POST /segment</code> • <code>GET /example</code> • <code>GET /health</code></div>
    </div>
  </div>

<script>
(function(){
  const fileInput = document.getElementById('file-input');
  const drop = document.getElementById('drop');
  const filesList = document.getElementById('files');
  const statusEl = document.getElementById('status');
  const btnProcess = document.getElementById('process');
  const btnExample = document.getElementById('example');
  const btnClear = document.getElementById('clear');

  // Prefix-safe URL builder
  const api = (p) => new URL(p, window.location.href).toString();

  // Keep the edge session fresh while the page is open
  setInterval(() => {
    fetch(api('app-metadata'), { cache: 'no-store' }).catch(()=>{});
  }, 55_000);

  let files = [];

  function setStatus(msg, cls=''){ statusEl.className='status ' + cls; statusEl.textContent=msg; }

  function renderChips(){
    filesList.innerHTML = '';
    for (const f of files) {
      const chip = document.createElement('div');
      chip.className = 'chip';
      chip.textContent = `${f.name}`;
      filesList.appendChild(chip);
    }
  }

  function addFiles(fileList){
    for (const f of fileList){
      if (!f.type.startsWith('image/')) continue;
      files.push(f);
    }
    renderChips();
    setStatus(files.length ? `${files.length} image(s) ready.` : 'No valid images yet.');
  }

  drop.addEventListener('dragover', (e)=>{ e.preventDefault(); drop.classList.add('dragover'); });
  drop.addEventListener('dragleave', ()=> drop.classList.remove('dragover'));
  drop.addEventListener('drop', (e)=>{
    e.preventDefault(); drop.classList.remove('dragover');
    addFiles(e.dataTransfer.files);
  });
  drop.addEventListener('click', ()=> fileInput.click());
  fileInput.addEventListener('change', ()=> addFiles(fileInput.files));

  async function downloadZipFromResponse(resp){
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'results.zip';
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  }

  function postFilesOnce(sendFiles){
    const fd = new FormData();
    sendFiles.forEach(f => fd.append('files', new File([f], f.name, { type: f.type })));
    return fetch(api('segment'), {
      method:'POST',
      body: fd,
      cache: 'no-store',
      headers: { 'X-Requested-With': 'fetch' }
    });
  }
  async function postWithRetry(sendFiles){
    let resp = await postFilesOnce(sendFiles);
    if (resp.status === 403) {
      await new Promise(r => setTimeout(r, 600));
      resp = await postFilesOnce(sendFiles);
    }
    return resp;
  }

  btnProcess.addEventListener('click', async ()=>{
    if (!files.length){ setStatus('Please add at least one image.', 'warn'); return; }
    setStatus('Uploading & processing…');
    try{
      const chunkSize = 4;
      for (let i=0;i<files.length;i+=chunkSize){
        const chunk = files.slice(i, i+chunkSize);
        const resp = await postWithRetry(chunk);
        if (!resp.ok) {
          if (resp.status === 403) {
            setStatus('Please refresh the page and try again (403).', 'warn');
            return;
          }
          const msg = await resp.text();
          throw new Error(`${resp.status} ${resp.statusText} – ${msg}`);
        }
        setStatus('Done. Downloading ZIP…', 'ok');
        await downloadZipFromResponse(resp);
      }
      setStatus('Ready.');
      fileInput.value = '';
      files = [];
      renderChips();
    }catch(err){
      setStatus('Error: ' + (err?.message || err), 'warn');
      fileInput.value = '';
      files = [];
      renderChips();
    }
  });

  btnExample.addEventListener('click', async ()=>{
    try{
      setStatus('Running example…');
      const resp = await fetch(api('example'), { cache: 'no-store', headers: { 'X-Requested-With': 'fetch' } });
      if (!resp.ok) {
        if (resp.status === 403) {
          setStatus('Please refresh the page and try again (403).', 'warn');
          return;
        }
        const msg = await resp.text();
        throw new Error(`${resp.status} ${resp.statusText} – ${msg}`);
      }
      setStatus('Done. Downloading ZIP…', 'ok');
      await downloadZipFromResponse(resp);
      setStatus('Ready.');
    }catch(err){
      setStatus('Error: ' + (err?.message || err), 'warn');
    }
  });

  btnClear.addEventListener('click', ()=>{
    files = []; filesList.innerHTML = ''; fileInput.value = ''; setStatus('Cleared.');
  });
})();
</script>
</body>
</html>""")

# (Optional) a simple favicon to quiet 404s in logs
@app.get("/favicon.ico")
async def favicon():
    return HTMLResponse(status_code=204)


@app.get("/health", tags=["Utilities"])
async def health_check():
    """
    Liveness/readiness probe for the service.
    Returns component status (dirs, GPU availability, and model load state).
    """
    # ... (existing implementation unchanged) ...
    checks = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "components": {},
    }
    checks["components"]["upload_dir"] = "healthy" if os.path.exists(UPLOAD_DIR) else "missing"
    checks["components"]["results_dir"] = "healthy" if os.path.exists(RESULTS_DIR) else "missing"
    try:
        checks["components"]["gpu"] = "available" if torch.cuda.is_available() else "cpu_only"
    except Exception:
        checks["components"]["gpu"] = "unknown"
    try:
        global model
        checks["components"]["model"] = "loaded" if model is not None else "not_loaded"
    except Exception:
        checks["components"]["model"] = "error"
    return checks


@app.get("/metadata", tags=["Utilities"])
async def metadata_legacy():
    """Legacy alias for app info (kept for compatibility)."""
    return {
        "name": "MONAI Endoscopic Tool Segmentation",
        "description": "AI-powered segmentation of surgical tools in endoscopic images",
        "version": "1.0.0",
        "output": "zip-with-original-and-blended-images",
    }


@app.get("/app-metadata", tags=["Utilities"])
async def app_metadata():
    """Basic app info. Use this as a lightweight ping endpoint."""
    # Alias that avoids WAFs sensitive to '/metadata' routes.
    return {
        "name": "MONAI Endoscopic Tool Segmentation",
        "description": "AI-powered segmentation of surgical tools in endoscopic images",
        "version": "1.0.0",
        "output": "zip-with-original-and-blended-images",
    }


@app.get(
    "/example",
    tags=["Examples"],
    response_class=FileResponse,
    responses={
        200: {
            "description": "ZIP built from the bundled `./real` example images.",
            "content": {"application/zip": {"schema": {"type": "string", "format": "binary"}}},
        },
        403: {
            "description": "Transient edge/WAF block. In Swagger, set the 't' query param (any value) and retry, or refresh /docs and execute again."
        },
        404: {"description": "`./real` folder not found on server."},
        500: {"description": "Server error while generating example results."},
    },
)
async def get_example(
    background_tasks: BackgroundTasks,
):
    """
    Processes images in `./real` and returns a ZIP with the original and blended images.

    **Swagger tips (403 handling):**
    - If you receive **403 Forbidden** in `/docs`, refresh `/docs` and retry. The 403 originates from an upstream gateway and is typically transient.
    """
    # ... (existing implementation unchanged) ...
    clean_old_files(background_tasks)

    example_dir = os.path.join(BASE_DIR, "real")
    if not os.path.exists(example_dir):
        raise HTTPException(status_code=404, detail="Example directory not found")

    session_id = str(uuid.uuid4())
    results_path = os.path.join(RESULTS_DIR, session_id)
    os.makedirs(results_path, exist_ok=True)

    result_paths = get_results(example_dir, results_path)
    if not result_paths:
        raise HTTPException(status_code=500, detail="No results produced")

    zip_path = create_zip_from_results(result_paths, session_id)
    headers = {"Content-Disposition": f'attachment; filename="{session_id}_results.zip"'}
    return FileResponse(zip_path, media_type="application/zip", headers=headers)


# Preflight helpers: respond to OPTIONS explicitly
@app.options("/segment")
async def options_segment():
    return Response(status_code=204)

@app.options("/{path:path}")
async def options_any(path: str):
    # Helps certain proxies that require an explicit 204 to non-GETs
    return Response(status_code=204)


@app.post(
    "/segment",
    response_class=FileResponse,
    tags=["Segmentation"],
    responses={
        200: {
            "description": "ZIP with originals and blends.",
            "content": {
                "application/zip": {
                    "schema": {"type": "string", "format": "binary"}
                }
            },
        },
        400: {"description": "Bad request (no files or unsupported format)."},
        403: {
            "description": "Transient edge/WAF block. Refresh /docs and try again."
        },
        500: {"description": "Server error while processing images."},
    },
)
async def segment_image(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="One or more PNG/JPG images."),
):
    """
    Upload and process images; returns a ZIP with only `*_original` and `*_blend` files.

    **Swagger tips (403 handling):**
    - If you receive **403 Forbidden** in `/docs`, refresh `/docs` and retry. This 403 is from an upstream gateway and is safe to retry.

    **Returns**
    - `application/zip` with:
        - `results/<stem>/<stem>.png` (re-oriented original)
        - `results/<stem>/<stem>_blend.png` (overlay)
    """
    clean_old_files(background_tasks)

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    session_id = str(uuid.uuid4())
    log_processing_event("PROCESSING_START", session_id, {"num_files": len(files)})

    upload_path = os.path.join(UPLOAD_DIR, session_id)
    results_path = os.path.join(RESULTS_DIR, session_id)
    os.makedirs(upload_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    saved = []
    for i, f in enumerate(files):
        if not f.filename.lower().endswith((".jpg", ".jpeg", ".png")):
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {f.filename}")
        safe = f"image_{i}.jpg"
        dst = os.path.join(upload_path, safe)
        with open(dst, "wb") as out:
            out.write(await f.read())
        saved.append(dst)

    if not saved:
        raise HTTPException(status_code=400, detail="No valid image files uploaded")

    try:
        result_paths = get_results(upload_path, results_path)
        if not result_paths:
            raise RuntimeError("Pipeline produced no outputs")

        zip_path = create_zip_from_results(result_paths, session_id)
        log_processing_event("PROCESSING_SUCCESS", session_id, {"zip": os.path.basename(zip_path)})

        # FIX: proper f-string and quoting
        headers = {"Content-Disposition": f'attachment; filename="{session_id}_results.zip"'}
        return FileResponse(zip_path, media_type="application/zip", headers=headers)

    except Exception as e:
        log_processing_event("PROCESSING_ERROR", session_id, {"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Error processing images: {e}")





@app.on_event("shutdown")
async def shutdown_event():
    try:
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
            os.makedirs(UPLOAD_DIR, exist_ok=True)
        if os.path.exists(RESULTS_DIR):
            shutil.rmtree(RESULTS_DIR, ignore_errors=True)
            os.makedirs(RESULTS_DIR, exist_ok=True)
    except Exception as e:
        logger.warning(f"Shutdown cleanup error: {e}")


def main():
    # In production on HU, prefer proxy_headers=True (or CLI flags).
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=False)


if __name__ == "__main__":
    main()