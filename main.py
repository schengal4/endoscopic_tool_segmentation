# main.py  (simplified returns-as-zip version)

import os
import torch
import shutil
import uvicorn
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from typing import List, Dict
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
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
from fastapi import Request, Response  # add at the top with your other imports
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware



# ---- logging ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- app ----
app = FastAPI(
    title="MONAI Endoscopic Tool Segmentation API",
    description="API for segmenting surgical tools in endoscopic images",
    version="1.0.0"
)
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

# # If you still want to browse results in dev, you can keep this mount;
# # it is not used by any endpoint response anymore.
# app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

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
    Returns dict with keys like "<userfilename>_original" and "<userfilename>_blend".
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
                except RuntimeError as e:
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
    return HTMLResponse("""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>MONAI Endoscopic Tool Segmentation</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root { --bg:#0b1020; --card:#131a2b; --muted:#8ca3c7; --accent:#5cc8ff; --ok:#5CFF9E; --warn:#ffc857; }
    *{box-sizing:border-box} body{margin:0;background:var(--bg);color:#e6eefc;font-family:Inter,ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial}
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
    .example{background:#2c7a7b}
    .status{margin-top:16px;min-height:24px;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;color:var(--muted)}
    .status.ok{color:var(--ok)} .status.warn{color:var(--warn)}
    .small{font-size:12px;color:#9cb2d8;margin-top:8px}
    .footer{margin-top:18px;color:#6a81a8;font-size:12px}
    a:any-link{color:#9ed5ff}
    input[type=file]{display:none}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>MONAI Endoscopic Tool Segmentation</h1>
      <p class="sub">Upload endoscopic images. You’ll get a ZIP with corrected <strong>original</strong> images and <strong>blended</strong> overlays. No links, just a direct download.</p>

      <label for="file-input">
        <div id="drop" class="drop">
          <strong>Drag & drop</strong> images here, or <u>click to choose</u>.
          <div class="small">Accepted: .jpg .jpeg .png</div>
          <input id="file-input" type="file" accept=".jpg,.jpeg,.png" multiple>
          <div id="files" class="files"></div>
        </div>
      </label>

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

  // --- prefix-safe URL builder (KEY CHANGE) ---
  const api = (p) => new URL(p, window.location.href).toString();

  let files = [];

  function setStatus(msg, cls=''){ statusEl.className='status ' + cls; statusEl.textContent=msg; }
  function prettyBytes(n){
    if(!Number.isFinite(n)) return '';
    const u=['B','KB','MB','GB']; let i=0; while(n>=1024 && i<u.length-1){ n/=1024; i++; }
    return n.toFixed(n<10 && i>0 ? 1 : 0) + ' ' + u[i];
  }
  function renderChips(){
    filesList.innerHTML='';
    files.forEach(f=>{
      const el=document.createElement('div');
      el.className='chip';
      el.textContent = `${f.name} • ${prettyBytes(f.size)}`;
      filesList.appendChild(el);
    });
  }
  function addFiles(newFiles){
    const accepted = ['image/jpeg','image/png'];
    for(const f of newFiles){
      if(!accepted.includes(f.type)) { setStatus(`Skipped ${f.name} (unsupported type)`, 'warn'); continue; }
      files.push(f);
    }
    renderChips();
  }

  // drag & drop
  drop.addEventListener('dragover', e=>{ e.preventDefault(); drop.classList.add('dragover');});
  drop.addEventListener('dragleave', ()=> drop.classList.remove('dragover'));
  drop.addEventListener('drop', e=>{
    e.preventDefault(); drop.classList.remove('dragover');
    addFiles(e.dataTransfer.files);
  });

  fileInput.addEventListener('change', e=> addFiles(e.target.files));

  btnClear.addEventListener('click', ()=>{
    files = []; renderChips(); fileInput.value=''; setStatus('Cleared. Ready.');
  });

  async function downloadZipFromResponse(resp){
    const cd = resp.headers.get('Content-Disposition') || '';
    const match = cd.match(/filename="?([^"]+)"?/i);
    const name = match ? match[1] : `results_${Date.now()}.zip`;
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  }

  btnProcess.addEventListener('click', async ()=>{
    if(files.length === 0){ setStatus('Please add at least one image.', 'warn'); return; }
    try{
      setStatus('Uploading & processing…');
      const fd = new FormData();
      files.forEach(f => fd.append('files', f, f.name));
      // CHANGED: use api('segment') instead of '/segment'
      const resp = await fetch(api('segment'), { method:'POST', body: fd });
      if(!resp.ok){
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

  btnExample.addEventListener('click', async ()=>{
    try{
      setStatus('Running example…');
      // CHANGED: use api('example') instead of '/example'
      const resp = await fetch(api('example'));
      if(!resp.ok){
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
})();
</script>
</body>
</html>
    """)

# (Optional) a simple favicon to quiet 404s in logs
@app.get("/favicon.ico")
async def favicon():
    return HTMLResponse(status_code=204)

@app.get("/health")
async def health_check():
    checks = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "components": {}
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

@app.get("/metadata")
async def app_metadata():
    return {
        "name": "MONAI Endoscopic Tool Segmentation",
        "description": "AI-powered segmentation of surgical tools in endoscopic images",
        "version": "1.0.0",
        "output": "zip-with-original-and-blended-images"
    }

@app.get("/example")
async def get_example(background_tasks: BackgroundTasks):
    """
    Processes images in ./real and returns a ZIP containing only the original
    (corrected orientation) and blended images.
    """
    clean_old_files(background_tasks)

    example_dir = os.path.join(BASE_DIR, "real")
    if not os.path.exists(example_dir):
        raise HTTPException(status_code=404, detail="Example directory not found")

    # Per run session
    session_id = str(uuid.uuid4())
    results_path = os.path.join(RESULTS_DIR, session_id)
    os.makedirs(results_path, exist_ok=True)

    result_paths = get_results(example_dir, results_path)
    if not result_paths:
        raise HTTPException(status_code=500, detail="No results produced")

    zip_path = create_zip_from_results(result_paths, session_id)

    headers = {"Content-Disposition": f'attachment; filename="{session_id}_results.zip"'}
    return FileResponse(zip_path, media_type="application/zip", headers=headers)

@app.options("/segment")
async def options_segment():
    return Response(status_code=204)

@app.post("/segment")
async def segment_image(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    Upload and process images; returns a ZIP with only *_original and *_blend files.
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

    # Run inference and build zip
    try:
        result_paths = get_results(upload_path, results_path)
        if not result_paths:
            raise RuntimeError("Pipeline produced no outputs")
        zip_path = create_zip_from_results(result_paths, session_id)
        log_processing_event("PROCESSING_SUCCESS", session_id, {"zip": os.path.basename(zip_path)})

        headers = {"Content-Disposition": f'attachment; filename="{session_id}_results.zip"'}
        return FileResponse(zip_path, media_type="application/zip", headers=headers)

    except Exception as e:
        log_processing_event("PROCESSING_ERROR", session_id, {"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Error processing images: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    # Clean up ephemeral dirs at shutdown
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
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)

if __name__ == "__main__":
    main()
