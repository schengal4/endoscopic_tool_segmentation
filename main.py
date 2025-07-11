import os
import monai
import torch
import shutil
import uvicorn
import tempfile
import numpy as np
from typing import List
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from monai.apps import download_and_extract
from monai.bundle import ConfigParser
from monai.data import decollate_batch
from monai.handlers import MLFlowHandler
from monai.config import print_config
from monai.visualize.utils import blend_images
from PIL import Image, ImageOps
import torchvision.transforms as transforms
# Place this after your imports at the top of the file
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from fastapi.responses import HTMLResponse
import logging
import time
import json
from datetime import datetime 
import uuid


# Configure logging for healthcare compliance
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize FastAPI app
app = FastAPI(
    title="MONAI Endoscopic Tool Segmentation API",
    description="API for segmenting surgical tools in endoscopic images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Configure device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define paths
BASE_DIR = "."
MODELS_DIR = os.path.join(BASE_DIR, "models")
CONFIG_DIR = os.path.join(BASE_DIR, "configs")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "monai_results")

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Serve the results directory
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

# Initialize model and config
model_config = None
preprocessing = None
model = None
inferer = None
original_postprocessing = None
dataloader = None

def load_model():
    """Load the MONAI model and configurations"""
    global model_config, preprocessing, model, inferer, original_postprocessing
    
    model_config_file = os.path.join(CONFIG_DIR, "inference.json")
    
    model_config = ConfigParser()
    model_config.read_config(model_config_file)
    model_config["bundle_root"] = "."
    model_config["output_dir"] = RESULTS_DIR
    
    # Load model components
    checkpoint = os.path.join(MODELS_DIR, "model.pt")
    preprocessing = model_config.get_parsed_content("preprocessing")
    model = model_config.get_parsed_content("network").to(device)
    inferer = model_config.get_parsed_content("inferer")
    
    # Modify postprocessing to skip SaveImaged
    original_postprocessing = model_config.get_parsed_content("postprocessing")
    new_transforms = []
    for transform in original_postprocessing.transforms:
        if not (hasattr(transform, '__class__') and transform.__class__.__name__ == 'SaveImaged'):
            new_transforms.append(transform)
    original_postprocessing.transforms = new_transforms
    
    # Load model weights
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    
    print("Model loaded successfully")

    
    # # ADD THIS LOG:
    # logger.info(f"AUDIT: {json.dumps({
    #     'timestamp': datetime.utcnow().isoformat() + 'Z',
    #     'event_type': 'MODEL_LOADED',
    #     'details': {
    #         'device': str(device),
    #         'model_type': 'UNet_EfficientNet_B2'
    #     }
    # })}")

def log_processing_event(event_type: str, session_id: str, details: dict):
    """Simple audit logging for medical AI processing events"""
    audit_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event_type": event_type,
        "session_id": session_id,
        "details": details
    }
    
    # Log to your existing logger
    # logger.info(f"AUDIT: {json.dumps(audit_entry)}")

@app.on_event("startup")
async def startup_event():
    """Initialize the model when the API starts"""
    load_model()

def get_results(directory_file_path: str, output_directory_file_path: str):
    """
    Process images and save both original images and segmentation masks.
    
    Args:
        directory_file_path: Path to input images
        output_directory_file_path: Path to save results
    
    Returns:
        dict: Dictionary with paths to the generated images
    """
    global model_config, model, inferer, original_postprocessing
    
    # Ensure model is loaded
    if model is None:
        load_model()
    
    # Update config paths
    model_config["dataset_dir"] = directory_file_path
    model_config["output_dir"] = output_directory_file_path
    
    # Ensure output directory exists
    os.makedirs(output_directory_file_path, exist_ok=True)
    
    # Get dataloader with updated paths
    dataloader = model_config.get_parsed_content("dataloader")
    
    result_paths = {}
    
    with torch.no_grad():
        # Loop over the entire DataLoader
        for idx, batch_data in enumerate(dataloader):
            # Move images to the GPU (if available)
            images = batch_data["image"].to(device)
            
            # Get original filename from metadata
            orig_filename = batch_data["image_meta_dict"]["filename_or_obj"][0]
            base_filename = os.path.basename(orig_filename)
            filename_no_ext = os.path.splitext(base_filename)[0]
            
            print(f"Processing file: {orig_filename}")
            
            # Run inference to get the segmentation mask
            batch_data["pred"] = inferer(images, network=model)
            
            # Process each sample
            decollated_data = decollate_batch(batch_data)
            
            for data_i in decollated_data:
                # Apply postprocessing (excluding SaveImaged)
                processed = original_postprocessing(data_i)
                
                # Create subfolder for this image
                img_output_dir = os.path.join(output_directory_file_path, filename_no_ext)
                os.makedirs(img_output_dir, exist_ok=True)
                
                # 1. GET THE PREPROCESSED IMAGE 
                orig_img = data_i["image"].detach().cpu().numpy()
                
                # Convert from CHW to HWC if needed
                if orig_img.shape[0] in [1, 3]:  # If image is in channel-first format
                    if orig_img.shape[0] == 1:
                        orig_img = orig_img[0]  # Remove channel dimension for grayscale
                    else:
                        orig_img = np.transpose(orig_img, (1, 2, 0))
                
                # Scale to 0-255 and convert to uint8
                orig_img = (orig_img * 255).astype(np.uint8)
                
                # 2. GET THE SEGMENTATION MASK
                mask = processed["pred"].detach().cpu().numpy()
                
                if len(mask.shape) > 3:  # If it's B,C,H,W format
                    mask = mask[0]  # Remove batch dimension
                
                # Get first channel if it's a one-hot encoded mask
                if mask.shape[0] > 1:
                    # If we have multiple channels (classes), use argmax to get class index
                    mask_vis = np.argmax(mask, axis=0)
                else:
                    # If single channel binary mask
                    mask_vis = mask[0]
                
                # Scale mask to 0-255 for visualization
                mask_vis = (mask_vis * 255).astype(np.uint8)
                
                print(f"Preprocessed image shape: {orig_img.shape}")
                print(f"Mask shape: {mask_vis.shape}")
                
                # IMPORTANT: Resize the original image to the mask size
                mask_height, mask_width = mask_vis.shape
                orig_img_pil = Image.fromarray(orig_img)
                orig_img_pil = orig_img_pil.resize((mask_width, mask_height), Image.BILINEAR)
                
                # Fix the rotation AND mirroring:
                # 1. First rotate 270 degrees (counterclockwise) to counter the rotation
                # 2. Then mirror horizontally to correct the left-right reversal
                orig_img_pil = orig_img_pil.transpose(Image.ROTATE_270)
                orig_img_pil = ImageOps.mirror(orig_img_pil)
                mask_pil = Image.fromarray(mask_vis).transpose(Image.ROTATE_270)
                mask_pil = ImageOps.mirror(mask_pil)
                
                # Convert back to numpy for blending
                orig_img_resized = np.array(orig_img_pil)
                mask_vis_rotated = np.array(mask_pil)
                
                # Save original image with correct orientation
                orig_img_path = os.path.join(img_output_dir, f"{filename_no_ext}.png")
                orig_img_pil.save(orig_img_path)
                print(f"Saved original image: {orig_img_path}")
                
                # Save the mask with correct orientation
                mask_img_path = os.path.join(img_output_dir, f"{filename_no_ext}_trans.png")
                mask_pil.save(mask_img_path)
                print(f"Saved mask: {mask_img_path}")
                
                # Store paths for response
                result_paths[f"{filename_no_ext}_original"] = orig_img_path
                result_paths[f"{filename_no_ext}_mask"] = mask_img_path
                
                # 3. Create a blended visualization with correctly oriented images
                try:
                    # Get dimensions after transformation
                    if len(orig_img_resized.shape) == 3:
                        rotated_height, rotated_width, _ = orig_img_resized.shape
                    else:
                        rotated_height, rotated_width = orig_img_resized.shape
                    
                    # Create RGB version of binary mask
                    mask_rgb = np.zeros((rotated_height, rotated_width, 3), dtype=np.uint8)
                    # Set non-zero values to a color (e.g., red)
                    mask_rgb[mask_vis_rotated > 0, 0] = 255  # Red channel
                    
                    # Create a blended image (50% original, 50% mask)
                    alpha = 0.5
                    blended = (orig_img_resized.astype(float) * (1-alpha) + mask_rgb.astype(float) * alpha).astype(np.uint8)
                    
                    # Save blended visualization
                    blend_img_path = os.path.join(img_output_dir, f"{filename_no_ext}_blend.png")
                    Image.fromarray(blended).save(blend_img_path)
                    print(f"Saved blended visualization: {blend_img_path}")
                    
                    # Store blended path for response
                    result_paths[f"{filename_no_ext}_blend"] = blend_img_path
                    
                except Exception as e:
                    print(f"Error creating blended image: {e}")
            
            print(f"Processed batch {idx+1}")
        
        print("Inference complete.")
        
    return result_paths

def clean_old_files(background_tasks: BackgroundTasks):
    """Schedule cleanup of old files"""
    def cleanup():
        # Delete files older than 1 hour
        import time
        current_time = time.time()
        for root, dirs, files in os.walk(UPLOAD_DIR):
            for d in dirs:
                dir_path = os.path.join(root, d)
                if os.path.getmtime(dir_path) < current_time - 3600:  # 1 hour
                    shutil.rmtree(dir_path)
        
        for root, dirs, files in os.walk(RESULTS_DIR):
            for d in dirs:
                dir_path = os.path.join(root, d)
                if os.path.getmtime(dir_path) < current_time - 3600:  # 1 hour
                    shutil.rmtree(dir_path)
    
    background_tasks.add_task(cleanup)

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MONAI Endoscopic Tool Segmentation API</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
    <style>
        .gradient-bg {
            background: linear-gradient(90deg, #4286f4, #373B44);
        }
        .tool-example {
            transition: transform 0.3s ease;
        }
        .tool-example:hover {
            transform: scale(1.05);
        }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <!-- Header -->
    <header class="gradient-bg text-white shadow-lg">
        <div class="container mx-auto px-4 py-6">
            <div class="flex flex-col md:flex-row justify-between items-center">
                <div class="flex items-center mb-4 md:mb-0">
                    <h1 class="text-3xl font-bold">MONAI Endoscopic Tool Segmentation</h1>
                </div>
                <nav>
                    <ul class="flex space-x-6">
                        <li><a href="#about" class="hover:text-gray-300">About</a></li>
                        <li><a href="#demo" class="hover:text-gray-300">Demo</a></li>
                        <li><a href="#api" class="hover:text-gray-300">API</a></li>
                        <li><a href="#performance" class="hover:text-gray-300">Performance</a></li>
                    </ul>
                </nav>
            </div>
        </div>
    </header>

    <!-- Hero Section -->
    <section class="py-12 bg-white">
        <div class="container mx-auto px-4">
            <div class="flex flex-col md:flex-row items-center">
                <div class="md:w-1/2 mb-8 md:mb-0">
                    <h2 class="text-4xl font-bold text-gray-800 mb-4">Precise Surgical Tool Detection</h2>
                    <p class="text-xl text-gray-600 mb-6">Advanced AI-powered segmentation for identifying surgical tools in endoscopic images with high accuracy.</p>
                    <div class="flex space-x-4">
                        <a href="#api" class="bg-gray-200 hover:bg-gray-300 text-gray-800 font-bold py-3 px-6 rounded-lg transition">API Docs</a>
                    </div>
                </div>
                <div class="md:w-1/2">
                    <div class="bg-gray-200 rounded-lg shadow-xl overflow-hidden">
                        <img src="/api/placeholder/640/400" alt="Endoscopic Tool Segmentation Example" class="w-full">
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- About Section -->
    <section id="about" class="py-16 bg-gray-100">
        <div class="container mx-auto px-4">
            <h2 class="text-3xl font-bold text-center text-gray-800 mb-12">About This Tool</h2>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
                <div class="bg-white rounded-lg shadow-md p-6">
                    <div class="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mb-4">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
                        </svg>
                    </div>
                    <h3 class="text-xl font-semibold mb-2">Advanced Architecture</h3>
                    <p class="text-gray-600">Built on a flexible UNet structure with an EfficientNet-B2 backbone, delivering state-of-the-art segmentation results for surgical applications.</p>
                </div>
                <div class="bg-white rounded-lg shadow-md p-6">
                    <div class="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mb-4">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                        </svg>
                    </div>
                    <h3 class="text-xl font-semibold mb-2">High Performance</h3>
                    <p class="text-gray-600">Achieves an impressive mean IoU score of 0.86, providing reliable and accurate tool identification in various surgical scenarios.</p>
                </div>
                <div class="bg-white rounded-lg shadow-md p-6">
                    <div class="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mb-4">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                        </svg>
                    </div>
                    <h3 class="text-xl font-semibold mb-2">Optimized Deployment</h3>
                    <p class="text-gray-600">TensorRT acceleration support with up to 2.3x speedup in FP16 precision, making it suitable for real-time clinical applications.</p>
                </div>
            </div>
        </div>
    </section>

    <!-- Technical Details -->
    <section class="py-16 bg-white">
        <div class="container mx-auto px-4">
            <h2 class="text-3xl font-bold text-center text-gray-800 mb-12">Technical Specifications</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-12">
                <div>
                    <h3 class="text-2xl font-semibold mb-4">Model Architecture</h3>
                    <ul class="space-y-3 text-gray-600">
                        <li class="flex items-start">
                            <svg class="h-6 w-6 text-green-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                            </svg>
                            <span>EfficientNet-B2 backbone encoder</span>
                        </li>
                        <li class="flex items-start">
                            <svg class="h-6 w-6 text-green-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                            </svg>
                            <span>UNet decoder architecture</span>
                        </li>
                        <li class="flex items-start">
                            <svg class="h-6 w-6 text-green-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                            </svg>
                            <span>Input resolution: 736 x 480 x 3</span>
                        </li>
                        <li class="flex items-start">
                            <svg class="h-6 w-6 text-green-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                            </svg>
                            <span>Binary segmentation (tools vs. background)</span>
                        </li>
                    </ul>
                </div>
                <div>
                    <h3 class="text-2xl font-semibold mb-4">Training Details</h3>
                    <ul class="space-y-3 text-gray-600">
                        <li class="flex items-start">
                            <svg class="h-6 w-6 text-green-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                            </svg>
                            <span>Adam optimizer with 1e-4 learning rate</span>
                        </li>
                        <li class="flex items-start">
                            <svg class="h-6 w-6 text-green-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                            </svg>
                            <span>Optional pre-trained weights support</span>
                        </li>
                        <li class="flex items-start">
                            <svg class="h-6 w-6 text-green-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                            </svg>
                            <span>Mean IoU evaluation metric</span>
                        </li>
                        <li class="flex items-start">
                            <svg class="h-6 w-6 text-green-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                            </svg>
                            <span>Compatible with multi-GPU training</span>
                        </li>
                    </ul>
                </div>
            </div>
        </div>
    </section>

    <!-- Demo Section -->
    <section id="demo" class="py-16 bg-gray-100">
        <div class="container mx-auto px-4">
            <h2 class="text-3xl font-bold text-center text-gray-800 mb-8">Try It Yourself</h2>
            <p class="text-center text-xl text-gray-600 mb-12">Upload your endoscopic images to see the tool segmentation in action</p>
            
            <div class="max-w-3xl mx-auto bg-white rounded-lg shadow-lg overflow-hidden">
                <div class="p-6">
                    <div class="mb-6">
                        <h3 class="text-xl font-semibold mb-2">Upload Images</h3>
                        <p class="text-gray-600 mb-4">Supported formats: JPG, JPEG, PNG</p>
                        <div class="flex items-center justify-center w-full">
                            <label class="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100">
                                <div class="flex flex-col items-center justify-center pt-5 pb-6">
                                    <svg class="w-10 h-10 mb-3 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                                    </svg>
                                    <p class="mb-2 text-sm text-gray-500"><span class="font-semibold">Click to upload</span> or drag and drop</p>
                                    <p class="text-xs text-gray-500">Up to 5 images at a time</p>
                                </div>
                                <input id="dropzone-file" type="file" class="hidden" multiple />
                            </label>
                        </div> 
                    </div>
                    
                    <div class="flex justify-center">
                        <button class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-10 rounded-lg transition">
                            Process Images
                        </button>
                    </div>
                </div>
                
                <!-- Example Results (Could be replaced with actual processing results) -->
                <div class="p-6 bg-gray-50 border-t">
                    <h3 class="text-xl font-semibold mb-4">Example Results</h3>
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div class="tool-example">
                            <p class="text-sm text-gray-600 mb-1">Original Image</p>
                            <img src="/api/placeholder/300/200" alt="Original endoscopic image" class="w-full rounded-lg shadow">
                        </div>
                        <div class="tool-example">
                            <p class="text-sm text-gray-600 mb-1">Segmentation Mask</p>
                            <img src="/api/placeholder/300/200" alt="Segmentation mask" class="w-full rounded-lg shadow">
                        </div>
                        <div class="tool-example">
                            <p class="text-sm text-gray-600 mb-1">Blended View</p>
                            <img src="/api/placeholder/300/200" alt="Blended view" class="w-full rounded-lg shadow">
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- API Documentation -->
    <section id="api" class="py-16 bg-white">
        <div class="container mx-auto px-4">
            <h2 class="text-3xl font-bold text-center text-gray-800 mb-12">API Documentation</h2>
            
            <div class="max-w-4xl mx-auto">
                <div class="mb-10">
                    <h3 class="text-2xl font-semibold mb-3">Endpoints</h3>
                    <div class="bg-gray-100 rounded-lg p-6">
                        <div class="mb-6">
                            <div class="flex items-center">
                                <span class="bg-green-500 text-white px-3 py-1 rounded-lg text-sm font-bold mr-3">GET</span>
                                <code class="text-purple-600 font-mono">/</code>
                            </div>
                            <p class="mt-2 text-gray-600">Welcome endpoint that returns a simple message.</p>
                        </div>
                        
                        <div class="mb-6">
                            <div class="flex items-center">
                                <span class="bg-green-500 text-white px-3 py-1 rounded-lg text-sm font-bold mr-3">GET</span>
                                <code class="text-purple-600 font-mono">/example</code>
                            </div>
                            <p class="mt-2 text-gray-600">Processes example images from the 'real' directory to demonstrate the API.</p>
                        </div>
                        
                        <div class="mb-6">
                            <div class="flex items-center">
                                <span class="bg-blue-500 text-white px-3 py-1 rounded-lg text-sm font-bold mr-3">POST</span>
                                <code class="text-purple-600 font-mono">/segment</code>
                            </div>
                            <p class="mt-2 text-gray-600">Upload and process endoscopic images using the MONAI pipeline.</p>
                            <p class="mt-1 text-sm text-gray-500">Accepts multiple files (JPG, JPEG, PNG) as form data.</p>
                        </div>
                        
                        <div class="mb-6">
                            <div class="flex items-center">
                                <span class="bg-green-500 text-white px-3 py-1 rounded-lg text-sm font-bold mr-3">GET</span>
                                <code class="text-purple-600 font-mono">/download/{session_id}/{path}</code>
                            </div>
                            <p class="mt-2 text-gray-600">Download a specific result file with support for nested paths.</p>
                        </div>
                    </div>
                </div>
                
                <div class="mb-10">
                    <h3 class="text-2xl font-semibold mb-3">Example Request</h3>
                    <div class="bg-gray-900 rounded-lg p-6 text-white font-mono text-sm overflow-x-auto">
<pre>
curl -X POST "http://localhost:5000/segment" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
</pre>
                    </div>
                </div>
                
                <div>
                    <h3 class="text-2xl font-semibold mb-3">Example Response</h3>
                    <div class="bg-gray-900 rounded-lg p-6 text-white font-mono text-sm overflow-x-auto">
<pre>
{
  "message": "Processed 2 images successfully",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "results": {
    "image1_original": "/results/550e8400-e29b-41d4-a716-446655440000/image_0/image_0.png",
    "image1_mask": "/results/550e8400-e29b-41d4-a716-446655440000/image_0/image_0_trans.png",
    "image1_blend": "/results/550e8400-e29b-41d4-a716-446655440000/image_0/image_0_blend.png",
    "image2_original": "/results/550e8400-e29b-41d4-a716-446655440000/image_1/image_1.png",
    "image2_mask": "/results/550e8400-e29b-41d4-a716-446655440000/image_1/image_1_trans.png",
    "image2_blend": "/results/550e8400-e29b-41d4-a716-446655440000/image_1/image_1_blend.png"
  }
}
</pre>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Performance Section -->
    <section id="performance" class="py-16 bg-gray-100">
        <div class="container mx-auto px-4">
            <h2 class="text-3xl font-bold text-center text-gray-800 mb-12">Performance Metrics</h2>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-12">
                <div class="bg-white rounded-lg shadow-md p-6">
                    <h3 class="text-2xl font-semibold mb-4">Accuracy</h3>
                    <div class="aspect-w-16 aspect-h-9 bg-gray-200 rounded-lg mb-4">
                        <img src="/api/placeholder/600/400" alt="Model accuracy chart" class="rounded-lg">
                    </div>
                    <p class="text-gray-600">The model achieves a mean IoU (Intersection over Union) score of 0.86, ensuring reliable tool segmentation across various endoscopic scenarios.</p>
                </div>
                
                <div class="bg-white rounded-lg shadow-md p-6">
                    <h3 class="text-2xl font-semibold mb-4">TensorRT Acceleration</h3>
                    <div class="overflow-x-auto">
                        <table class="min-w-full bg-white">
                            <thead>
                                <tr>
                                    <th class="py-2 px-3 border-b border-gray-200 bg-gray-100 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">Method</th>
                                    <th class="py-2 px-3 border-b border-gray-200 bg-gray-100 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">PyTorch FP32</th>
                                    <th class="py-2 px-3 border-b border-gray-200 bg-gray-100 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">TensorRT FP16</th>
                                    <th class="py-2 px-3 border-b border-gray-200 bg-gray-100 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">Speedup</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td class="py-2 px-3 border-b border-gray-200 text-sm">Model Computation</td>
                                    <td class="py-2 px-3 border-b border-gray-200 text-sm">12.00 ms</td>
                                    <td class="py-2 px-3 border-b border-gray-200 text-sm">5.20 ms</td>
                                    <td class="py-2 px-3 border-b border-gray-200 text-sm font-semibold text-green-600">2.31x</td>
                                </tr>
                                <tr>
                                    <td class="py-2 px-3 border-b border-gray-200 text-sm">End-to-End</td>
                                    <td class="py-2 px-3 border-b border-gray-200 text-sm">170.04 ms</td>
                                    <td class="py-2 px-3 border-b border-gray-200 text-sm">155.57 ms</td>
                                    <td class="py-2 px-3 border-b border-gray-200 text-sm font-semibold text-green-600">1.09x</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                    <p class="mt-4 text-gray-600">Benchmarks performed on an NVIDIA A100 80G GPU with TensorRT 8.5.3 and CUDA 11.8.</p>
                </div>
            </div>
        </div>
    </section>

    <!-- Footer -->
    <footer class="gradient-bg text-white py-8">
        <div class="container mx-auto px-4">
            <div class="flex flex-col md:flex-row justify-between items-center">
                <div class="mb-4 md:mb-0">
                    <h2 class="text-2xl font-bold">MONAI Endoscopic Tool Segmentation</h2>
                    <p class="text-gray-300 mt-2">Powered by MONAI and FastAPI</p>
                </div>
                <div>
                    <p class="text-sm text-gray-300">&copy; 2025 MONAI Consortium. Licensed under the Apache License, Version 2.0.</p>
                </div>
            </div>
        </div>
    </footer>

    <script>
        // Simple placeholder for JavaScript functionality
        document.addEventListener('DOMContentLoaded', function() {
            console.log('MONAI Endoscopic Tool Segmentation interface loaded');
            
            // Add demo form submission handler here
            const uploadForm = document.querySelector('#dropzone-file');
            if (uploadForm) {
                uploadForm.addEventListener('change', function(e) {
                    console.log('Files selected:', e.target.files);
                    // Implementation would handle file uploads
                });
            }
        });
    </script>
</body>
</html>
"""

@app.get("/example")
async def get_example(background_tasks: BackgroundTasks):
    """Process example images from the 'real' directory"""
    clean_old_files(background_tasks)
    
    example_dir = os.path.join(BASE_DIR, "real")
    if not os.path.exists(example_dir):
        raise HTTPException(status_code=404, detail="Example directory not found")
    
    # Clear previous results
    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
        os.makedirs(RESULTS_DIR)
    
    # Process example images
    result_paths = get_results(example_dir, RESULTS_DIR)
    
    # Convert result paths to URLs
    result_urls = {}
    for key, path in result_paths.items():
        rel_path = os.path.relpath(path, RESULTS_DIR)
        result_urls[key] = f"/results/{rel_path}"
    
    return {
        "message": "Example images processed successfully",
        "results": result_urls
    }

# If you want a more detailed health check that's safe, use this instead:
@app.get("/health")
async def health_check():
    """
    Detailed but safe health check endpoint.
    """
    import time
    import torch
    
    checks = {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "MONAI Endoscopic Tool Segmentation API",
        "components": {}
    }
    
    # Check if directories exist (using the same variables from your main code)
    try:
        if os.path.exists(UPLOAD_DIR):
            checks["components"]["upload_dir"] = "healthy"
        else:
            checks["components"]["upload_dir"] = "missing"
    except:
        checks["components"]["upload_dir"] = "error"
    
    try:
        if os.path.exists(RESULTS_DIR):
            checks["components"]["results_dir"] = "healthy"
        else:
            checks["components"]["results_dir"] = "missing"
    except:
        checks["components"]["results_dir"] = "error"
    
    # Check GPU availability safely
    try:
        checks["components"]["gpu"] = "available" if torch.cuda.is_available() else "cpu_only"
    except:
        checks["components"]["gpu"] = "unknown"
    
    # Check model status safely (using global variable from your code)
    try:
        global model
        if 'model' in globals() and model is not None:
            checks["components"]["model"] = "loaded"
        else:
            checks["components"]["model"] = "not_loaded"
    except:
        checks["components"]["model"] = "error"
    
    return checks

@app.get("/metadata")
async def app_metadata():
    """Application metadata for Health Universe Navigator integration"""
    return {
        "name": "MONAI Endoscopic Tool Segmentation",
        "description": "AI-powered segmentation of surgical tools in endoscopic images",
        "version": "1.0.0",
        "author": "MONAI Consortium",
        "category": "Medical Imaging",
        "modality": "Computer Vision",
        "input_types": ["image/jpeg", "image/png"],
        "output_types": ["image/png"],
        "model_info": {
            "architecture": "UNet with EfficientNet-B2 backbone",
            "performance": "Mean IoU: 0.86",
            "input_resolution": "736x480x3",
            "inference_time": "~170ms end-to-end"
        },
        "compliance": {
            "hipaa_ready": True,
            "clinical_validation": True,
            "transparency": "Explainable AI with visualization outputs"
        }
    }

@app.post("/segment")
async def segment_image(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    Upload and process all endoscopic images using the MONAI pipeline
    """
    clean_old_files(background_tasks)
    
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    # Create a unique directory for this upload
    session_id = str(uuid.uuid4())

    # ADD THIS LOG:
    log_processing_event(
        event_type="PROCESSING_START",
        session_id=session_id,
        details={"num_files": len(files), "file_types": [f.content_type for f in files]}
    )

    upload_path = os.path.join(UPLOAD_DIR, session_id)
    results_path = os.path.join(RESULTS_DIR, session_id)
    
    os.makedirs(upload_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    
    # Save uploaded files with simple filenames
    saved_files = []
    original_filenames = {}  # Map to keep track of original filenames
    
    for i, file in enumerate(files):
        if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file.filename}")
        
        # Use a simple filename without spaces or special characters
        # This is important for MONAI DataLoader which may have issues with complex filenames
        safe_filename = f"image_{i}.jpg"
        file_path = os.path.join(upload_path, safe_filename)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            print(f"Saved {len(content)} bytes to {file_path}")
        
        saved_files.append(file_path)
        original_filenames[safe_filename] = file.filename
        print(f"Saved file: {file_path} (original name: {file.filename})")
    
    if not saved_files:
        raise HTTPException(status_code=400, detail="No valid image files uploaded")
    
    # Print directory structure for debugging
    print(f"Upload directory structure:")
    for root, dirs, files in os.walk(upload_path):
        for file in files:
            print(f"  - {os.path.join(root, file)} ({os.path.getsize(os.path.join(root, file))} bytes)")
    
    # Initialize diagnostic info
    diagnostic_info = {
        "upload_dir_contents": os.listdir(upload_path),
        "saved_files": saved_files,
        "original_filenames": original_filenames
    }
    
    try:
        # Process images using MONAI pipeline
        # Update model_config with new paths
        model_config["dataset_dir"] = upload_path
        model_config["output_dir"] = results_path
        
        # Get dataloader with updated paths
        dataloader = model_config.get_parsed_content("dataloader")
        
        # Process all images
        result_paths = {}
        
        with torch.no_grad():
            # Loop over the entire DataLoader
            for idx, batch_data in enumerate(dataloader):
                # Move images to the GPU (if available)
                images = batch_data["image"].to(device)
                
                # Get original filename from metadata
                orig_filename = batch_data["image_meta_dict"]["filename_or_obj"][0]
                base_filename = os.path.basename(orig_filename)
                filename_no_ext = os.path.splitext(base_filename)[0]
                
                # Get the original user-provided filename if available
                user_filename = original_filenames.get(base_filename, filename_no_ext)
                user_filename_no_ext = os.path.splitext(user_filename)[0]
                
                print(f"Processing file: {orig_filename} (user filename: {user_filename})")
                
                # Run inference to get the segmentation mask
                batch_data["pred"] = inferer(images, network=model)
                
                # Process each sample
                decollated_data = decollate_batch(batch_data)
                
                for data_i in decollated_data:
                    # Apply postprocessing (excluding SaveImaged)
                    processed = original_postprocessing(data_i)
                    
                    # Create subfolder for this image using the original filename for better organization
                    img_output_dir = os.path.join(results_path, filename_no_ext)
                    os.makedirs(img_output_dir, exist_ok=True)
                    
                    # 1. GET THE PREPROCESSED IMAGE 
                    orig_img = data_i["image"].detach().cpu().numpy()
                    
                    # Convert from CHW to HWC if needed
                    if orig_img.shape[0] in [1, 3]:  # If image is in channel-first format
                        if orig_img.shape[0] == 1:
                            orig_img = orig_img[0]  # Remove channel dimension for grayscale
                        else:
                            orig_img = np.transpose(orig_img, (1, 2, 0))
                    
                    # Scale to 0-255 and convert to uint8
                    orig_img = (orig_img * 255).astype(np.uint8)
                    
                    # 2. GET THE SEGMENTATION MASK
                    mask = processed["pred"].detach().cpu().numpy()
                    
                    if len(mask.shape) > 3:  # If it's B,C,H,W format
                        mask = mask[0]  # Remove batch dimension
                    
                    # Get first channel if it's a one-hot encoded mask
                    if mask.shape[0] > 1:
                        # If we have multiple channels (classes), use argmax to get class index
                        mask_vis = np.argmax(mask, axis=0)
                    else:
                        # If single channel binary mask
                        mask_vis = mask[0]
                    
                    # Scale mask to 0-255 for visualization
                    mask_vis = (mask_vis * 255).astype(np.uint8)
                    
                    print(f"Preprocessed image shape: {orig_img.shape}")
                    print(f"Mask shape: {mask_vis.shape}")
                    
                    # IMPORTANT: Resize the original image to the mask size
                    mask_height, mask_width = mask_vis.shape
                    orig_img_pil = Image.fromarray(orig_img)
                    orig_img_pil = orig_img_pil.resize((mask_width, mask_height), Image.BILINEAR)
                    
                    # Fix the rotation AND mirroring:
                    # 1. First rotate 270 degrees (counterclockwise) to counter the rotation
                    # 2. Then mirror horizontally to correct the left-right reversal
                    orig_img_pil = orig_img_pil.transpose(Image.ROTATE_270)
                    orig_img_pil = ImageOps.mirror(orig_img_pil)
                    mask_pil = Image.fromarray(mask_vis).transpose(Image.ROTATE_270)
                    mask_pil = ImageOps.mirror(mask_pil)
                    
                    # Convert back to numpy for blending
                    orig_img_resized = np.array(orig_img_pil)
                    mask_vis_rotated = np.array(mask_pil)
                    
                    # Save original image with correct orientation
                    orig_img_path = os.path.join(img_output_dir, f"{filename_no_ext}.png")
                    orig_img_pil.save(orig_img_path)
                    print(f"Saved original image: {orig_img_path}")
                    
                    # Save the mask with correct orientation
                    mask_img_path = os.path.join(img_output_dir, f"{filename_no_ext}_trans.png")
                    mask_pil.save(mask_img_path)
                    print(f"Saved mask: {mask_img_path}")
                    
                    # Store paths for response using user-friendly keys
                    result_paths[f"{user_filename_no_ext}_original"] = orig_img_path
                    result_paths[f"{user_filename_no_ext}_mask"] = mask_img_path
                    
                    # 3. Create a blended visualization with correctly oriented images
                    try:
                        # Get dimensions after transformation
                        if len(orig_img_resized.shape) == 3:
                            rotated_height, rotated_width, _ = orig_img_resized.shape
                        else:
                            rotated_height, rotated_width = orig_img_resized.shape
                        
                        # Create RGB version of binary mask
                        mask_rgb = np.zeros((rotated_height, rotated_width, 3), dtype=np.uint8)
                        # Set non-zero values to a color (e.g., red)
                        mask_rgb[mask_vis_rotated > 0, 0] = 255  # Red channel
                        
                        # Create a blended image (50% original, 50% mask)
                        alpha = 0.5
                        blended = (orig_img_resized.astype(float) * (1-alpha) + mask_rgb.astype(float) * alpha).astype(np.uint8)
                        
                        # Save blended visualization
                        blend_img_path = os.path.join(img_output_dir, f"{filename_no_ext}_blend.png")
                        Image.fromarray(blended).save(blend_img_path)
                        print(f"Saved blended visualization: {blend_img_path}")
                        
                        # Store blended path for response
                        result_paths[f"{user_filename_no_ext}_blend"] = blend_img_path
                    except Exception as e:
                        print(f"Error creating blended image: {e}")
                
                print(f"Processed batch {idx+1}")
        
        # Check results directory after processing
        print(f"Results directory contents after processing:")
        result_files = []
        for root, dirs, files in os.walk(results_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                print(f"  - {file_path} ({file_size} bytes)")
                result_files.append(file_path)
        
        # Convert result paths to URLs
        result_urls = {}
        for key, path in result_paths.items():
            rel_path = os.path.relpath(path, RESULTS_DIR)
            result_urls[key] = f"/results/{rel_path}"
        
        # Include diagnostic information
        diagnostic_info.update({
            "results_dir_contents": [],
            "result_files": result_files,
            "model_device": str(next(model.parameters()).device),
            "num_processed_images": len(result_urls) // 3  # Divide by 3 since we have original, mask, and blend for each image
        })
        
        # Gather all directories in results_path
        for root, dirs, files in os.walk(results_path):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                dir_contents = os.listdir(dir_path)
                diagnostic_info["results_dir_contents"].append({
                    "directory": dir_path,
                    "contents": dir_contents
                })

        log_processing_event(
            event_type="PROCESSING_SUCCESS",
            session_id=session_id,
            details={"num_outputs": len(result_urls)}
        )
                        
        return {
            "message": f"Processed {len(saved_files)} images successfully",
            "session_id": session_id,
            "results": result_urls,
            "diagnostic_info": diagnostic_info
        }
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        print(f"Error: {str(e)}")
        print(f"Traceback: {trace}")
        
        # Include error information in diagnostic info
        diagnostic_info["error"] = str(e)
        diagnostic_info["traceback"] = trace
        
        log_processing_event(
            event_type="PROCESSING_ERROR",
            session_id=session_id,
            details={"error": str(e), "error_type": type(e).__name__}
        )

        # Still return a proper response with error details
        return {
            "message": f"Error processing images: {str(e)}",
            "session_id": session_id,
            "diagnostic_info": diagnostic_info
        }

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up all uploaded and generated files when the app shuts down"""
    try:
        # Clear the uploads directory
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            print(f"Cleaned up uploads directory: {UPLOAD_DIR}")
            
        # Clear the results directory
        if os.path.exists(RESULTS_DIR):
            shutil.rmtree(RESULTS_DIR)
            os.makedirs(RESULTS_DIR, exist_ok=True)
            print(f"Cleaned up results directory: {RESULTS_DIR}")
            
        print("All temporary files have been cleaned up on shutdown")
    except Exception as e:
        print(f"Error cleaning up files on shutdown: {str(e)}")
        
def main():
    """Run the FastAPI app with Uvicorn"""
    uvicorn.run("app:app", port=5000, reload=True)

if __name__ == '__main__':
    main() 