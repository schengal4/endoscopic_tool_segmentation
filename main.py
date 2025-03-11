import os
import monai
import torch
from monai.apps import download_and_extract
from monai.bundle import ConfigParser
from monai.data import decollate_batch
from monai.handlers import MLFlowHandler
from monai.config import print_config
from monai.visualize.utils import blend_images
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import numpy as np
import shutil
import fastapi
# Parse the variables in the config file.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_config_file = os.path.join("./configs", "inference.json")

model_config = ConfigParser()
model_config.read_config(model_config_file)
model_config["bundle_root"] = "."
output_dir = os.path.abspath("./monai_results")
model_config["output_dir"] = output_dir
model_config["dataset_dir"] = "./real"

# Identify which version of the model we want to load.
checkpoint = os.path.join("./models", "model.pt")
# Generate functions for preprocessing, inference, etc.
preprocessing = model_config.get_parsed_content("preprocessing")
model = model_config.get_parsed_content("network").to(device)
inferer = model_config.get_parsed_content("inferer")

# Modify the postprocessing to SKIP the SaveImaged transform
original_postprocessing = model_config.get_parsed_content("postprocessing")
new_transforms = []
for transform in original_postprocessing.transforms:
    # Skip the SaveImaged transform as we'll handle it manually
    if not (hasattr(transform, '__class__') and transform.__class__.__name__ == 'SaveImaged'):
        new_transforms.append(transform)
original_postprocessing.transforms = new_transforms

dataloader = model_config.get_parsed_content("dataloader")


def get_results(directory_file_path: str, output_directory_file_path: str):
    """
    Process images and save both original images and segmentation masks.
    
    Args:
        directory_file_path: Path to input images
        output_directory_file_path: Path to save results
    """
    model_config["dataset_dir"] = directory_file_path
    model_config["output_dir"] = output_directory_file_path
    
    # Ensure output directory exists
    os.makedirs(output_directory_file_path, exist_ok=True)
    
    # Load the model weights
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    
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
                # 1. First rotate 90 degrees clockwise to counter the counterclockwise rotation
                # 2. Then mirror horizontally to correct the left-right reversal
                orig_img_pil = orig_img_pil.transpose(Image.ROTATE_270)
                orig_img_pil = ImageOps.mirror(orig_img_pil)
                mask_pil = Image.fromarray(mask_vis).transpose(Image.ROTATE_270)
                mask_pil = ImageOps.mirror(mask_pil)
                
                # Convert back to numpy for blending
                orig_img_resized = np.array(orig_img_pil)
                mask_vis_rotated = np.array(mask_pil)
                
                print(f"Resized, rotated, and mirrored original image shape: {orig_img_resized.shape}")
                
                # Save original image with correct orientation
                orig_img_path = os.path.join(img_output_dir, f"{filename_no_ext}.png")
                orig_img_pil.save(orig_img_path)
                print(f"Saved original image: {orig_img_path}")
                
                # Save the mask with correct orientation
                mask_img_path = os.path.join(img_output_dir, f"{filename_no_ext}_trans.png")
                mask_pil.save(mask_img_path)
                print(f"Saved mask: {mask_img_path}")
                
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
                except Exception as e:
                    print(f"Error creating blended image: {e}")
                    print(f"orig_img_resized shape: {orig_img_resized.shape}, mask_rgb shape: {mask_rgb.shape}")
            
            print(f"Processed batch {idx+1}")
        
        print("Inference complete.")
        
@app.get("/example")
def get_example():
    if os.path.exists(os.path.abspath("./monai_results")):
        shutil.rmtree(os.path.abspath("./monai_results"))
    get_results("./real", os.path.abspath("./monai_results"))
    
   
if __name__ == '__main__':
    main()