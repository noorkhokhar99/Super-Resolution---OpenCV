#!pip install git+https://github.com/huggingface/diffusers.git

from PIL import Image
from diffusers import LDMSuperResolutionPipeline
import torch

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model_id = "CompVis/ldm-super-resolution-4x-openimages"
pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
pipeline = pipeline.to(device)

# Load local image (provide the path to your image)
image_path = "demo.png"  # Change this to your image file
low_res_img = Image.open(image_path).convert("RGB")
low_res_img = low_res_img.resize((128, 128))  # Resize for compatibility

# Run pipeline for super-resolution
upscaled_image = pipeline(low_res_img, num_inference_steps=100, eta=1).images[0]

# Save output image
output_path = "ldm_generated_image.png"
upscaled_image.save(output_path)

print(f"Upscaled image saved as {output_path}")
