import cv2
from cv2 import dnn_superres
import os

# Function to check CUDA availability
def is_cuda_available():
    build_info = cv2.getBuildInformation()
    return "CUDA: YES" in build_info and "CUDNN: YES" in build_info

# Check CUDA status
cuda_enabled = is_cuda_available()
print(f"CUDA Enabled: {cuda_enabled}")

# Select backend (GPU if available, otherwise CPU)
backend = cv2.dnn.DNN_BACKEND_CUDA if cuda_enabled else cv2.dnn.DNN_BACKEND_DEFAULT
target = cv2.dnn.DNN_TARGET_CUDA if cuda_enabled else cv2.dnn.DNN_TARGET_CPU

# Load image
image_path = 'test.png'  # Change to your image
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file '{image_path}' not found!")

image = cv2.imread(image_path)

# Ensure image is loaded
if image is None:
    raise ValueError("Failed to load image. Check the file path and format.")

# Function to upscale an image using Super-Resolution
def upscale_image(image, model_path, model_name, scale_factor):
    sr = dnn_superres.DnnSuperResImpl_create()

    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found!")

    # Load and configure model
    sr.readModel(model_path)
    sr.setModel(model_name, scale_factor)
    sr.setPreferableBackend(backend)
    sr.setPreferableTarget(target)

    # Upscale the image
    upscaled = sr.upsample(image)
    return upscaled

# First pass: Upscale with EDSR (x4)
upscaled_edsr = upscale_image(image, 'EDSR_x4.pb', 'edsr', 4)

# Second pass: Further upscale with LapSRN (x8)
upscaled_lapsrn = upscale_image(upscaled_edsr, 'LapSRN_x8.pb', 'lapsrn', 8)

# Save only the final best quality image
output_path = "super_resolution_best.png"
cv2.imwrite(output_path, upscaled_lapsrn)

print(f"✅ High-Quality Super-Resolution image saved: {output_path}")
