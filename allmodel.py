import os
import torch
import cv2
import numpy as np
import gradio as gr
from PIL import Image
from huggingface_hub import hf_hub_download  # ✅ Fixed: Replaced cached_download
from super_image import ImageLoader, EdsrModel, MsrnModel, MdsrModel, AwsrnModel, A2nModel, CarnModel, PanModel, \
    HanModel, DrlnModel, RcanModel

# Define Input & Output Directories
input_folder = "model"  # Folder containing input images
output_folder = "out"   # Folder where output images will be saved

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Titles and Descriptions
title = "Super-Resolution Image Enhancement"
description = "State-of-the-Art Image Super-Resolution Models."
article = "<p style='text-align: center'><a href='https://github.com/eugenesiow/super-image'>Github Repo</a>" \
          "| <a href='https://eugenesiow.github.io/super-image/'>Documentation</a> " \
          "| <a href='https://github.com/eugenesiow/super-image#scale-x2'>Models</a></p>"

# Available models
models = {
    'EDSR': EdsrModel,
    'MSRN': MsrnModel,
    'MDSR': MdsrModel,
    'AWSRN-BAM': AwsrnModel,
    'A2N': A2nModel,
    'CARN': CarnModel,
    'PAN': PanModel,
    'HAN': HanModel,
    'DRLN': DrlnModel,
    'RCAN': RcanModel
}

# ✅ Fix: Ensure model loading with correct scale
def get_model(model_name, scale):
    """Load the specified super-resolution model."""
    try:
        model_class = models.get(model_name, EdsrModel)  # Default to EDSR
        return model_class.from_pretrained(f'eugenesiow/{model_name.lower()}', scale=scale)
    except Exception as e:
        print(f"❌ Error loading model {model_name}: {e}")
        return None

# ✅ Fix: Added error handling for invalid inputs
def process_image(img, scale, model_name):
    """Perform super-resolution on an image using the selected model."""
    if scale is None or model_name is None:
        return "Error: Please select a valid scale and model."

    try:
        scale = int(scale.replace('x', ''))  # Convert scale to integer
        model = get_model(model_name, scale)

        if model is None:
            return "Error: Failed to load model."

        # Load and process image
        inputs = ImageLoader.load_image(img)
        preds = model(inputs)
        preds = preds.data.cpu().numpy()
        pred = preds[0].transpose((1, 2, 0)) * 255.0

        return Image.fromarray(pred.astype('uint8'), 'RGB')

    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None

# ✅ Fix: Added error handling for batch processing
def process_all_images(scale="x2", model_name="EDSR"):
    """Batch process all images in the input folder and save enhanced images."""
    if not os.path.exists(input_folder):
        print("⚠️ Input folder does not exist. Please create a 'model' folder and add images.")
        return

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(input_folder, filename)
            try:
                img = Image.open(img_path)
                output_img = process_image(img, scale, model_name)

                if output_img:
                    output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_SR.png")
                    output_img.save(output_path)
                    print(f"✅ Processed {filename} → {output_path}")
                else:
                    print(f"❌ Failed to process {filename}")
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

# ✅ Fix: Improved Gradio Interface with Defaults
gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Radio(["x2", "x3", "x4"], label='Scale Factor', value="x2"),
        gr.Dropdown(choices=list(models.keys()), label='Model', value="EDSR")
    ],
    outputs=gr.Image(type="pil", label="Output Image"),
    title=title,
    description=description,
    article=article,
    allow_flagging='never'
).launch(debug=False)

# ✅ Fix: Only run batch processing if images exist
if os.path.exists(input_folder) and len(os.listdir(input_folder)) > 0:
    process_all_images()
else:
    print("⚠️ No images found in the 'model' folder. Skipping batch processing.")
