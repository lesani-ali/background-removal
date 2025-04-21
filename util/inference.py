from tqdm import tqdm
from PIL import Image
import numpy as np
import os
import logging
from pipeline.pipeline import BackgroundRemovalPipeline


def load_image(image_path: str) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    return image


def save_image(image: np.ndarray, output_path: str) -> None:
    output_image = Image.fromarray(image)
    output_image.save(output_path)


def run_inference(
    model: BackgroundRemovalPipeline,
    input_img_dir: str,
    output_img_dir: str,
) -> None:

    input_imgs = os.listdir(input_img_dir)
    total_images = len(input_imgs)

    for idx, input_img in enumerate(tqdm(input_imgs, desc="Processing images", ncols=100)):
        if not input_img.lower().endswith(('.jpg', '.png')): continue

        logging.info(
            f'\nProcessing image {idx + 1}/{total_images}: {input_img}'
        )
        # Read input image
        base_name = input_img.split('.')[0]
        img_path = os.path.join(input_img_dir, input_img)
        in_img = load_image(img_path)

        # Get no background image
        out_img = model(in_img)

        # Save output image
        save_image(out_img, os.path.join(output_img_dir, base_name + '.png'))

        logging.info(
            f'Saved image: {os.path.join(output_img_dir, base_name + ".png")}'
        )
