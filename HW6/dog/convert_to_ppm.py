#!/usr/bin/env python3

import os
import sys

try:
    from PIL import Image
except ImportError:
    print("Pillow is not installed in this environment.")
    print("Try loading Anaconda: module load anaconda/4.9.2")
    sys.exit(1)

def convert_image(input_path, output_path):
    """Convert a single image to binary PPM (P6)"""
    im = Image.open(input_path).convert("RGB")
    im.save(output_path, format="PPM")
    print(f"Saved: {output_path}")

def batch_convert(folder):
    """Convert all .jpg/.png/.jpeg images in folder to .ppm"""
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(folder, fname)
            output_fname = os.path.splitext(fname)[0] + ".ppm"
            output_path = os.path.join(folder, output_fname)
            convert_image(input_path, output_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 convert_to_ppm.py <image_or_folder>")
        sys.exit(1)

    path = sys.argv[1]

    if os.path.isdir(path):
        batch_convert(path)
    elif os.path.isfile(path):
        output_path = os.path.splitext(path)[0] + ".ppm"
        convert_image(path, output_path)
    else:
        print("Error: path does not exist.")
        sys.exit(1)

