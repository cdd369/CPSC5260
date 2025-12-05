from PIL import Image
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python ppm_to_png.py <input.ppm>")
    sys.exit(1)

input_file = sys.argv[1]

# Create output filename by replacing extension with .png
base_name = os.path.splitext(input_file)[0]
output_file = base_name + ".png"

try:
    # Open PPM file
    im = Image.open(input_file)

    # Save as PNG
    im.save(output_file)
    print(f"Converted {input_file} -> {output_file}")
except Exception as e:
    print(f"Error: {e}")
