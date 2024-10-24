# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 00:38:22 2023

@author: Saagar
 
Creates an Inverse and Augmented training set. Creates a folder with photos and 
a linked JSON file. 
"""

import os
from PIL import Image, ImageOps
import json
import shutil
import base64

# directory of crops and polygons
crops = r''
# directory of augmented crops and polygon files
inverseCrops = r''
d = 640  # crop width/height (always square)


def image_to_base64(img_path):
    """Convert an image to base64."""
    with open(img_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read())
    return encoded.decode('utf-8')


for filename in os.listdir(crops):
    # Check if the file is an image (e.g., JPG or JPEG)
    if filename.lower().endswith(('.jpg', '.jpeg')):
        # Load the original image
        original_image = Image.open(os.path.join(crops, filename))

        # Load the associated JSON file
        json_filename = os.path.splitext(filename)[0] + '.json'
        json_filepath = os.path.join(crops, json_filename)
        if os.path.isfile(json_filepath):
            with open(json_filepath, 'r') as json_file:
                json_data = json.load(json_file)

            # Invert the colors of the image
            inverted_image = ImageOps.invert(original_image)

            # Copy the original image to the inverseCrops directory
            shutil.copy(os.path.join(crops, filename),
                        os.path.join(inverseCrops, filename))

            # Save the inverted image to the inverseCrops directory
            inverted_filename = os.path.splitext(filename)[0] + "_inv.jpg"
            inverted_image.save(os.path.join(
                inverseCrops, inverted_filename), format='JPEG')

            # Copy the JSON file to the inverseCrops directory
            shutil.copy(json_filepath, os.path.join(
                inverseCrops, json_filename))

            # Update JSON data with inverted image information
            json_data["imagePath"] = inverted_filename
            json_data["imageData"] = image_to_base64(
                os.path.join(inverseCrops, inverted_filename))

            # Save the updated JSON file
            inverted_json_filename = os.path.splitext(
                inverted_filename)[0] + ".json"
            with open(os.path.join(inverseCrops, inverted_json_filename), 'w') as inverted_json_file:
                json.dump(json_data, inverted_json_file)

            print(f'{filename} processed and saved in {inverseCrops}')
        else:
            print(f'Skipping {filename} as no associated JSON file found.')
