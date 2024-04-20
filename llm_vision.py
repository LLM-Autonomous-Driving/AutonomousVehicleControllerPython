import ollama
from ollama import generate

import glob
import pandas as pd
from PIL import Image

import os
import time
from io import BytesIO

directory_for_data = 'LLM_generated_csv/'

llava_7b = 'llava:7b'
llava_mistral = 'mistral'
llava_bakllava = 'bakllava'
llava_13b = 'llava:13b-v1.6'
llava_34b = 'llava:34b-v1.6-q2_K'

default_model = llava_13b
default_prompt = ("The image is taken from an autonomous vehicle, state whether the vehicle should begin to turn or "
                  "not. "
                  "The task is for the vehicle to stay centered on the yellow line, but more importantly to avoid "
                  "obstacles. "
                  "Be Clear and concise. Take a moment to breath deeply and you are an expert driver that knows "
                  "exactly what to do. "
                  "Your only responsibility is to make sure the vehicle stays on the yellow line and avoids obstacles. "
                  "Don't worry about anything but the steering. Only focus on the steering. You are an expert at "
                  "steering the vehicle. "
                  "Specify steering direction")

description_column = 'description'
image_column = 'image_file'
time_column = 'time_taken'


# Load the DataFrame from a CSV file, or create a new one if the file doesn't exist
def load_or_create_dataframe(description_file):
    if os.path.isfile(description_file):
        df = pd.read_csv(description_file)
    else:
        df = pd.DataFrame(columns=[image_column, description_column, time_column])
    return df


def get_png_files(folder_path):
    return glob.glob(f"{folder_path}/*.png")


class ImageDescription:
    def __init__(self, description_file, model=default_model, prompt=default_prompt):
        self.description = description_file
        self.model = model
        self.prompt = prompt
        self.df = load_or_create_dataframe(description_file)

    # processing the images
    def process_image(self, image_file):
        # print(f"\nProcessing {image_file}\n")
        with Image.open(image_file) as img:
            with BytesIO() as buffer:
                img.save(buffer, format='PNG')
                image_bytes = buffer.getvalue()

        full_response = ''

        start = time.time()
        # Generate a description of the image
        for response in generate(model=self.model,
                                 prompt=self.prompt,
                                 images=[image_bytes],
                                 stream=True):
            # Print the response to the console and add it to the full response
            # print(response['response'], end='', flush=True)
            full_response += response['response']
        end = time.time()
        time_taken = end - start
        # Add a new row to the DataFrame
        self.df.loc[len(self.df)] = [image_file, full_response, time_taken]

    def bulk_process_images(self, images_folder, limit=5):
        # image_files = get_png_files(images_folder)
        image_files = [
            'images/camera_image_1735.png',
            'images/camera_image_1765.png',
            'images/camera_image_1795.png',
            'images/camera_image_1825.png',
        ]
        counter = 0
        print(f"Processing {limit} out of {len(image_files)} images")
        start = time.time()
        for image_file in image_files:
            if counter >= limit:
                break
            if image_file not in self.df[image_column].values:
                self.process_image(image_file)
                print(f"Processed {counter}: {image_file}")
                counter += 1
        end = time.time()
        print(f"Processed {counter} images in {end - start:.2f} seconds using {self.model} model.")

        # Save the DataFrame to a CSV file
        self.df.to_csv(self.description, index=False)


# For testing purposes
if __name__ == '__main__':
    image_description_7b = ImageDescription(directory_for_data + llava_7b + 'image_descriptions.csv', model=llava_7b, prompt=default_prompt)
    image_description_13b = ImageDescription(directory_for_data + llava_13b + 'image_descriptions.csv', model=llava_13b,
                                             prompt=default_prompt)
    image_description_bakllava = ImageDescription(directory_for_data + llava_bakllava + 'image_descriptions.csv', model=llava_bakllava,
                                                  prompt=default_prompt)
    image_description_mistral = ImageDescription(directory_for_data + llava_mistral + 'image_descriptions.csv', model=llava_mistral,
                                                 prompt=default_prompt)
    # image_description_34b = ImageDescription(directory_for_data + llava_34b+'image_descriptions.csv', model=llava_34b,
    # prompt=default_prompt)

    image_description_7b.bulk_process_images('images', limit=5)
    image_description_13b.bulk_process_images('images', limit=5)
    image_description_bakllava.bulk_process_images('images', limit=5)
    image_description_mistral.bulk_process_images('images', limit=5)
    # image_description_34b.bulk_process_images('images', limit=5)
