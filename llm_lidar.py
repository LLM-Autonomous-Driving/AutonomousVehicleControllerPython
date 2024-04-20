from ollama import chat

import glob
import pandas as pd

import os
import time

llava_13b = 'llava:13b-v1.6'
llava_7b = 'llava:7b'
llava_34b = 'llava:34b-v1.6-q2_K'
llava_mistral = 'mistral'
llava_bakllava = 'bakllava'
wizard_math = 'wizard-math'
gemma_7b = 'gemma:7b'
mixtral = 'mixtral'
wizard_vicuna = 'wizard-vicuna-uncensored:13b-q6_K'
deep_seek_coder = 'deepseek-coder:6.7b'

data_file = 'data/data_without_speed.txt'
directory_for_data = 'LLM_generated_csv/'

default_model = llava_13b
default_prompt = "Change this prompt to suit your needs."

# read text file instructions/instructions.md
try:
    with open('instructions/instructions_2.md', 'r') as file:
        default_prompt = file.read()
except FileNotFoundError:
    # if the file is not found, throw an error
    raise FileNotFoundError(
        "instructions_2.md file not found. Please make sure the file exists in the instructions folder.")


lidar_data_column = 'lidar_data'
steer_suggestion_column = 'steer_suggestion'
actual_steer_column = 'actual_steer'
difference_column = 'difference'
time_column = 'time_taken'


# Load the DataFrame from a CSV file, or create a new one if the file doesn't exist
def load_or_create_dataframe(description_file):
    if os.path.isfile(description_file):
        df = pd.read_csv(description_file)
    else:
        df = pd.DataFrame(columns=[lidar_data_column,
                                   steer_suggestion_column,
                                   actual_steer_column,
                                   # difference_column,
                                   time_column])
    return df


def get_file_lines(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()


# Function to extract steer value from a line and return the rest of the values
def extract_steer( line):
    values = line.strip().split(', ')
    steer = None
    remaining_values = []
    for value in values:
        key, val = value.split(': ')
        if key == 'steer':
            steer = float(val)
        else:
            remaining_values.append(value)
    return ', '.join(remaining_values), steer

class SteerSuggestion:
    def __init__(self, description_file, model=default_model, prompt=default_prompt):
        self.description = description_file
        self.model = model
        self.prompt = prompt
        self.df = load_or_create_dataframe(description_file)
        self.messages = []
        self.messages.append({'role': 'system', 'content': self.prompt})

    # processing the images
    def process_data(self, lidar_data_line):
        full_response = ''
        lidar_data, actual_steer = extract_steer(lidar_data_line)

        self.messages.append({
            'role': 'user',
            'content': lidar_data
        })

        new_message = [
            {
                'role': 'user',
                'content': default_prompt + lidar_data
            }
        ]

        start = time.time()
        # Generate a description of the image
        for response in chat(model=self.model, messages=new_message, stream=True):
            # Print the response to the console and add it to the full response
            # print(response, end='', flush=True) # For debugging purposes
            full_response += response['message']['content']
        end = time.time()
        time_taken = end - start
        # Add a new row to the DataFrame
        self.df.loc[len(self.df)] = [lidar_data, full_response, actual_steer, time_taken]

    def bulk_process_data(self, file, limit=5):
        lines = get_file_lines(file)

        counter = 0
        print(f"Processing {limit} out of {len(lines)} lines")
        start = time.time()
        for line in lines:
            if counter >= limit:
                break
            if line not in self.df[lidar_data_column].values:
                self.process_data(line)
                print(f"Processed {counter + 1}/{limit}")
                counter += 1
        end = time.time()
        print(f"Processed {counter} images in {end - start:.2f} seconds using {self.model} model.")

        # Save the DataFrame to a CSV file
        self.df.to_csv(self.description, index=False)


# For testing purposes
if __name__ == '__main__':
    steer_suggestion_llava7b = SteerSuggestion(directory_for_data +llava_7b + 'steer_suggestion2.1.csv', model=llava_7b, prompt=default_prompt)
    steer_suggestion_llava13b = SteerSuggestion(directory_for_data +llava_13b + 'steer_suggestion2.1.csv', model=llava_13b,
                                                prompt=default_prompt)
    steer_suggestion_bakllava = SteerSuggestion(directory_for_data +llava_bakllava + 'steer_suggestion2.1.csv', model=llava_bakllava,
                                                prompt=default_prompt)
    steer_suggestion_mistral = SteerSuggestion(directory_for_data +llava_mistral + 'steer_suggestion2.1.csv', model=llava_mistral,
                                               prompt=default_prompt)
    steer_suggestion_wizard_math = SteerSuggestion(directory_for_data +wizard_math + 'steer_suggestion2.1.csv', model=wizard_math,
                                                  prompt=default_prompt)
    steer_suggestion_gemma_7b = SteerSuggestion(directory_for_data +gemma_7b + 'steer_suggestion2.1.csv', model=gemma_7b,
                                                prompt=default_prompt)
    steer_suggestion_mixtral = SteerSuggestion(directory_for_data +mixtral + 'steer_suggestion2.1.csv', model=mixtral,
                                                  prompt=default_prompt)
    steer_suggestion_wizard_vicuna = SteerSuggestion(directory_for_data +wizard_vicuna + 'steer_suggestion2.1.csv', model=wizard_vicuna,
                                                    prompt=default_prompt)
    steer_suggestion_deep_seek_coder = SteerSuggestion(directory_for_data +deep_seek_coder + 'steer_suggestion2.1.csv', model=deep_seek_coder,
                                                        prompt=default_prompt)


    steer_suggestion_llava7b.bulk_process_data(data_file, limit=20)
    steer_suggestion_llava13b.bulk_process_data(data_file, limit=20)
    steer_suggestion_bakllava.bulk_process_data(data_file, limit=20)
    steer_suggestion_mistral.bulk_process_data(data_file, limit=20)
    steer_suggestion_wizard_math.bulk_process_data(data_file, limit=20)
    steer_suggestion_gemma_7b.bulk_process_data(data_file, limit=20)
    steer_suggestion_wizard_vicuna.bulk_process_data(data_file, limit=20)
    steer_suggestion_deep_seek_coder.bulk_process_data(data_file, limit=20)
    steer_suggestion_mixtral.bulk_process_data(data_file, limit=20)
