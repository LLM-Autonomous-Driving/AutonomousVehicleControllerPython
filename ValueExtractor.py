from ollama import chat

import glob
import pandas as pd

import matplotlib.pyplot as plt

import os
import time

gemma_7b = 'gemma:7b'
mistral = 'mistral'
llava_7b = 'llava:7b'

directory_for_data = 'LLM_generated_csv/'

lidar_data_column = 'lidar_data'
steer_suggestion_column = 'steer_suggestion'
actual_steer_column = 'actual_steer'
time_column = 'time_taken'

prompt = ("Extract the number value from the following prompt, if the value is not present, please output 99999. "
          "Make sure to only output a number and nothing else:  ")


# Load the DataFrame from a CSV file
def load_dataframe(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        # if the file is not found, throw an error
        raise FileNotFoundError(
            file_path + " file not found. Please make sure the file exists.")
    return df


def extract_steer(steer_suggestion):
    import re
    # Using regular expression to find a number in the string
    match = re.search(r'\d+', str(steer_suggestion))
    if match:
        return int(match.group())  # Extracted number
    else:
        return 99999  # Return 99999 if no number is found






# Find the difference between the actual steer and the suggested steer
def find_difference(row):
    return extract_steer(row[steer_suggestion_column]) - row[actual_steer_column]


# Calculate average time taken to process the data
def average_time_taken(dataframe):
    return dataframe[time_column].mean()


# Calculate standard deviation of time taken to process the data
def standard_deviation_time_taken(dataframe):
    return dataframe[time_column].std()


# Calculate variance of time taken to process the data
def variance_time_taken(dataframe):
    return dataframe[time_column].var()


# Calculate mean difference for dataframe then extract outliers and return both the mean and outliers
def mean_difference(dataframe):
    return dataframe.apply(find_difference, axis=1).mean()




# Calculate standard deviation of difference for dataframe then filter outliers and return both the SD and outliers
def standard_deviation_difference(dataframe):
    # Calculate the standard deviation of the difference
    data = dataframe.apply(find_difference, axis=1)
    return data.std()



# Calculate variance of difference for dataframe then extract outliers and return both the variance and outliers
def variance_difference(dataframe):
    return dataframe.apply(find_difference, axis=1).var()


# Output the information calculated for each dataframe
def output_information(dataframe, name):
    mean_difference_data = mean_difference(dataframe)
    standard_deviation_difference_data = standard_deviation_difference(dataframe)
    variance_difference_data = variance_difference(dataframe)
    average_time_taken_data = average_time_taken(dataframe)
    standard_deviation_time_taken_data = standard_deviation_time_taken(dataframe)
    variance_time_taken_data = variance_time_taken(dataframe)

def mean_difference_graphs(dataframe1, name1, dataframe2, name2, dataframe3, name3):
    # Calculate the mean difference for each dataframe
    mean_difference_data1 = mean_difference(dataframe1)
    mean_difference_data2 = mean_difference(dataframe2)
    mean_difference_data3 = mean_difference(dataframe3)

    # Create a bar chart to compare the mean difference for each dataframe
    plt.bar([name1, name2, name3], [mean_difference_data1, mean_difference_data2, mean_difference_data3])
    plt.xlabel('Dataframe')
    plt.ylabel('Mean Difference')
    plt.title('Mean Difference for each Dataframe')
    plt.show()

def standard_deviation_difference_graphs(dataframe1, name1, dataframe2, name2, dataframe3, name3):
    # Calculate the standard deviation of the difference for each dataframe
    standard_deviation_difference_data1 = standard_deviation_difference(dataframe1)
    standard_deviation_difference_data2 = standard_deviation_difference(dataframe2)
    standard_deviation_difference_data3 = standard_deviation_difference(dataframe3)

    # Create a bar chart to compare the standard deviation of the difference for each dataframe
    plt.bar([name1, name2, name3], [standard_deviation_difference_data1, standard_deviation_difference_data2, standard_deviation_difference_data3])
    plt.xlabel('Dataframe')
    plt.ylabel('Standard Deviation of Difference')
    plt.title('Standard Deviation of Difference for each Dataframe')
    plt.show()

def variance_difference_graphs(dataframe1, name1, dataframe2, name2, dataframe3, name3):
    # Calculate the variance of the difference for each dataframe
    variance_difference_data1 = variance_difference(dataframe1)
    variance_difference_data2 = variance_difference(dataframe2)
    variance_difference_data3 = variance_difference(dataframe3)

    # Create a bar chart to compare the variance of the difference for each dataframe
    plt.bar([name1, name2, name3], [variance_difference_data1, variance_difference_data2, variance_difference_data3])
    plt.xlabel('Dataframe')
    plt.ylabel('Variance of Difference')
    plt.title('Variance of Difference for each Dataframe')
    plt.show()

def average_time_taken_graphs(dataframe1, name1, dataframe2, name2, dataframe3, name3):
    # Calculate the average time taken for each dataframe
    average_time_taken_data1 = average_time_taken(dataframe1)
    average_time_taken_data2 = average_time_taken(dataframe2)
    average_time_taken_data3 = average_time_taken(dataframe3)

    # Create a bar chart to compare the average time taken for each dataframe
    plt.bar([name1, name2, name3], [average_time_taken_data1, average_time_taken_data2, average_time_taken_data3])
    plt.xlabel('Dataframe')
    plt.ylabel('Average Time Taken')
    plt.title('Average Time Taken for each Dataframe')
    plt.show()

def standard_deviation_time_taken_graphs(dataframe1, name1, dataframe2, name2, dataframe3, name3):
    # Calculate the standard deviation of the time taken for each dataframe
    standard_deviation_time_taken_data1 = standard_deviation_time_taken(dataframe1)
    standard_deviation_time_taken_data2 = standard_deviation_time_taken(dataframe2)
    standard_deviation_time_taken_data3 = standard_deviation_time_taken(dataframe3)

    # Create a bar chart to compare the standard deviation of the time taken for each dataframe
    plt.bar([name1, name2, name3], [standard_deviation_time_taken_data1, standard_deviation_time_taken_data2, standard_deviation_time_taken_data3])
    plt.xlabel('Dataframe')
    plt.ylabel('Standard Deviation of Time Taken')
    plt.title('Standard Deviation of Time Taken for each Dataframe')
    plt.show()

def variance_time_taken_graphs(dataframe1, name1, dataframe2, name2, dataframe3, name3):
    # Calculate the variance of the time taken for each dataframe
    variance_time_taken_data1 = variance_time_taken(dataframe1)
    variance_time_taken_data2 = variance_time_taken(dataframe2)
    variance_time_taken_data3 = variance_time_taken(dataframe3)

    # Create a bar chart to compare the variance of the time taken for each dataframe
    plt.bar([name1, name2, name3], [variance_time_taken_data1, variance_time_taken_data2, variance_time_taken_data3])
    plt.xlabel('Dataframe')
    plt.ylabel('Variance of Time Taken')
    plt.title('Variance of Time Taken for each Dataframe')
    plt.show()


if __name__ == '__main__':
    # Load the dataframes from the CSV files
    gemma_dataframe = load_dataframe(directory_for_data + 'gemma:7bsteer_suggestion2.1.csv')
    mistral_dataframe = load_dataframe(directory_for_data + 'mistralsteer_suggestion2.1.csv')
    Llama7b_dataframe = load_dataframe(directory_for_data + 'llava:7bsteer_suggestion2.1.csv')

    # Output the information for each dataframe
    mean_difference_graphs(gemma_dataframe, gemma_7b, mistral_dataframe, mistral, Llama7b_dataframe, llava_7b)
    standard_deviation_difference_graphs(gemma_dataframe, gemma_7b, mistral_dataframe, mistral, Llama7b_dataframe, llava_7b)
    variance_difference_graphs(gemma_dataframe, gemma_7b, mistral_dataframe, mistral, Llama7b_dataframe, llava_7b)
    average_time_taken_graphs(gemma_dataframe, gemma_7b, mistral_dataframe, mistral, Llama7b_dataframe, llava_7b)
    standard_deviation_time_taken_graphs(gemma_dataframe, gemma_7b, mistral_dataframe, mistral, Llama7b_dataframe, llava_7b)
    variance_time_taken_graphs(gemma_dataframe, gemma_7b, mistral_dataframe, mistral, Llama7b_dataframe, llava_7b)


