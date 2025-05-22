import pandas as pd
import numpy as np
import os

def process_csv(input_filename):
    # Generate output filename
    base_name = os.path.splitext(input_filename)[0]
    output_filename = f"{base_name}_forR.csv"

    # Load the CSV file
    df = pd.read_csv(input_filename)

    # Count the number of points for each track
    track_counts = df['track_number'].value_counts()

    # Filter out tracks with less than 3 points
    df = df[df['track_number'].isin(track_counts[track_counts >= 3].index)]

    # Create a mapping of old track numbers to new sequential numbers starting from 1
    unique_tracks = df['track_number'].unique()
    track_mapping = {old: new for new, old in enumerate(unique_tracks, start=1)}

    # Apply the new track numbers
    df['track_number'] = df['track_number'].map(track_mapping)

    # Increase frame numbers by 1
    df['frame'] += 1

    # Save the processed data
    df.to_csv(output_filename, index=False)

    print(f"Processed {input_filename} -> {output_filename}")

def batch_process_folder(folder_path):
    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            full_path = os.path.join(folder_path, filename)
            process_csv(full_path)

# Specify the folder containing your CSV files
folder_path = '/Users/george/Desktop/tdt_analysis/for_Alan/differntRecordingLengths/2s'

# Run the batch processing
batch_process_folder(folder_path)

print("Batch processing completed.")
