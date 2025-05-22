#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:24:55 2024

@author: george
"""
import pandas as pd
import numpy as np
import os

# Input filename
input_filename = '/Users/george/Desktop/sRg_calc_csv/GB_165_2022_03_01_HTEndothelial_NonBapta_plate1_2_MMStack_Default_bin10_locsID_tracks.csv'

# Generate output filename
base_name = os.path.splitext(input_filename)[0]
output_filename = f"{base_name}_forR.csv"

# Load the CSV file
df = pd.read_csv(input_filename)

# Count the number of points for each track
track_counts = df['track_number'].value_counts()

# Filter out tracks with less than 3 points
df = df[df['track_number'].isin(track_counts[track_counts >= 3].index)]

# Create a mapping of old track numbers to new sequential numbers
unique_tracks = df['track_number'].unique()
track_mapping = {old: new for new, old in enumerate(unique_tracks)}

# Apply the new track numbers
df['track_number'] = df['track_number'].map(track_mapping)

# Increase frame numbers by 1
df['frame'] += 1

# Save the processed data
df.to_csv(output_filename, index=False)

print(f"Processed data saved to {output_filename}")
