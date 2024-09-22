#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <path> <layer_name>"
    exit 1
fi

# Assign the arguments to variables
path=$1
layer_name=$2

# Count the number of .pt files in the given path
pt_files=("$path"/checkpoints/*.pt)
pt_file_count=${#pt_files[@]}

# Check if there are any .pt files in the directory
if [ "$pt_file_count" -eq 0 ]; then
    echo "No .pt files found in the directory: $path"
    exit 1
fi

# Loop through each .pt file and run the python script
echo "Found $pt_file_count checkpoints"
for ((i=1; i<pt_file_count+1; i++)); do
    python extract_activations.py "$path" "$i" "$layer_name"
done

echo "All extractions completed."
