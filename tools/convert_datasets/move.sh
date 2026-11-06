#!/bin/bash

# Set paths
SOURCE_DIR="../../../dataset/gta/labels"
TARGET_DIR="../../../dataset/gta/labels/test"
# LIST_FILE="gtav_split_val.txt"
# LIST_FILE="gtav_split_train.txt"
LIST_FILE="gtav_split_test.txt"



# Create target directory if it doesn't exist
# mkdir -p "$TARGET_DIR"

# Loop through each line in the txt file
while IFS= read -r filename; do
    # Check if the file exists in the source directory
    if [ -f "$SOURCE_DIR/$filename" ]; then
        mv "$SOURCE_DIR/$filename" "$TARGET_DIR"
        echo "Moved: $filename"
    else
        echo "File not found: $filename"
    fi
done < "$LIST_FILE"