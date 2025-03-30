#!/bin/bash

# Input features path
FEATURE_FOLDER="/home/NETID/gayat23/cuda_assignment/features_output"

# Output folder for all rotation vectors
OUTPUT_FOLDER="/home/NETID/gayat23/cuda_assignment/sc_depth_pl/rotation_results"

mkdir -p $OUTPUT_FOLDER

# Get all feature CSVs sorted
FILES=($(ls $FEATURE_FOLDER/feature_*.csv | sort -V))

# Loop over consecutive pairs
for ((i=0; i<${#FILES[@]}-1; i++))
do
    img1=${FILES[$i]}
    img2=${FILES[$i+1]}
    
    echo "Processing: $img1 and $img2"
    
    # Run RectifyNet decoder
    ./rectify_net "$img1" "$img2"
    
    # Save output rotation vector
    cp /home/NETID/gayat23/cuda_assignment/sc_depth_pl/rotation_output.csv $OUTPUT_FOLDER/rotation_${i}.csv
done

echo "All rotation vectors saved in $OUTPUT_FOLDER/"
