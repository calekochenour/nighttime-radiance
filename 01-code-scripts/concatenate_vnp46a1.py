"""
-------------------------------------------------------------------------------
 Concatenates already-preprocessed VNP46A1 GeoTiff files that are spatially
 adjacent in the longitudinal direction and exports single GeoTiff files
 containing the concatenated data. Used in cases when a study area bounding
 box intersects two VNP46A1 grid cells.

 This script uses the following folder structure:

   - nighttime-radiance/
     - 01-code-scripts/
       - concatenate_vnp46a1.py
     - 02-raw-data/
     - 03-processed-data/
     - 04-graphics-outputs/
     - 05-papers-writings/

 Running the script from the 'nighttime-radiance/' folder works by default. If
 the script runs from a different folder, the paths in the environment setup
 section may have to be changed.
-------------------------------------------------------------------------------
"""
# -------------------------ENVIRONMENT SETUP--------------------------------- #
# Import packages
import os
import warnings
import glob
import viirs

# Set options
warnings.simplefilter("ignore")

# -------------------------USER-DEFINED VARIABLES---------------------------- #
# Define path to folder containing preprocessed VNP46A1 GeoTiff files
geotiff_input_folder = os.path.join(
    "03-processed-data", "raster", "south-korea", "vnp46a1-grid"
)

# Defne path to output folder to store concatenated, exported GeoTiff files
geotiff_output_folder = os.path.join(
    "03-processed-data", "raster", "south-korea", "vnp46a1-grid-concatenated"
)

# Set start date and end date for processing
start_date, end_date = "2020-04-20", "2020-04-29"

# -------------------------DATA PREPROCESSING-------------------------------- #
# Concatenate and export adjacent images that have the same acquisition date
dates = viirs.create_date_range(start_date=start_date, end_date=end_date)
geotiff_files = glob.glob(os.path.join(geotiff_input_folder, "*.tif"))
concatenated_dates = 0
skipped_dates = 0
processed_dates = 0
total_dates = len(dates)
for date in dates:
    adjacent_images = []
    for file in geotiff_files:
        if date in viirs.extract_date_vnp46a1(geotiff_path=file):
            adjacent_images.append(file)
    adjacent_images_sorted = sorted(adjacent_images)
    if len(adjacent_images_sorted) == 2:
        viirs.concatenate_preprocessed_vnp46a1(
            west_geotiff_path=adjacent_images_sorted[0],
            east_geotiff_path=adjacent_images_sorted[1],
            output_folder=geotiff_output_folder,
        )
        concatenated_dates += 1
    else:
        skipped_dates += 1
    processed_dates += 1
    print(f"Processed dates: {processed_dates} of {total_dates}\n\n")

print(
    f"Concatenated dates: {concatenated_dates}, Skipped dates: {skipped_dates}"
)

# -------------------------SCRIPT COMPLETION--------------------------------- #
print("\n")
print("-" * (18 + len(os.path.basename(__file__))))
print(f"Completed script: {os.path.basename(__file__)}")
print("-" * (18 + len(os.path.basename(__file__))))
