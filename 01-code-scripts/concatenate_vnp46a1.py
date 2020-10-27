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

 Running the script from the '01-code-scripts/' folder works by default. If the
 script runs from a different folder, the paths in the environment setup
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

# Set working directory
os.chdir("..")

# -------------------------USER-DEFINED VARIABLES---------------------------- #
# Define path to folder containing preprocessed VNP46A1 GeoTiff files
geotiff_input_folder = os.path.join(
    "03-processed-data", "raster", "south-korea", "vnp46a1-grid"
)

# Defne path to output folder to store concatenated, exported GeoTiff files
geotiff_output_folder = os.path.join(
    "03-processed-data", "raster", "south-korea", "vnp46a1-grid-concatenated"
)

# -------------------------DATA PREPROCESSING-------------------------------- #
# Concatenate and export adjacent images that have the same acquisition date
dates = viirs.create_date_range(start_date="2020-01-01", end_date="2020-01-05")
for date in dates:
    adjacent_images = []
    for file in glob.glob(os.path.join(geotiff_input_folder, "*.tif")):
        if date in viirs.extract_date_vnp46a1(geotiff_path=file):
            adjacent_images.append(file)
    adjacent_images_sorted = sorted(adjacent_images)
    if len(adjacent_images_sorted) == 2:
        viirs.concatenate_preprocessed_vnp46a1(
            west_geotiff_path=adjacent_images_sorted[0],
            east_geotiff_path=adjacent_images_sorted[1],
            output_folder=geotiff_output_folder,
        )

# -------------------------SCRIPT COMPLETION--------------------------------- #
print("\n")
print("-" * (18 + len(os.path.basename(__file__))))
print(f"Completed script: {os.path.basename(__file__)}")
print("-" * (18 + len(os.path.basename(__file__))))
