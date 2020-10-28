"""
-------------------------------------------------------------------------------
 Clips already concatenated (and already-preprocessed) VNP46A1 GeoTiff
 files to a specified country bounding box.

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
import geopandas as gpd
import viirs

# Set options
warnings.simplefilter("ignore")

# Set working directory
os.chdir("..")

# -------------------------USER-DEFINED VARIABLES---------------------------- #
# Set path to folder containing concateanted preprocessed VNP46A1 files
geotiff_input_folder = os.path.join(
    "03-processed-data", "raster", "south-korea", "vnp46a1-grid-concatenated"
)

# Set path to output folder to store clipped, exported files
geotiff_output_folder = os.path.join(
    "03-processed-data", "raster", "south-korea", "vnp46a1-clipped"
)

# Set path to shapefile for clipping GeoTiff files
shapefile_path = os.path.join(
    "02-raw-data", "vector", "south-korea", "gadm36_south_korea.shp"
)

# Set country name for clipping (for file export name)
clip_country = "South Korea"

# -------------------------DATA PREPROCESSING-------------------------------- #
# Clip images to bounding box and export clipped images to GeoTiff files
geotiff_files = glob.glob(os.path.join(geotiff_input_folder, "*.tif"))
clipped_files = 0
total_files = len(geotiff_files)
for file in geotiff_files:
    viirs.clip_vnp46a1(
        geotiff_path=file,
        clip_boundary=gpd.read_file(shapefile_path),
        clip_country=clip_country,
        output_folder=geotiff_output_folder,
    )
    clipped_files += 1
    print(f"Clipped file: {clipped_files} of {total_files}\n\n")

# -------------------------SCRIPT COMPLETION--------------------------------- #
print("\n")
print("-" * (18 + len(os.path.basename(__file__))))
print(f"Completed script: {os.path.basename(__file__)}")
print("-" * (18 + len(os.path.basename(__file__))))
