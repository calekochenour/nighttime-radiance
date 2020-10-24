"""
-------------------------------------------------------------------------------
 Preprocesses NASA VNP46A1 HDF5 files. This script takes raw .h5 files and
 completes the following preprocessing tasks:

   - Extracts radiance and qualify flag bands;
   - Masks radiance for fill values, clouds, and sensor problems;
   - Fills masked data with NaN values;
   - Creates a georeferencing transform;
   - Creates export metadata; and,
   - Exports radiance data to GeoTiff format.

 The script assumes a folder structure as follows:

   - nighttime-radiance/
     - 01-code-scripts/
       - preprocess_vnp46a1.py
     - 02-raw-data/
     - 03-processed-data/
     - 04-graphics-outputs/
     - 05-papers-writings/

 Running the script from the '01-code-scripts/' folder works by default. If the
 script runs from a different folder, the paths in the environment setup
 section may have to be changed.
-------------------------------------------------------------------------------
Author: Cale Kochenour
Contact: cale.kochenour@alumni.psu.edu
Updated: 10/24/2020
-------------------------------------------------------------------------------
"""
# -------------------------ENVIRONMENT SETUP--------------------------------- #
import os
import glob
import viirs

# Set working directory to main 'nighttime-radiance/' folder
os.chdir("..")
print(f"Working directory: {os.getcwd()}")

# -------------------------USER-DEFINED VARIABLES---------------------------- #
# Define path folder containing input VNP46A1 HDF5 files
hdf5_input_folder = os.path.join("02-raw-data", "hdf", "south-korea")

# Defne path to output folder to store exported GeoTiff files
geotiff_output_folder = os.path.join(
    "03-processed-data", "raster", "south-korea", "vnp46a1-grid"
)

# -------------------------DATA PREPROCESSING-------------------------------- #
# Preprocess each HDF5 file (extract bands, mask for fill values, clouds, and
#  sensor problems, fill masked values with NaN, export to GeoTiff)
hdf5_files = glob.glob(os.path.join(hdf5_input_folder, "*.h5"))
processed_files = 0
total_files = len(hdf5_files)
for hdf5 in hdf5_files:
    viirs.preprocess_vnp46a1(
        hdf5_path=hdf5, output_folder=geotiff_output_folder
    )
    processed_files += 1
    print(f"Preprocessed file: {processed_files} of {total_files}\n\n")

# -------------------------MAIN FUNCTION------------------------------------- #
# # Export paths
# def main():
#     """Exports projects paths.
#     """
#     export_layer_paths(
#         project_path=input_project_file_path, output_path=output_text_file_path
#     )


# -------------------------RUN SCRIPT---------------------------------------- #
# if __name__ == "__main__":
#     main()

# -------------------------SCRIPT COMPLETION--------------------------------- #
print("\n")
print("-" * (18 + len(os.path.basename(__file__))))
print(f"Completed script: {os.path.basename(__file__)}")
print("-" * (18 + len(os.path.basename(__file__))))
