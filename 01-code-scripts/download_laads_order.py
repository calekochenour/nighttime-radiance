"""
-------------------------------------------------------------------------------
 Downloads a LAADS Web Order. The script will download files to the directory
 specified in the 'download_directory' variable. Any folders not existing in
 the specified path will be created during the download.

 Level-1 and Atmosphere Archive & Distribution System (LAADS) home:

   - https://ladsweb.modaps.eosdis.nasa.gov/about/purpose/

 Files can be searched for and data orders can be placed here:

   - https://ladsweb.modaps.eosdis.nasa.gov/search/

 User accounts (needed to obtain a token) can be created here:

   - https://urs.earthdata.nasa.gov/

 Download parameters will accompany the LAADS order completion email:

   -e robots=off : Bypass the robots.txt file, to allow access to all files in
                 the order

   -m            : Enable mirroring options (-r -N -l inf) for recursive
                 download, timestamping & unlimited depth

   -np           : Do not recurse into the parent location

   -R .html,.tmp : Reject (do not save) any .html or .tmp files (which are
                 extraneous to the order)

   -nH           : Do not create a subdirectory with the Host name
                 (ladsweb.modaps.eosdis.nasa.gov)

   --cut-dirs=3  : Do not create subdirectories for the first 3 levels
                 (archive/orders/{ORDER_ID})

   --header      : Adds the header with your appKey (which is encrypted via
                   SSL)

   -P            : Specify the directory prefix (may be relative or absolute)

 This script uses the following folder structure:

   - nighttime-radiance/
     - 01-code-scripts/
       - preprocess_vnp46a1.py
     - 02-raw-data/
     - 03-processed-data/
     - 04-graphics-outputs/
     - 05-papers-writings/
-------------------------------------------------------------------------------
"""
# -------------------------ENVIRONMENT SETUP--------------------------------- #
# Import packages
import os

# -------------------------USER-DEFINED VARIABLES---------------------------- #
# Set LAADS token (specific to user account)
token = os.environ.get("LAADS_TOKEN")

# Set path to file containing order ID
order_id_file_path = os.path.join("05-papers-writings", "laads-order.txt")

# Set location for data downloaded (for test and real data)
test_directory = "05-papers-writings"
data_directory = os.path.join("02-raw-data", "raster", "south-korea")

# Test the script by downloading LAADS README
#  Set True for README download, False for LAADS data order download
test_download = True

# -------------------------DATA ACQUISITION---------------------------------- #
# Get order ID from file
with open(order_id_file_path, mode="r") as file:
    order_id = int(file.readline())

# Set wget download string
download_str = (
    (
        "wget -e robots=off -m -np -R .html,.tmp -nH --cut-dirs=3 "
        '"https://ladsweb.modaps.eosdis.nasa.gov/archive/README" '
        f'--header "Authorization: Bearer {token}" -P {test_directory}'
    )
    if test_download
    else (
        "wget -e robots=off -m -np -R .html,.tmp -nH --cut-dirs=3 "
        f'"https://ladsweb.modaps.eosdis.nasa.gov/archive/orders/{order_id}/" '
        f'--header "Authorization: Bearer {token}" -P {data_directory}'
    )
)

# Download data
os.system(download_str)

# -------------------------SCRIPT COMPLETION--------------------------------- #
print("\n")
print("-" * (18 + len(os.path.basename(__file__))))
print(f"Completed script: {os.path.basename(__file__)}")
print("-" * (18 + len(os.path.basename(__file__))))
