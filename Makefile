# Set phony targets
.PHONY: all download preprocess concatenate clip clean

# Download, preprocess, concatenate, and clip
all: download preprocess concatenate clip

# Download a LAADS order
download: 01-code-scripts/download_laads_order.py
	python 01-code-scripts/download_laads_order.py

# Preprocess VNP46A1 HDF5 files
preprocess: 01-code-scripts/preprocess_vnp46a1.py
	python 01-code-scripts/preprocess_vnp46a1.py

# Concatenate adjacent VNP46A1 GeoTiff files
concatenate: 01-code-scripts/concatenate_vnp46a1.py
	python 01-code-scripts/concatenate_vnp46a1.py

# Clip VNP46A1 concatenated GeoTiff files
clip: 01-code-scripts/clip_vnp46a1.py
	python 01-code-scripts/clip_vnp46a1.py

# Delete raw HDF5, preprocessed GeoTiff, and unclipped GeoTiff files
clean:
	rm -f 02-raw-data/hdf/south-korea/*.h5
	rm -f 03-processed-data/raster/south-korea/vnp46a1-grid/*.tif
	rm -f 03-processed-data/raster/south-korea/vnp46a1-grid-concatenated/*.tif
