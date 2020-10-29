# VNP46A1 Data Workflow

This document specifies how to use the code in the `01-code-scripts/` folder to download and preprocess VNP46A1 data.

The project `Makefile` centralizes the execution of code for this repository. Navigate to the root repository to run the `Makefile` commands:

```bash
$ cd ~/nighttime-radiance
```

## Download data

Downloading data requires an order to have been placed at NASA LAADS. Once the order has completed and you have changed the variables (as necessary) in the user-defined variables section of the `01-code-scripts/download_laads_order.py` file, run the following command:

```bash
$ make download
```

## Preprocess Data

Preprocessing data requires VNP46A1 HDF5 (.h5) files to have been downloaded. Once files are downloaded and you have changed the variables (as necessary) in the user-defined variables section of the `01-code-scripts/preprocess_vnp46a1.py` file, run the following command:

```bash
$ make preprocess
```

## Concatenate Data (Optional)

Concatenating data is necessary when the study area crossed into multiple VNP46A1 images. The workflow handles concatenating horizontally-adjacent images at this time. This step requires preprocessed GeoTiff files. Once files are preprocessed into GeoTiffs and you have changed the variables (as necessary) in the user-defined variables section of the `01-code-scripts/concatenate_vnp46a1.py` file, run the following command:

```bash
$ make concatenate
```

## Clip Data

Clipping data requires preprocessed (and optionally concatenated) GeoTiff files. Once files are preprocessed into GeoTiffs and you have changed the variables (as necessary) in the user-defined variables section of the `01-code-scripts/clip_vnp46a1.py` file, run the following command:

```bash
$ make clip
```

## Full Workflow

To run the full workflow in succession (download, preprocess, concatenate, clip), ensure all user-defined variables in all scripts are set correctly and run the following command:

```bash
$ make all
```

Note that this command includes the data concatenation. The `Makefile` contents will have to be changed if the concatenation script is not required for the study area.
