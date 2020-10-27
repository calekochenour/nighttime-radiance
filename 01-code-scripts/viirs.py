""" Module to work with NASA VIIRS DNB data """

import os
import re
import datetime as dt
import numpy as np
import numpy.ma as ma
import pandas as pd
import rasterio as rio
from rasterio.transform import from_origin


def concatenate_preprocessed_vnp46a1(
    west_geotiff_path, east_geotiff_path, output_folder
):
    """Concatenates horizontally-adjacent preprocessed VNP46A1 GeoTiff
    file and exports the concatenated array to a single GeoTiff.

    Paramaters
    ----------
    west_geotiff_path : str
        Path to the West-most GeoTiff.

    east_geotiff_path : str
        Path to the East-most GeoTiff.

    output_folder : str
        Path to the folder where the concatenated file will be
        exported to.

    Returns
    -------
    message : str
        Indication of concatenation completion status (success
        or failure).

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Concatenate adjacent VNP46A1 GeoTiff files
    print(
        (
            f"Started concatenating:\n    "
            f"{os.path.basename(west_geotiff_path)}\n    "
            f"{os.path.basename(east_geotiff_path)}"
        )
    )

    try:
        print("Concatenating West and East arrays...")
        # Concatenate West and East images along the 1-axis
        concatenated = np.concatenate(
            (
                read_geotiff_into_array(geotiff_path=west_geotiff_path),
                read_geotiff_into_array(geotiff_path=east_geotiff_path),
            ),
            axis=1,
        )

        print("Getting bounding box information...")
        # Get bounding box (left, top, bottom) from west image and
        #  (right) from east image
        longitude_min = extract_geotiff_bounding_box(
            geotiff_path=west_geotiff_path
        ).left
        longitude_max = extract_geotiff_bounding_box(
            geotiff_path=east_geotiff_path
        ).right
        latitude_min = extract_geotiff_bounding_box(
            geotiff_path=west_geotiff_path
        ).bottom
        latitude_max = extract_geotiff_bounding_box(
            geotiff_path=west_geotiff_path
        ).top

        print("Creating transform...")
        # Set transform (west bound, north bound, x cell size, y cell size)
        concatenated_transform = from_origin(
            longitude_min,
            latitude_max,
            (longitude_max - longitude_min) / concatenated.shape[1],
            (latitude_max - latitude_min) / concatenated.shape[0],
        )

        print("Creating metadata...")
        # Create metadata for GeoTiff export
        metadata = create_metadata(
            array=concatenated,
            transform=concatenated_transform,
            driver="GTiff",
            nodata=np.nan,
            count=1,
            crs="epsg:4326",
        )

        print("Setting file export name...")
        # Get name for the exported file
        export_name = create_concatenated_export_name(
            west_image_path=west_geotiff_path,
            east_image_path=east_geotiff_path,
        )

        print("Exporting to GeoTiff...")
        # Export concatenated array
        export_array(
            array=concatenated,
            output_path=os.path.join(output_folder, export_name),
            metadata=metadata,
        )
    except Exception as error:
        message = print(f"Concatenating failed: {error}")
    else:
        message = print(
            (
                f"Completed concatenating:\n    "
                f"{os.path.basename(west_geotiff_path)}\n    "
                f"{os.path.basename(east_geotiff_path)}\n\n"
            )
        )

    return message


def create_concatenated_export_name(west_image_path, east_image_path):
    """Creates a file name indicating the concatenation of adjacent two files.

    Paramaters
    ----------
    west_image_path : str
        Path to the West-most image.

    east_image_past : str
        Path to the East-most image.

    Returns
    -------
    export_name : str
        New file name for export, indicating concatenation.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Extract the horizontal grid numbers from the West and East images
    west_image_horizontal_grid_number, east_image_horizontal_grid_number = (
        os.path.basename(west_image_path)[18:20],
        os.path.basename(east_image_path)[18:20],
    )

    # Replace the specific date/time with YYYYJJJ and the single horizontal
    #  grid number with both the West and East numbers
    export_name = (
        os.path.basename(west_image_path)
        .replace(os.path.basename(west_image_path)[35:41], "")
        .replace(
            west_image_horizontal_grid_number,
            west_image_horizontal_grid_number
            + east_image_horizontal_grid_number,
        )
    )

    return export_name


def create_date_range(start_date, end_date):
    """Creates a list of dates between a specified start and end date.

    Parameters
    ----------
    start_date : str
        Start date, formatted as 'YYYY-MM-DD'.

    end_date : str
        Start date, formatted as 'YYYY-MM-DD'.

    Returns
    -------
    date_range : list (of str)
        List of dates between and including the start and end dates,
        with each date formatted as 'YYYYMMDD'.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Get list of dates
    dates = [
        dt.datetime.strftime(date, "%Y%m%d")
        for date in pd.date_range(start=start_date, end=end_date)
    ]

    return dates


def create_metadata(
    array, transform, driver="GTiff", nodata=0, count=1, crs="epsg:4326"
):
    """Creates export metadata, for use with
    exporting an array to raster format.

    Parameters
    ----------
    array : numpy array
        Array containing data for export.

    transform : rasterio.transform affine object
        Affine transformation for the georeferenced array.

    driver : str
        File type/format for export. Defaults to GeoTiff ('GTiff').

    nodata : int or float
        Value in the array indicating no data. Defaults to 0.

    count : int
        Number of bands in the array for export. Defaults to 1.

    crs : str
        Coordinate reference system for the georeferenced
        array. Defaults to EPSG 4326 ('epsg:4326').

    Returns
    -------
    metadata : dict
        Dictionary containing the export metadata.

    Example
    -------
        >>> # Imports
        >>> import numpy as np
        >>> from rasterio.transform import from_origin
        >>> # Create array
        >>> arr = np.array([[1,2],[3,4]])
        >>> transform = from_origin(-73.0, 43.0, 0.5, 0.5)
        >>> meta = create_metadata(arr, transform)
        # Display metadata
        >>> meta
        {'driver': 'GTiff',
         'dtype': dtype('int32'),
         'nodata': 0,
         'width': 2,
         'height': 2,
         'count': 1,
         'crs': 'epsg:4326',
         'transform': Affine(0.5, 0.0, -73.0,
                0.0, -0.5, 43.0)}
    """
    # Define metadata
    metadata = {
        "driver": driver,
        "dtype": array.dtype,
        "nodata": nodata,
        "width": array.shape[1],
        "height": array.shape[0],
        "count": count,
        "crs": crs,
        "transform": transform,
    }

    return metadata


def create_transform_vnp46a1(hdf5):
    """Creates a geographic transform for a VNP46A1 HDF5 file,
    based on longitude bounds, latitude bounds, and cell size.

    Parameters
    ----------
    hdf5 : str
        Path to an existsing VNP46A1 HDF5 file.

    Returns
    -------
    transform : affine.Affine object
        Affine transformation for the georeferenced array.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Extract bounding box from top-level dataset
    with rio.open(hdf5) as dataset:
        longitude_min = int(
            dataset.tags()["HDFEOS_GRIDS_VNP_Grid_DNB_WestBoundingCoord"]
        )
        longitude_max = int(
            dataset.tags()["HDFEOS_GRIDS_VNP_Grid_DNB_EastBoundingCoord"]
        )
        latitude_min = int(
            dataset.tags()["HDFEOS_GRIDS_VNP_Grid_DNB_SouthBoundingCoord"]
        )
        latitude_max = int(
            dataset.tags()["HDFEOS_GRIDS_VNP_Grid_DNB_NorthBoundingCoord"]
        )

        # Extract number of row and columns from first
        #  Science Data Set (subdataset/band)
        with rio.open(dataset.subdatasets[0]) as science_data_set:
            num_rows, num_columns = (
                science_data_set.meta.get("height"),
                science_data_set.meta.get("width"),
            )

    # Define transform (top-left corner, cell size)
    transform = from_origin(
        longitude_min,
        latitude_max,
        (longitude_max - longitude_min) / num_columns,
        (latitude_max - latitude_min) / num_rows,
    )

    return transform


def export_array(array, output_path, metadata):
    """Exports a numpy array to a GeoTiff.

    Parameters
    ----------
    array : numpy array
        Numpy array to be exported to GeoTiff.

    output_path : str
        Path to the output file (includeing filename).

    metadata : dict
        Dictionary containing the metadata required
        for export.

    Returns
    -------
    output_message : str
        Message indicating success or failure of export.

    Example
    -------
        >>> # Define export output paths
        >>> radiance_mean_outpath = os.path.join(
        ...     output_directory,
        ...     "radiance-mean.tif")
        # Define export transform
        >>> transform = from_origin(
        ...     lon_min,
        ...     lat_max,
        ...     coord_spacing,
        ...     coord_spacing)
        >>> # Define export metadata
        >>> export_metadata = {
        ...     "driver": "GTiff",
        ...     "dtype": radiance_mean.dtype,
        ...     "nodata": 0,
        ...     "width": radiance_mean.shape[1],
        ...     "height": radiance_mean.shape[0],
        ...     "count": 1,
        ...     "crs": 'epsg:4326',
        ...     "transform": transform
        ... }
        >>> # Export mean radiance
        >>> export_array(
        >>>     array=radiance_mean,
        >>>     output_path=radiance_mean_outpath,
        >>>     metadata=export_metadata)
        Exported: radiance-mean.tif
    """
    # Write numpy array to GeoTiff
    try:
        with rio.open(output_path, "w", **metadata) as dst:
            dst.write(array, 1)
    except Exception as error:
        output_message = print(f"ERROR: {error}")
    else:
        output_message = print(f"Exported: {os.path.split(output_path)[-1]}")

    return output_message


def extract_acquisition_date_vnp46a1(hdf5_path):
    """Returns the acquisition date of a VNP46A1 HDF5 file.

    Parameters
    ----------
    hdf5_path : str
        Path to a VNP46A1 HDF5 file.

    Returns
    -------
    acquisition_date : str
        Acquisition date of the image, formatted as 'YYYY-MM-DD'.

    Example
    -------
        >>> hdf5_file = "VNP46A1.A2020001.h30v05.001.2020004003738.h5"
        >>> extract_acquisition_date_vnp46a1(hdf5_file)
        '2020-01-01'
    """
    # Open file and extract date
    with rio.open(hdf5_path) as dataset:
        acquisition_date = dataset.tags()[
            "HDFEOS_GRIDS_VNP_Grid_DNB_RangeBeginningDate"
        ]

    return acquisition_date


def extract_band_vnp46a1(hdf5_path, band_name):
    """Extracts the specified band (Science Data Set) from a NASA VNP46A1
    HDF5 file.

    Available Science Data Sets include:

    BrightnessTemperature_M12
    Moon_Illumination_Fraction
    Moon_Phase_Angle
    QF_Cloud_Mask
    QF_DNB
    QF_VIIRS_M10
    QF_VIIRS_M11
    QF_VIIRS_M12
    QF_VIIRS_M13
    QF_VIIRS_M15
    QF_VIIRS_M16
    BrightnessTemperature_M13
    Radiance_M10
    Radiance_M11
    Sensor_Azimuth
    Sensor_Zenith
    Solar_Azimuth
    Solar_Zenith
    UTC_Time
    BrightnessTemperature_M15
    BrightnessTemperature_M16
    DNB_At_Sensor_Radiance_500m
    Glint_Angle
    Granule
    Lunar_Azimuth
    Lunar_Zenith

    Parameters
    ----------
    hdf5_path : str
        Path to the VNP46A1 HDF5 (.h5) file.

    band_name : str
        Name of the band (Science Data Set) to be extracted. Must be an exact
        match to an available Science Data Set.

    Returns
    -------
    band : numpy array
        Array containing the data for the specified band (Science Data Set).

    Example
    -------
        >>> qf_cloud_mask = extract_band_vnp46a1(
        ...     hdf5='VNP46A1.A2020001.h30v05.001.2020004003738.h5',
        ...     band='QF_Cloud_Mask'
        ... )
        >>> type(qf_cloud_mask)
        numpy.ndarray
    """
    # Raise error for invalid band name
    band_names = [
        "BrightnessTemperature_M12",
        "Moon_Illumination_Fraction",
        "Moon_Phase_Angle",
        "QF_Cloud_Mask",
        "QF_DNB",
        "QF_VIIRS_M10",
        "QF_VIIRS_M11",
        "QF_VIIRS_M12",
        "QF_VIIRS_M13",
        "QF_VIIRS_M15",
        "QF_VIIRS_M16",
        "BrightnessTemperature_M13",
        "Radiance_M10",
        "Radiance_M11",
        "Sensor_Azimuth",
        "Sensor_Zenith",
        "Solar_Azimuth",
        "Solar_Zenith",
        "UTC_Time",
        "BrightnessTemperature_M15",
        "BrightnessTemperature_M16",
        "DNB_At_Sensor_Radiance_500m",
        "Glint_Angle",
        "Granule",
        "Lunar_Azimuth",
        "Lunar_Zenith",
    ]
    if band_name not in band_names:
        raise ValueError(
            f"Invalid band name. Must be one of the following: {band_names}"
        )

    # Open top-level dataset, loop through Science Data Sets (subdatasets),
    #  and extract specified band
    with rio.open(hdf5_path) as dataset:
        for science_data_set in dataset.subdatasets:
            if re.search(f"{band_name}$", science_data_set):
                with rio.open(science_data_set) as src:
                    band = src.read(1)

    return band


def extract_date_vnp46a1(geotiff_path):
    """Extracts the file date from a preprocessed VNP46A1 GeoTiff.

    Parameters
    ----------
    geotiff_path : str
        Path to the GeoTiff file.

    Returns
    -------
    date : str
        Acquisition date of the preprocessed VNP46A1 GeoTiff.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Get date (convert YYYYJJJ to YYYYMMDD)
    date = dt.datetime.strptime(
        os.path.basename(geotiff_path)[9:16], "%Y%j"
    ).strftime("%Y%m%d")

    return date


def extract_geotiff_bounding_box(geotiff_path):
    """Extracts the bounding box from a GeoTiff file.

    Parameters
    ----------
    geotiff_path : str
        Path to the GeoTiff file.

    Returns
    -------
    bounding_box : rasterio.coords.BoundingBox
        Bounding box for the GeoTiff

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Extract bounding box
    with rio.open(geotiff_path) as src:
        bounding_box = src.bounds

    return bounding_box


def extract_qa_bits(qa_band, start_bit, end_bit):
    """Extracts the QA bitmask values for a specified bitmask (starting
     and ending bit).

    Parameters
    ----------
    qa_band : numpy array
        Array containing the raw QA values (base-2) for all bitmasks.

    start_bit : int
        First bit in the bitmask.

    end_bit : int
        Last bit in the bitmask.

    Returns
    -------
    qa_values : numpy array
        Array containing the extracted QA values (base-10) for the
        bitmask.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Initialize QA bit string/pattern to check QA band against
    qa_bits = 0

    # Add each specified QA bit flag value/string/pattern
    #  to the QA bits to check/extract
    for bit in range(start_bit, end_bit + 1):
        qa_bits += bit ** 2

    # Check QA band against specified QA bits to see what
    #  QA flag values are set
    qa_flags_set = qa_band & qa_bits

    # Get base-10 value that matches bitmask documentation
    #  (0-1 for single bit, 0-3 for 2 bits, or 0-2^N for N bits)
    qa_values = qa_flags_set >> start_bit

    return qa_values


def preprocess_vnp46a1(hdf5_path, output_folder):
    """Preprocessed a NASA VNP46A1 HDF5 (.h5 file)

    Preprocessing steps include masking data for fill values, clouds, and
    sensor problems, filling masked values, and exporting data to a GeoTiff.

    Parameters
    ----------
    hdf5_path : str
        Path to the VNP46A1 HDF5 (.h5) file to be preprocessed.

    output_folder : str
        Path to the folder where the preprocessed file will be exported to.

    Returns
    -------
    message : str
        Indication of preprocessing completion status (success or failure).

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Preprocess VNP46A1 HDF5 file
    print(f"Started preprocessing: {os.path.basename(hdf5_path)}")
    try:
        print("Extracting bands...")
        # Extract DNB_At_Sensor_Radiance_500m, QF_Cloud_Mask, QF_DNB
        dnb_at_sensor_radiance = extract_band_vnp46a1(
            hdf5_path=hdf5_path, band_name="DNB_At_Sensor_Radiance_500m"
        )
        qf_cloud_mask = extract_band_vnp46a1(
            hdf5_path=hdf5_path, band_name="QF_Cloud_Mask"
        )
        qf_dnb = extract_band_vnp46a1(hdf5_path=hdf5_path, band_name="QF_DNB")

        print("Applying scale factor...")
        # Apply scale factor to radiance values
        dnb_at_sensor_radiance_scaled = (
            dnb_at_sensor_radiance.astype("float") * 0.1
        )

        print("Masking for fill values...")
        # Mask radiance for fill value (DNB_At_Sensor_Radiance_500m == 65535)
        masked_for_fill_value = ma.masked_where(
            dnb_at_sensor_radiance_scaled == 6553.5,
            dnb_at_sensor_radiance_scaled,
            copy=True,
        )

        print("Masking for clouds...")
        # Extract QF_Cloud_Mask bits 6-7 (Cloud Detection Results &
        #  Confidence Indicator)
        cloud_detection_bitmask = extract_qa_bits(
            qa_band=qf_cloud_mask, start_bit=6, end_bit=7
        )

        # Mask radiance for 'probably cloudy' (cloud_detection_bitmask == 2)
        masked_for_probably_cloudy = ma.masked_where(
            cloud_detection_bitmask == 2, masked_for_fill_value, copy=True
        )

        # Mask radiance for 'confident cloudy' (cloud_detection_bitmask == 3)
        masked_for_confident_cloudy = ma.masked_where(
            cloud_detection_bitmask == 3, masked_for_probably_cloudy, copy=True
        )

        print("Masking for sensor problems...")
        # Mask radiance for sensor problems (QF_DNB != 0)
        #  (0 = no problems, any number > 0 means some kind of issue)
        masked_for_sensor_problems = ma.masked_where(
            qf_dnb > 0, masked_for_confident_cloudy, copy=True
        )

        print("Filling masked values...")
        # Set fill value to np.nan and fill masked values
        ma.set_fill_value(masked_for_sensor_problems, np.nan)
        filled_data = masked_for_sensor_problems.filled()

        print("Creating metadata...")
        # Create metadata (for export)
        metadata = create_metadata(
            array=filled_data,
            transform=create_transform_vnp46a1(hdf5_path),
            driver="GTiff",
            nodata=np.nan,
            count=1,
            crs="epsg:4326",
        )

        print("Exporting to GeoTiff...")
        # Export masked array to GeoTiff (no data set to np.nan in export)
        export_name = (
            f"{os.path.basename(hdf5_path)[:-3].lower().replace('.', '-')}.tif"
        )
        export_array(
            array=filled_data,
            output_path=os.path.join(output_folder, export_name),
            metadata=metadata,
        )
    except Exception as error:
        message = print(f"Preprocessing failed: {error}")
    else:
        message = print(
            f"Completed preprocessing: {os.path.basename(hdf5_path)}\n"
        )

    return message


def read_geotiff_into_array(geotiff_path, dimensions=1):
    """Reads a GeoTif file into a NumPy array.

    Parameters
    ----------
    geotiff_path : str
        Path to the GeoTiff file.

    dimensions : int, optional
        Number of bands to read in. Default value is 1.

    Returns
    -------
    array : numpy array
        Array containing the data.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Read-in array
    with rio.open(geotiff_path) as src:
        array = src.read(dimensions)

    return array
