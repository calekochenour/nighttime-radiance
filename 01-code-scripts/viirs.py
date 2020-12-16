""" Module to work with NASA VIIRS DNB data """

import os
import re
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import numpy.ma as ma
import pandas as pd
import rasterio as rio
from rasterio.transform import from_origin
import earthpy.plot as ep
import earthpy.spatial as es


def calculate_statistic(data, statistic="mean"):
    """Calculates the specified statistic over input arrays covering
    the same geographic area.

    Parameters
    ----------
    data : list of numpy arrays
        List of arrays containing the data. Individual arrays can
        contain NaN values.

    statistic : str (optional)
        Statistic to be calculated over the arrays in the
        list. Default value is 'mean'. Function supports
        'mean', 'variance', 'deviation', and 'median'.

    Returns
    -------
    data_statistic : numpy array
        Array containing the statistic value for each pixel, computed
        over the number of arrays in the input list.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Raise errors
    if not isinstance(data, list):
        raise TypeError("Input data must be of type list.")

    # Calculate statistic (mean, variance, standard deviation, or median)
    if statistic == "mean":
        data_statistic = np.nanmean(np.stack(data), axis=0)
    elif statistic == "variance":
        data_statistic = np.nanvar(np.stack(data), axis=0)
    elif statistic == "deviation":
        data_statistic = np.nanstd(np.stack(data), axis=0)
    elif statistic == "median":
        data_statistic = np.nanmedian(np.stack(data), axis=0)
    else:
        raise ValueError(
            "Invalid statistic. Function supports "
            "'mean', 'variance', 'deviation', or 'median'."
        )

    return data_statistic


def clip_vnp46a1(geotiff_path, clip_boundary, clip_country, output_folder):
    """Clips an image to a bounding box and exports the clipped image to
    a GeoTiff file.

    Paramaters
    ----------
    geotiff_path : str
        Path to the GeoTiff image to be clipped.

    clip_boundary : geopandas geodataframe
        Geodataframe for containing the boundary used for clipping.

    clip_country : str
        Name of the country the data is being clipped to. The country
        name is used in the name of the exported file. E.g. 'South Korea'.
        Spaces and capital letters are acceptable and handled within the
        function.

    output_folder : str
        Path to the folder where the clipped file will be exported to.

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
    # Clip VNP46A1 file
    print(
        f"Started clipping: Clip {os.path.basename(geotiff_path)} "
        f"to {clip_country} boundary"
    )
    try:
        print("Clipping image...")
        # Clip image (return clipped array and new metadata)
        with rio.open(geotiff_path) as src:
            cropped_image, cropped_metadata = es.crop_image(
                raster=src, geoms=clip_boundary
            )

        print("Setting export name...")
        # Set export name
        export_name = create_clipped_export_name(
            image_path=geotiff_path, country_name=clip_country
        )

        print("Exporting to GeoTiff...")
        # Export file
        export_array(
            array=cropped_image[0],
            output_path=os.path.join(output_folder, export_name),
            metadata=cropped_metadata,
        )
    except Exception as error:
        message = print(f"Clipping failed: {error}\n")
    else:
        message = print(
            f"Completed clipping: Clip {os.path.basename(geotiff_path)} "
            f"to {clip_country} boundary\n"
        )

    return message


def clip_vnp46a2(geotiff_path, clip_boundary, clip_country, output_folder):
    """Clips an image to a bounding box and exports the clipped image to
    a GeoTiff file.

    Paramaters
    ----------
    geotiff_path : str
        Path to the GeoTiff image to be clipped.

    clip_boundary : geopandas geodataframe
        Geodataframe for containing the boundary used for clipping.

    clip_country : str
        Name of the country the data is being clipped to. The country
        name is used in the name of the exported file. E.g. 'South Korea'.
        Spaces and capital letters are acceptable and handled within the
        function.

    output_folder : str
        Path to the folder where the clipped file will be exported to.

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
    # Clip VNP46A2 file
    print(
        f"Started clipping: Clip {os.path.basename(geotiff_path)} "
        f"to {clip_country} boundary"
    )
    try:
        print("Clipping image...")
        # Clip image (return clipped array and new metadata)
        with rio.open(geotiff_path) as src:
            cropped_image, cropped_metadata = es.crop_image(
                raster=src, geoms=clip_boundary
            )

        print("Setting export name...")
        # Set export name
        export_name = create_clipped_export_name(
            image_path=geotiff_path, country_name=clip_country
        )

        print("Exporting to GeoTiff...")
        # Export file
        export_array(
            array=cropped_image[0],
            output_path=os.path.join(output_folder, export_name),
            metadata=cropped_metadata,
        )
    except Exception as error:
        message = print(f"Clipping failed: {error}\n")
    else:
        message = print(
            f"Completed clipping: Clip {os.path.basename(geotiff_path)} "
            f"to {clip_country} boundary\n"
        )

    return message


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
        message = print(f"Concatenating failed: {error}\n")
    else:
        message = print(
            (
                f"Completed concatenating:\n    "
                f"{os.path.basename(west_geotiff_path)}\n    "
                f"{os.path.basename(east_geotiff_path)}\n"
            )
        )

    return message


def concatenate_preprocessed_vnp46a2(
    west_geotiff_path, east_geotiff_path, output_folder
):
    """Concatenates horizontally-adjacent preprocessed VNP46A2 GeoTiff
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
        message = print(f"Concatenating failed: {error}\n")
    else:
        message = print(
            (
                f"Completed concatenating:\n    "
                f"{os.path.basename(west_geotiff_path)}\n    "
                f"{os.path.basename(east_geotiff_path)}\n"
            )
        )

    return message


def create_clipped_export_name(image_path, country_name):
    """Creates a file name indicating a clipped file.

    Paramaters
    ----------
    image_path : str
        Path to the original (unclipped image).

    Returns
    -------
    export_name : str
        New file name for export, indicating clipping.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Set export name
    image_source = os.path.basename(image_path)[:7]
    image_date = extract_date_vnp46a1(image_path)
    image_country = country_name.replace(" ", "-").lower()
    export_name = f"{image_source}-{image_date}-clipped-{image_country}.tif"

    return export_name


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

    # Replace the single horizontal grid number with both the West and
    #  East numbers; remove series and processing time information
    data_source_and_date = os.path.basename(west_image_path)[:16]
    vertical_grid_number = os.path.basename(west_image_path)[21:23]
    export_name = (
        f"{data_source_and_date}-h{west_image_horizontal_grid_number}"
        f"{east_image_horizontal_grid_number}v{vertical_grid_number}.tif"
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


def create_transform_vnp46a2(hdf5):
    """Creates a geographic transform for a VNP46A2 HDF5 file,
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
        longitude_min = int(dataset.tags()["WestBoundingCoord"])
        longitude_max = int(dataset.tags()["EastBoundingCoord"])
        latitude_min = int(dataset.tags()["SouthBoundingCoord"])
        latitude_max = int(dataset.tags()["NorthBoundingCoord"])

        # Extract number of row and columns from first
        #  Science Data Set (subdataset/band)
        with rio.open(dataset.subdatasets[0]) as band:
            num_rows, num_columns = (
                band.meta.get("height"),
                band.meta.get("width"),
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


def extract_band_vnp46a2(hdf5_path, band_name):
    """Extracts the specified band (Science Data Set) from a NASA VNP46A2
    HDF5 file.

    Available Science Data Sets include:

    DNB_BRDF-Corrected_NTL
    DNB_Lunar_Irradiance
    Gap_Filled_DNB_BRDF-Corrected_NTL
    Latest_High_Quality_Retrieval
    Mandatory_Quality_Flag
    QF_Cloud_Mask
    Snow_Flag

    Parameters
    ----------
    hdf5_path : str
        Path to the VNP46A2 HDF5 (.h5) file.

    band_name : str
        Name of the band (Science Data Set) to be extracted. Must be an exact
        match to an available Science Data Set.

    Returns
    -------
    band : numpy array
        Array containing the data for the specified band (Science Data Set).

    Example
    -------
        >>> qf_cloud_mask = extract_band_vnp46a2(
        ...     hdf5='VNP46A2.A2016153.h30v05.001.2020267141459.h5',
        ...     band='QF_Cloud_Mask'
        ... )
        >>> type(qf_cloud_mask)
        numpy.ndarray
    """
    # Raise error for invalid band name
    band_names = [
        "DNB_BRDF-Corrected_NTL",
        "DNB_Lunar_Irradiance",
        "Gap_Filled_DNB_BRDF-Corrected_NTL",
        "Latest_High_Quality_Retrieval",
        "Mandatory_Quality_Flag",
        "QF_Cloud_Mask",
        "Snow_Flag",
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


def extract_date_vnp46a2(geotiff_path):
    """Extracts the file date from a preprocessed VNP46A2 GeoTiff.

    Parameters
    ----------
    geotiff_path : str
        Path to the GeoTiff file.

    Returns
    -------
    date : str
        Acquisition date of the preprocessed VNP46A2 GeoTiff.

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


def extract_geotiff_metadata(geotiff_path):
    """Extract metadata from a GeoTiff file.

    Parameters
    ----------
    geotiff_path : str
        Path to the GeoTiff file.

    Returns
    -------
    metadata : dict
        Dictionary containing the metadata.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Read-in array
    with rio.open(geotiff_path) as src:
        metadata = src.meta

    return metadata


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


def get_masking_details(array):
    """Returns information about how many pixels are masked in an array.

    Parameters
    ----------
    array : numpy.ma.core.MaskedArray
        Masked array.

    Returns
    -------
    tuple

        total : int
            Total number of pixels in the array.

        masked : int
            Number of masked pixels in the array.

        unmasked : int
            Number of unmasked pixels in the array.

    message : str
        Message providing the masking information.

    Example
    -------
    >>>
    >>>
    >>>
    >>>
    """
    # Get masking information
    total = array.shape[0] * array.shape[1]
    masked = ma.count_masked(array)
    unmasked = array.count()

    # Create message
    message = print(f"Masked: {masked}/{total}, Unmasked: {unmasked}/{total}")

    return message


def get_unique_values(array):
    """Returns the unique values from a NumPy array as a list.

    Parameters
    ----------
    array : numppy array
        Array from which to get the unique values.

    Returns
    -------
    values : list
        List of unique values from the array.

    Example
    ------
    >>>
    >>>
    >>>
    >>>
    """
    # Get unique values
    values = np.unique(array).tolist()

    return values


def plot_quality_flag_bitmask(bitmask_array, bitmask_name, axis):
    """Plots the discrete bitmask values for an image.

    Parameters
    ----------
    bitmask_array : numpy array
        Array containing the base-10 bitmask values.

    bitmask_name : str
        Name of the bitmask layer. Valid names: 'Day/Night', 'Land/Water
        Background', 'Cloud Mask Quality', 'Cloud Detection',
        'Shadow Detected', 'Cirrus Detection', 'Snow/Ice Surface', and
        'QF DNB'.

    Returns
    -------
    tuple

        fig : matplotlib.figure.Figure object
            The figure object associated with the histogram.

        ax : matplotlib.axes._subplots.AxesSubplot objects
            The axes objects associated with the histogram.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Store possible bitmask values and titles (for plotting)
    vnp46a1_bitmasks = {
        "Day/Night": {"values": [0, 1], "labels": ["Night", "Day"]},
        "Land/Water Background": {
            "values": [0, 1, 2, 3, 5],
            "labels": [
                "Land & Desert",
                "Land no Desert",
                "Inland Water",
                "Sea Water",
                "Coastal",
            ],
        },
        "Cloud Mask Quality": {
            "values": [0, 1, 2, 3],
            "labels": ["Poor", "Low", "Medium", "High"],
        },
        "Cloud Detection": {
            "values": [0, 1, 2, 3],
            "labels": [
                "Confident Clear",
                "Probably Clear",
                "Probably Cloudy",
                "Confident Cloudy",
            ],
        },
        "Shadow Detected": {
            "values": [0, 1],
            "labels": ["No Shadow", "Shadow"],
        },
        "Cirrus Detection": {
            "values": [0, 1],
            "labels": ["No Cirrus Cloud", "Cirrus Cloud"],
        },
        "Snow/Ice Surface": {
            "values": [0, 1],
            "labels": ["No Snow/Ice", "Snow/Ice"],
        },
        "QF DNB": {
            "values": [0, 1, 2, 4, 8, 16, 256, 512, 1024, 2048],
            "labels": [
                "No Sensor Problems",
                "Substitute Calibration",
                "Out of Range",
                "Saturation",
                "Temperature not Nominal",
                "Stray Light",
                "Bowtie Deleted / Range Bit",
                "Missing EV",
                "Calibration Fail",
                "Dead Detector",
            ],
        },
    }

    # Raise errors
    if bitmask_name not in vnp46a1_bitmasks.keys():
        raise ValueError(
            f"Invalid name. Valid names are: {list(vnp46a1_bitmasks.keys())}"
        )

    # Get values and labels for bitmask
    bitmask_values = vnp46a1_bitmasks.get(bitmask_name).get("values")
    bitmask_labels = vnp46a1_bitmasks.get(bitmask_name).get("labels")

    # Create colormap with the number of values in the bitmask
    cmap = plt.cm.get_cmap("tab20b", len(bitmask_values))

    # Add start bin of 0 to list of bitmask values
    bins = [0] + bitmask_values

    # Normalize colormap to discrete intervals
    bounds = [((a + b) / 2) for a, b in zip(bins[:-1], bins[1::1])] + [
        2 * (bins[-1]) - bins[-2]
    ]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # Plot bitmask on axis
    bitmask = axis.imshow(bitmask_array, cmap=cmap, norm=norm)
    ep.draw_legend(
        im_ax=bitmask,
        classes=bitmask_values,
        cmap=cmap,
        titles=bitmask_labels,
    )
    axis.set_title(f"{bitmask_name}", size=16)
    axis.set_axis_off()

    return axis


def plot_quality_flag_bitmask_vnp46a2(bitmask_array, bitmask_name, axis):
    """Plots the discrete bitmask values for an image.

    Parameters
    ----------
    bitmask_array : numpy array
        Array containing the base-10 bitmask values.

    bitmask_name : str
        Name of the bitmask layer. Valid names: 'Mandatory Quality Flag',
        'Snow Flag', 'Day/Night', 'Land/Water Background',
        'Cloud Mask Quality', 'Cloud Detection', 'Shadow Detected',
        'Cirrus Detection', and 'Snow/Ice Surface'.

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot objects
        The axes objects associated with plot.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Store possible bitmask values and titles (for plotting)
    vnp46a2_bitmasks = {
        "Mandatory Quality Flag": {
            "values": [0, 1, 2, 255],
            "labels": [
                "High-Quality (Persistent)",
                "High-Quality (Ephemeral)",
                "Poor-Quality",
                "No Retrieval",
            ],
        },
        "Snow Flag": {
            "values": [0, 1, 255],
            "labels": ["No Snow/Ice", "Snow/Ice", "Fill Value"],
        },
        "Day/Night": {"values": [0, 1], "labels": ["Night", "Day"]},
        "Land/Water Background": {
            "values": [0, 1, 2, 3, 5, 7],
            "labels": [
                "Land & Desert",
                "Land no Desert",
                "Inland Water",
                "Sea Water",
                "Coastal",
                "No Data / Unknown",
            ],
        },
        "Cloud Mask Quality": {
            "values": [0, 1, 2, 3],
            "labels": ["Poor", "Low", "Medium", "High"],
        },
        "Cloud Detection": {
            "values": [0, 1, 2, 3],
            "labels": [
                "Confident Clear",
                "Probably Clear",
                "Probably Cloudy",
                "Confident Cloudy",
            ],
        },
        "Shadow Detected": {
            "values": [0, 1],
            "labels": ["No Shadow", "Shadow"],
        },
        "Cirrus Detection": {
            "values": [0, 1],
            "labels": ["No Cirrus Cloud", "Cirrus Cloud"],
        },
        "Snow/Ice Surface": {
            "values": [0, 1],
            "labels": ["No Snow/Ice", "Snow/Ice"],
        },
    }

    # Raise errors
    if bitmask_name not in vnp46a2_bitmasks.keys():
        raise ValueError(
            f"Invalid name. Valid names are: {list(vnp46a2_bitmasks.keys())}"
        )

    # Plot bitmask on axis
    bitmask = axis.imshow(
        bitmask_array,
        #         cmap="Accent",
        vmin=vnp46a2_bitmasks.get(bitmask_name).get("values")[0],
        vmax=vnp46a2_bitmasks.get(bitmask_name).get("values")[-1],
    )
    ep.draw_legend(
        im_ax=bitmask,
        classes=vnp46a2_bitmasks.get(bitmask_name).get("values"),
        titles=vnp46a2_bitmasks.get(bitmask_name).get("labels"),
    )
    axis.set_title(f"{bitmask_name}", size=16)
    axis.set_axis_off()

    return axis


def plot_quality_flag_bitmask_single_band(bitmask_array, bitmask_name):
    """Plots the discrete bitmask values for an image.

    Parameters
    ----------
    bitmask_array : numpy array
        Array containing the base-10 bitmask values.

    bitmask_name : str
        Name of the bitmask layer. Valid names: 'Day/Night', 'Land/Water
        Background', 'Cloud Mask Quality', 'Cloud Detection',
        'Shadow Detected', 'Cirrus Detection', 'Snow/Ice Surface', and
        'QF DNB'.

    Returns
    -------
    tuple

        fig : matplotlib.figure.Figure object
            The figure object associated with the histogram.

        ax : matplotlib.axes._subplots.AxesSubplot objects
            The axes objects associated with the histogram.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Store possible bitmask values and titles (for plotting)
    vnp46a1_bitmasks = {
        "Day/Night": {"values": [0, 1], "labels": ["Night", "Day"]},
        "Land/Water Background": {
            "values": [0, 1, 2, 3, 5],
            "labels": [
                "Land & Desert",
                "Land no Desert",
                "Inland Water",
                "Sea Water",
                "Coastal",
            ],
        },
        "Cloud Mask Quality": {
            "values": [0, 1, 2, 3],
            "labels": ["Poor", "Low", "Medium", "High"],
        },
        "Cloud Detection": {
            "values": [0, 1, 2, 3],
            "labels": [
                "Confident Clear",
                "Probably Clear",
                "Probably Cloudy",
                "Confident Cloudy",
            ],
        },
        "Shadow Detected": {
            "values": [0, 1],
            "labels": ["No Shadow", "Shadow"],
        },
        "Cirrus Detection": {
            "values": [0, 1],
            "labels": ["No Cirrus Cloud", "Cirrus Cloud"],
        },
        "Snow/Ice Surface": {
            "values": [0, 1],
            "labels": ["No Snow/Ice", "Snow/Ice"],
        },
        "QF DNB": {
            "values": [0, 1, 2, 4, 8, 16, 256, 512, 1024, 2048],
            "labels": [
                "No Sensor Problems",
                "Substitute Calibration",
                "Out of Range",
                "Saturation",
                "Temperature not Nominal",
                "Stray Light",
                "Bowtie Deleted / Range Bit",
                "Missing EV",
                "Calibration Fail",
                "Dead Detector",
            ],
        },
    }

    # Raise errors
    if bitmask_name not in vnp46a1_bitmasks.keys():
        raise ValueError(
            f"Invalid name. Valid names are: {list(vnp46a1_bitmasks.keys())}"
        )

    # Get values and labels for bitmask
    bitmask_values = vnp46a1_bitmasks.get(bitmask_name).get("values")
    bitmask_labels = vnp46a1_bitmasks.get(bitmask_name).get("labels")

    # Create colormap with the number of values in the bitmask
    cmap = plt.cm.get_cmap("tab20b", len(bitmask_values))

    # Add start bin of 0 to list of bitmask values
    bins = [0] + bitmask_values

    # Normalize colormap to discrete intervals
    bounds = [((a + b) / 2) for a, b in zip(bins[:-1], bins[1::1])] + [
        2 * (bins[-1]) - bins[-2]
    ]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # Plot bitmask
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(12, 8))
        bitmask = ax.imshow(bitmask_array, cmap=cmap, norm=norm)
        ep.draw_legend(
            im_ax=bitmask,
            classes=bitmask_values,
            cmap=cmap,
            titles=bitmask_labels,
        )
        ax.set_title(f"{bitmask_name} Bitmask", size=20)
        ax.set_axis_off()

    return fig, ax


def plot_quality_flag_bitmask_single_band_vnp46a2(bitmask_array, bitmask_name):
    """Plots the discrete bitmask values for an image.

    Parameters
    ----------
    bitmask_array : numpy array
        Array containing the base-10 bitmask values.

    bitmask_name : str
        Name of the bitmask layer. Valid names: 'Mandatory Quality Flag',
        'Snow Flag', 'Day/Night', 'Land/Water Background',
        'Cloud Mask Quality', 'Cloud Detection', 'Shadow Detected',
        'Cirrus Detection', and 'Snow/Ice Surface'.

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot objects
        The axes objects associated with plot.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Store possible bitmask values and titles (for plotting)
    vnp46a2_bitmasks = {
        "Mandatory Quality Flag": {
            "values": [0, 1, 2, 255],
            "labels": [
                "High-Quality (Persistent)",
                "High-Quality (Ephemeral)",
                "Poor-Quality",
                "No Retrieval",
            ],
        },
        "Snow Flag": {
            "values": [0, 1, 255],
            "labels": ["No Snow/Ice", "Snow/Ice", "Fill Value"],
        },
        "Day/Night": {"values": [0, 1], "labels": ["Night", "Day"]},
        "Land/Water Background": {
            "values": [0, 1, 2, 3, 5, 7],
            "labels": [
                "Land & Desert",
                "Land no Desert",
                "Inland Water",
                "Sea Water",
                "Coastal",
                "No Data / Unknown",
            ],
        },
        "Cloud Mask Quality": {
            "values": [0, 1, 2, 3],
            "labels": ["Poor", "Low", "Medium", "High"],
        },
        "Cloud Detection": {
            "values": [0, 1, 2, 3],
            "labels": [
                "Confident Clear",
                "Probably Clear",
                "Probably Cloudy",
                "Confident Cloudy",
            ],
        },
        "Shadow Detected": {
            "values": [0, 1],
            "labels": ["No Shadow", "Shadow"],
        },
        "Cirrus Detection": {
            "values": [0, 1],
            "labels": ["No Cirrus Cloud", "Cirrus Cloud"],
        },
        "Snow/Ice Surface": {
            "values": [0, 1],
            "labels": ["No Snow/Ice", "Snow/Ice"],
        },
    }

    # Raise errors
    if bitmask_name not in vnp46a2_bitmasks.keys():
        raise ValueError(
            f"Invalid name. Valid names are: {list(vnp46a2_bitmasks.keys())}"
        )

    # Plot bitmask on axis
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(12, 8))
        bitmask = ax.imshow(
            bitmask_array,
            #             cmap="Accent",
            vmin=vnp46a2_bitmasks.get(bitmask_name).get("values")[0],
            vmax=vnp46a2_bitmasks.get(bitmask_name).get("values")[-1],
        )
        ep.draw_legend(
            im_ax=bitmask,
            classes=vnp46a2_bitmasks.get(bitmask_name).get("values"),
            titles=vnp46a2_bitmasks.get(bitmask_name).get("labels"),
        )
        ax.set_title(f"{bitmask_name}", size=16)
        ax.set_axis_off()

    return fig, ax


def plot_quality_flags_vnp46a1(vnp46a1_quality_stack, data_source="NASA"):
    """Plots all VIIRS VNP46A1 DNB QF Cloud Mask bitmasks and the
    QF DNB bitmask.

    Parameters
    ----------
    vnp46a1_quality_stack : numpy array
        3D array containing the quality flag bitmask layers.

    data_source : str, optional
        Location of the data. Default value is 'NASA'.

    Returns
    -------
    tuple

        fig : matplotlib.figure.Figure object
            The figure object associated with the histogram.

        ax : matplotlib.axes._subplots.AxesSubplot objects
            The axes objects associated with the histogram.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Configure plot
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(15, 20))
        plt.suptitle("VNP46A1 Quality Flag Bitmasks", size=20)
        plt.subplots_adjust(top=0.935)

        # Plot bitmasks
        # Day/night
        plot_quality_flag_bitmask(
            bitmask_array=vnp46a1_quality_stack[0],
            bitmask_name="Day/Night",
            axis=ax[0][0],
        )

        # Land/water background
        plot_quality_flag_bitmask(
            bitmask_array=vnp46a1_quality_stack[1],
            bitmask_name="Land/Water Background",
            axis=ax[0][1],
        )

        # Cloud mask quality
        plot_quality_flag_bitmask(
            bitmask_array=vnp46a1_quality_stack[2],
            bitmask_name="Cloud Mask Quality",
            axis=ax[1][0],
        )

        # Cloud detection
        plot_quality_flag_bitmask(
            bitmask_array=vnp46a1_quality_stack[3],
            bitmask_name="Cloud Detection",
            axis=ax[1][1],
        )

        # Shadow detected
        plot_quality_flag_bitmask(
            bitmask_array=vnp46a1_quality_stack[4],
            bitmask_name="Shadow Detected",
            axis=ax[2][0],
        )

        # Cirrus detection
        plot_quality_flag_bitmask(
            bitmask_array=vnp46a1_quality_stack[5],
            bitmask_name="Cirrus Detection",
            axis=ax[2][1],
        )

        # Snow/ice surface
        plot_quality_flag_bitmask(
            bitmask_array=vnp46a1_quality_stack[6],
            bitmask_name="Snow/Ice Surface",
            axis=ax[3][0],
        )

        # QF DNB
        plot_quality_flag_bitmask(
            bitmask_array=vnp46a1_quality_stack[7],
            bitmask_name="QF DNB",
            axis=ax[3][1],
        )

        # Add caption
        fig.text(
            0.5,
            0.1,
            f"Data Source: {data_source}",
            ha="center",
            fontsize=12,
        )

    return fig, ax


def plot_quality_flags_vnp46a2(vnp46a2_quality_stack, data_source="NASA"):
    """Plots all VIIRS VNP46A2 DNB QF Cloud Mask bitmasks, the Mandatory
    Quality Flag, and Snow Flag.

    Parameters
    ----------
    vnp46a2_quality_stack : numpy array
        3D array containing the quality flag bitmask layers.

    data_source : str, optional
        Location of the data. Default value is 'NASA'.

    Returns
    -------
    tuple

        fig : matplotlib.figure.Figure object
            The figure object associated with the histogram.

        ax : matplotlib.axes._subplots.AxesSubplot objects
            The axes objects associated with the histogram.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Configure plot
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(15, 20))
        plt.suptitle("VNP46A2 Quality Flag Bitmasks", size=20)
        plt.subplots_adjust(top=0.935)

        # Plot bitmasks
        # Mandatory Quality Flag
        plot_quality_flag_bitmask_vnp46a2(
            bitmask_array=vnp46a2_quality_stack[0],
            bitmask_name="Mandatory Quality Flag",
            axis=ax[0][0],
        )

        # Snow flag
        plot_quality_flag_bitmask_vnp46a2(
            bitmask_array=vnp46a2_quality_stack[1],
            bitmask_name="Snow Flag",
            axis=ax[0][1],
        )

        # Day/night
        plot_quality_flag_bitmask_vnp46a2(
            bitmask_array=vnp46a2_quality_stack[2],
            bitmask_name="Day/Night",
            axis=ax[1][0],
        )

        # Land/water background
        plot_quality_flag_bitmask_vnp46a2(
            bitmask_array=vnp46a2_quality_stack[3],
            bitmask_name="Land/Water Background",
            axis=ax[1][1],
        )

        # Cloud mask quality
        plot_quality_flag_bitmask_vnp46a2(
            bitmask_array=vnp46a2_quality_stack[4],
            bitmask_name="Cloud Mask Quality",
            axis=ax[2][0],
        )

        # Cloud detection
        plot_quality_flag_bitmask_vnp46a2(
            bitmask_array=vnp46a2_quality_stack[5],
            bitmask_name="Cloud Detection",
            axis=ax[2][1],
        )

        # Shadow detected
        plot_quality_flag_bitmask_vnp46a2(
            bitmask_array=vnp46a2_quality_stack[6],
            bitmask_name="Shadow Detected",
            axis=ax[3][0],
        )

        # Cirrus detection
        plot_quality_flag_bitmask_vnp46a2(
            bitmask_array=vnp46a2_quality_stack[6],
            bitmask_name="Cirrus Detection",
            axis=ax[3][1],
        )

        # Snow/ice surface
        plot_quality_flag_bitmask_vnp46a2(
            bitmask_array=vnp46a2_quality_stack[8],
            bitmask_name="Snow/Ice Surface",
            axis=ax[4][0],
        )

        # Add caption
        fig.text(
            0.5,
            0.1,
            f"Data Source: {data_source}",
            ha="center",
            fontsize=12,
        )

        # Remove unused axis
        fig.delaxes(ax[4][1])

    return fig, ax


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

        print("Masking for sea water...")
        # Extract QF_Cloud_Mask bits 1-3 (Land/Water Background)
        land_water_bitmask = extract_qa_bits(
            qa_band=qf_cloud_mask, start_bit=1, end_bit=3
        )

        # Mask radiance for sea water (land_water_bitmask == 3)
        masked_for_sea_water = ma.masked_where(
            land_water_bitmask == 3, masked_for_confident_cloudy, copy=True
        )

        print("Masking for sensor problems...")
        # Mask radiance for sensor problems (QF_DNB != 0)
        #  (0 = no problems, any number > 0 means some kind of issue)
        # masked_for_sensor_problems = ma.masked_where(
        #     qf_dnb > 0, masked_for_confident_cloudy, copy=True
        # )
        masked_for_sensor_problems = ma.masked_where(
            qf_dnb > 0, masked_for_sea_water, copy=True
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
        message = print(f"Preprocessing failed: {error}\n")
    else:
        message = print(
            f"Completed preprocessing: {os.path.basename(hdf5_path)}\n"
        )

    return message


def preprocess_vnp46a2(hdf5_path, output_folder):
    """Preprocessed a NASA VNP46A2 HDF5 (.h5 file)

    Preprocessing steps include masking data for fill values, clouds, and
    sensor problems, filling masked values, and exporting data to a GeoTiff.

    Parameters
    ----------
    hdf5_path : str
        Path to the VNP46A2 HDF5 (.h5) file to be preprocessed.

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
        # Extract DNB BRDF-Corrected radiance
        dnb_brdf_corrected_ntl = extract_band_vnp46a2(
            hdf5_path=hdf5_path, band_name="DNB_BRDF-Corrected_NTL"
        )

        # Extract Mandatory Quality Flag, QF Cloud Mask, and Snow Flag bands
        mandatory_quality_flag = extract_band_vnp46a2(
            hdf5_path=hdf5_path, band_name="Mandatory_Quality_Flag"
        )
        qf_cloud_mask = extract_band_vnp46a2(
            hdf5_path=hdf5_path, band_name="QF_Cloud_Mask"
        )

        print("Applying scale factor...")
        # Apply scale factor to radiance values
        dnb_brdf_corrected_ntl_scaled = (
            dnb_brdf_corrected_ntl.astype("float") * 0.1
        )

        print("Masking for fill values...")
        # Mask radiance for fill value (dnb_brdf_corrected_ntl == 65535)
        masked_for_fill_value = ma.masked_where(
            dnb_brdf_corrected_ntl_scaled == 6553.5,
            dnb_brdf_corrected_ntl_scaled,
            copy=True,
        )

        print("Masking for poor quality and no retrieval...")
        # Mask radiance for 'poor quality' (mandatory_quality_flag == 2)
        masked_for_poor_quality = ma.masked_where(
            mandatory_quality_flag == 2, masked_for_fill_value, copy=True
        )

        # Mask radiance for 'no retrieval' (mandatory_quality_flag == 255)
        masked_for_no_retrieval = ma.masked_where(
            mandatory_quality_flag == 255, masked_for_poor_quality, copy=True
        )

        print("Masking for clouds...")
        # Extract QF_Cloud_Mask bits 6-7 (Cloud Detection Results &
        #  Confidence Indicator)
        cloud_detection_bitmask = extract_qa_bits(
            qa_band=qf_cloud_mask, start_bit=6, end_bit=7
        )

        # Mask radiance for 'probably cloudy' (cloud_detection_bitmask == 2)
        masked_for_probably_cloudy = ma.masked_where(
            cloud_detection_bitmask == 2, masked_for_no_retrieval, copy=True
        )

        # Mask radiance for 'confident cloudy' (cloud_detection_bitmask == 3)
        masked_for_confident_cloudy = ma.masked_where(
            cloud_detection_bitmask == 3, masked_for_probably_cloudy, copy=True
        )

        print("Masking for sea water...")
        # Extract QF_Cloud_Mask bits 1-3 (Land/Water Background)
        land_water_bitmask = extract_qa_bits(
            qa_band=qf_cloud_mask, start_bit=1, end_bit=3
        )

        # Mask radiance for sea water (land_water_bitmask == 3)
        masked_for_sea_water = ma.masked_where(
            land_water_bitmask == 3, masked_for_confident_cloudy, copy=True
        )

        print("Filling masked values...")
        # Set fill value to np.nan and fill masked values
        ma.set_fill_value(masked_for_sea_water, np.nan)
        filled_data = masked_for_sea_water.filled()

        print("Creating metadata...")
        # Create metadata (for export)
        metadata = create_metadata(
            array=filled_data,
            transform=create_transform_vnp46a2(hdf5_path),
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
        message = print(f"Preprocessing failed: {error}\n")
    else:
        message = print(
            f"Completed preprocessing: {os.path.basename(hdf5_path)}\n"
        )

    return message


def read_geotiff_into_array(geotiff_path, dimensions=1):
    """Reads a GeoTiff file into a NumPy array.

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


def save_figure(output_path):
    """Saves the current figure to a specified location.

    Parameters
    ----------
    output_path : str
        Path (including file name and extension)
        for the output file.

    Returns
    -------
    message : str
        Message indicating location of saved file
        (upon success) or error message (upon failure)/

    Example
    -------
    >>> # Set output path sand save figure
    >>> outpath = os.path.join("04-graphics-outputs", "figure.png")
    >>> save_figure(outpath)
    Saved plot: 04-graphics-outputs\\figure.png
    """
    # Save figure
    try:
        plt.savefig(
            fname=output_path, facecolor="k", dpi=300, bbox_inches="tight"
        )
    except Exception as error:
        message = print(f"Failed to save plot: {error}")
    else:
        message = print(f"Saved plot: {os.path.split(output_path)[-1]}")

    # Return message
    return message


def stack_quality_flags_vnp46a1(vnp46a1_path):
    """Creates a stacked (3D) NumPy array containing all of the VNP46A1
    quality flag bitmask layers.

    Parameters
    ----------
    vnp46a1_path : str
        Path to the VNP46A1 HDF5 (.h5) file.

    Returns
    -------
    quality_flag_stack : numpy array
        3D array containing the quality flag bitmask layers.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Extract QF CLoud Mask and QF DNB bands
    qf_cloud_mask = extract_band_vnp46a1(
        hdf5_path=vnp46a1_path, band_name="QF_Cloud_Mask"
    )
    qf_dnb = extract_band_vnp46a1(hdf5_path=vnp46a1_path, band_name="QF_DNB")

    # Extract QF Cloud Mask bitmasks
    day_night = extract_qa_bits(qf_cloud_mask, 0, 0)
    land_water_background = extract_qa_bits(qf_cloud_mask, 1, 3)
    cloud_mask_quality = extract_qa_bits(qf_cloud_mask, 4, 5)
    cloud_detection = extract_qa_bits(qf_cloud_mask, 6, 7)
    shadow_detected = extract_qa_bits(qf_cloud_mask, 8, 8)
    cirrus_detection = extract_qa_bits(qf_cloud_mask, 9, 9)
    snow_ice_surface = extract_qa_bits(qf_cloud_mask, 10, 10)

    # Create stack
    quality_flag_stack = np.stack(
        arrays=[
            day_night,
            land_water_background,
            cloud_mask_quality,
            cloud_detection,
            shadow_detected,
            cirrus_detection,
            snow_ice_surface,
            qf_dnb,
        ]
    )

    return quality_flag_stack


def stack_quality_flags_vnp46a2(vnp46a2_path):
    """Creates a stacked (3D) NumPy array containing all of the VNP46A2
    quality flag bitmask layers.

    Parameters
    ----------
    vnp46a2_path : str
        Path to the VNP46A2 HDF5 (.h5) file.

    Returns
    -------
    quality_flag_stack : numpy array
        3D array containing the quality flag bitmask layers.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Extract Mandatory Qyality Flag, QF Cloud Mask, and Snow Flag bands
    mandatory_quality_flag = extract_band_vnp46a2(
        hdf5_path=vnp46a2_path, band_name="Mandatory_Quality_Flag"
    )
    qf_cloud_mask = extract_band_vnp46a2(
        hdf5_path=vnp46a2_path, band_name="QF_Cloud_Mask"
    )
    snow_flag = extract_band_vnp46a2(
        hdf5_path=vnp46a2_path, band_name="Snow_Flag"
    )

    # Extract QF Cloud Mask bitmasks
    day_night = extract_qa_bits(qf_cloud_mask, 0, 0)
    land_water_background = extract_qa_bits(qf_cloud_mask, 1, 3)
    cloud_mask_quality = extract_qa_bits(qf_cloud_mask, 4, 5)
    cloud_detection = extract_qa_bits(qf_cloud_mask, 6, 7)
    shadow_detected = extract_qa_bits(qf_cloud_mask, 8, 8)
    cirrus_detection = extract_qa_bits(qf_cloud_mask, 9, 9)
    snow_ice_surface = extract_qa_bits(qf_cloud_mask, 10, 10)

    # Create stack
    quality_flag_stack = np.stack(
        arrays=[
            mandatory_quality_flag,
            snow_flag,
            day_night,
            land_water_background,
            cloud_mask_quality,
            cloud_detection,
            shadow_detected,
            cirrus_detection,
            snow_ice_surface,
        ]
    )

    return quality_flag_stack
