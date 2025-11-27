"""
Create a image database from a folder of images

expects a folder structure like
Iguanas_From_Above/
└── 2020_2021_2022_2023_2024/
    ├── Island_A/
    │   ├── BT01_11012023/
    │   │   ├── Island_A_BT01_DJI_0001_11012023.JPG
    │   │   ├── Island_A_BT01_DJI_0002_11012023.JPG
    │   │   └── Island_A_BT01_DJI_0003_11012023.JPG
    │   ├── BT02_15012023/
    │   │   ├── Island_A_BT02_DJI_0001_15012023.JPG
    │   │   └── Island_A_BT02_DJI_0002_15012023.JPG
    │   └── BT03_18012023/
    │       └── Island_A_BT03_DJI_0001_18012023.JPG
    ├── Island_B/
    │   ├── CR01_05022023/
    │   │   ├── Island_B_CR01_DJI_0001_05022023.JPG
    │   │   └── Island_B_CR01_DJI_0002_05022023.JPG
    │   └── CR02_08022023/
    │       └── Island_B_CR02_DJI_0001_08022023.JPG
    └── Island_C/
        ├── FL01_28012023/
        │   ├── Island_C_FL01_DJI_0001_28012023.JPG
        │   ├── Island_C_FL01_DJI_0002_28012023.JPG
        │   └── Island_C_FL01_DJI_0003_28012023.JPG
        └── FL02_02022023/
            └── Island_C_FL02_DJI_0001_02022023.JPG

"""
import numpy as np
from loguru import logger
from pathlib import Path


from active_learning.database import images_data_extraction, create_image_db
from active_learning.types.image_metadata import ExposureMode, ExposureProgram, CompositeMetaData, \
    convert_to_serialisable_dataframe
from active_learning.types.image_metadata import list_images
import geopandas as gpd

from active_learning.database import derive_image_metadata


def reorder_columns(df):
    reordered_columns = [
        # Dimensional/Categorical (Primary grouping variables)
        'expedition_phase',
        'YYYYMMDD',
        'year_month',
        'island',
        'island_code',
        'site_code',
        'flight_code',
        'mission_folder',
        'drone_name',


        # Datetime
        'datetime_digitized',
        'datetime',
        'datetime_original',

        # Core Image Identification
        'image_name',
        'folder_name',
        'filepath',
        'image_hash',

        # Location/GPS
        'latitude',
        'longitude',
        'GpsLatitude',
        'GpsLongitude',
        'geometry',
        'gps_latitude',
        'gps_latitude_ref',
        'gps_longitude',
        'gps_longitude_ref',
        'gps_altitude',
        'gps_altitude_ref',

        # Flight/Drone Position & Movement
        'AbsoluteAltitude',
        'RelativeAltitude',
        'GimbalYawDegree',
        'GimbalRollDegree',
        'GimbalPitchDegree',
        'FlightRollDegree',
        'FlightYawDegree',
        'FlightPitchDegree',
        'FlightXSpeed', # Matrice
        'FlightYSpeed',
        'FlightZSpeed',
        'bearing_to_prev',
        'flight_direction',

        # Image Dimensions & Coverage
        'image_height',
        'image_width',
        'pixel_x_dimension',
        'pixel_y_dimension',
        'ground_width_m',
        'ground_height_m',

        # Ground Sampling Distance
        'gsd_abs_width_cm',
        'gsd_abs_height_cm',
        'gsd_abs_avg_cm',
        'gsd_rel_width_cm',
        'gsd_rel_height_cm',
        'gsd_rel_avg_cm',

        # Analysis/Quality Metrics
        'distance_to_prev',
        'time_diff_seconds',
        'speed_m_per_s',
        'risk_score',
        'shift_mm',
        'shift_pixels',
        'forward_overlap_pct',
        'overlap_distance',
        'angle_diff',
        'flight_mode',
        'relevant_dimension',

        'is_oblique',
        'is_nadir',

        # Camera Settings (Most Important)
        'make',
        'model',
        'exposure_time',
        'f_number',
        'focal_length',
        'focal_length_in_35mm_film',
        'photographic_sensitivity',

        # Camera Settings (Extended)
        'exposure_mode',
        'exposure_program',
        'exposure_bias_value',
        'metering_mode',
        'white_balance',
        'digital_zoom_ratio',
        'max_aperture_value',
        'subject_distance',
        'lens_specification',
        'light_source',

        # Technical Image Properties
        'bits_per_sample',
        'color_space',
        'compression',
        'contrast',
        'saturation',
        'scene_capture_type',
        'sharpness',
        'orientation',
        'resolution_unit',
        'samples_per_pixel',
        'x_resolution',
        'y_resolution',
        'y_and_c_positioning',
        'height',

        # Equipment/System Info
        'body_serial_number',
        'software',
        'exif_version',
        'gps_version_id',
        'gain_control',

        # JPEG Technical
        'jpeg_interchange_format',
        'jpeg_interchange_format_length',
        'image_description',
        'xp_keywords',
        
        # TODO this is a special case for a Matrics 4e drone
        # Specialized Equipment (LRF - Laser Range Finder)
        'LRFTargetDistance',
        'LRFTargetLat',
        'LRFTargetLon',
        'LRFTargetAbsAlt',
        'LRFTargetAlt',
        'SensorTemperature',
        'LensTemperature',
        'ImageSource',
        'ProductName'
    ]
    missing_columns = set(reordered_columns) - set(df.columns)
    extra_columns = set(df.columns) - set(reordered_columns)
    # delete the missing columns from the list
    reordered_columns = [col for col in reordered_columns if col in df.columns]
    # To reorder your DataFrame:
    df_reordered = df[reordered_columns]

    return df_reordered

if __name__ == "__main__":
    island_folder_path = Path("/Volumes/G-DRIVE/Iguanas_From_Above/2020_2021_2022_2023_2024")

    # get_analysis_ready_image_metadata = main(base_folder=island_folder_path, local_epsg="25833")
    database_path = '/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/submission/Flight Database Statistics'

    current_date = "2025_11_11"
    updated_database_path = f'/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/submission/Flight Database Statistics/2020_2021_2022_2023_2024_database_analysis_ready_{current_date}.parquet'
    # gdf_db = gpd.read_parquet(database_path)

    # setting this to None will avoid re-processing existing images
    island_folder_path = Path('/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/submission/Flight Database Statistics/Mavic 2 Pro/flight_6')
    island_folder_path = Path('/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/submission/Flight Database Statistics/Matrice 4e/Rerik')
    gdf_db = None
    gdf_analysis_ready_image_metadata = create_image_db(images_path=island_folder_path,
                                                        local_epsg="32715",
                                                        gdf_preexisting_database=gdf_db)


    #

    # reorder the columns
    gdf_analysis_ready_image_metadata = reorder_columns(gdf_analysis_ready_image_metadata)

    # remove infitiy points
    gdf_analysis_ready_image_metadata = gdf_analysis_ready_image_metadata[
        ~gdf_analysis_ready_image_metadata.geometry.apply(lambda geom: np.isinf(geom.x) or np.isinf(geom.y))
    ]
    # remove images which don't belong in there manually

    gdf_analysis_ready_image_metadata = convert_to_serialisable_dataframe(gdf_analysis_ready_image_metadata)
    gdf_analysis_ready_image_metadata.to_parquet(updated_database_path)
    # save as geojson
    geojson_path = str(updated_database_path).replace('.parquet', '.geojson')
    gdf_analysis_ready_image_metadata.to_file(geojson_path, driver='GeoJSON')

    logger.info(f"Image metadata saved to {updated_database_path}")
