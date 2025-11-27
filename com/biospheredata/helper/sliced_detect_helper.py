# arrange an instance segmentation model for test
from pathlib import Path

from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.predict import get_sliced_prediction
from loguru import logger
from com.biospheredata.helper.image.image_coordinates import local_coordinates_to_wgs84


def sliced_prediction(base_path: Path, image_name: str, detection_model: AutoDetectionModel, export_visuals=True):
    """
    @deprecated
    :param image_path:
    :param yolov5_model_path:
    :return:
    """
    image_path = str(base_path.joinpath(image_name))

    logger.info(f"opening image: {image_path}")
    prediction_result = get_sliced_prediction(
        read_image(image_path),
        detection_model,
        slice_height=832,
        slice_width=832,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )
    logger.info(f"done with prediction")
    logger.info(prediction_result.object_prediction_list)
    logger.info(f"iguanas: {len(prediction_result.object_prediction_list)}")

    if export_visuals:
        prediction_result.export_visuals(export_dir=base_path, text_size=1)  ## TODO pass the path or Don't do it at all

    prediction_visual_image = base_path.joinpath(
        f"prediction_visual.png")  ## TODO rename the visual to allow multi user usage
    if len(prediction_result.object_prediction_list) > 0:
        return [obj_pred.to_coco_annotation().json for obj_pred in prediction_result.object_prediction_list], list(
            detection_model.category_names), prediction_visual_image
    else:
        return [], detection_model.category_names, None


# def slice_very_big_raster(base_path, image_name, image_size=-1):
#     """
#     slice the very big geotiff into smaller parts which can be handled by Yolo
#     https://www.youtube.com/watch?v=H5uQ85VXttg
#     :param base_path:
#     :param image_name:
#     :return:
#     """
#
#     sliced_paths = []
#     image_names = []
#
#     dem_path = str(base_path.joinpath(image_name))
#     dem = gdal.Open(dem_path)
#     gt = dem.GetGeoTransform()
#
#     # get coordinates of upper left corner
#     xmin = gt[0]
#     ymax = gt[3]
#     res = gt[1]
#
#     # determine total length of raster
#     xlen = res * dem.RasterXSize
#     ylen = res * dem.RasterYSize
#
#
#     if image_size > 0:
#         xdiv = int(round(dem.RasterXSize / image_size))
#         ydiv = int(round(dem.RasterYSize / image_size))
#         sliced_path = base_path.joinpath(f"sliced_{image_size}px")
#     else:
#         # number of tiles in x and y direction
#         xdiv = 2
#         ydiv = 2
#         sliced_path = base_path.joinpath(f"sliced")
#
#     sliced_path.mkdir(exist_ok=True)
#
#     # size of a single tile
#     xsize = xlen / xdiv
#     ysize = ylen / ydiv
#
#     # create lists of x and y coordinates
#     xsteps = [xmin + xsize * i for i in range(xdiv + 1)]
#     ysteps = [ymax - ysize * i for i in range(ydiv + 1)]
#
#
#
#     # loop over min and max x and y coordinates
#     for i in range(xdiv):
#         for j in range(ydiv):
#             xmin = xsteps[i]
#             xmax = xsteps[i + 1]
#             ymax = ysteps[j]
#             ymin = ysteps[j + 1]
#
#             print("xmin: " + str(xmin))
#             print("xmax: " + str(xmax))
#             print("ymin: " + str(ymin))
#             print("ymax: " + str(ymax))
#             print("\n")
#
#             ## TODO why is this not working?
#             # # use gdal warp
#             # gdal.Warp("dem" + str(i) + str(j) + ".tif", dem,
#             #           outputBounds=(xmin, ymin, xmax, ymax), dstNodata=-9999)
#             # or gdal translate to subset the input raster
#
#             sliced_image_path = sliced_path.joinpath(f"dem_translate_{i}_{j}.tif")
#             sliced_jpg_image_path = sliced_path.joinpath(f"dem_translate_{i}_{j}.jpg")
#             image_name = f"dem_translate_{i}_{j}.tif"
#
#             if not os.path.exists(str(sliced_image_path)):
#                 return_value = gdal.Translate(str(sliced_image_path), dem, projWin=(xmin, ymax, xmax, ymin),
#                                               xRes=res,
#                                               yRes=-res)
#
#             ## For image recognition this is necesarry ## TODO there should be a more elegant ways
#             import rasterio.shutil
#             print(f"odm_orthophoto.tif to jpg")
#
#             if not os.path.exists(str(sliced_jpg_image_path)):
#                 rasterio.shutil.copy(
#                     str(sliced_image_path),
#                     str(sliced_jpg_image_path),
#                     driver='JPEG')
#
#             sliced_paths.append(sliced_path)
#             image_names.append(image_name)
#
#     # close the open dataset!!!
#     dem = None
#     return sliced_path, image_names


def coco_annotation_list_geopandas(predictions: dict, sliced_image_path: Path, detection_model_category_names):
    """
    get the original georeferenced image and project the pixel coordinates to

    :param detection_model_category_names:
    :param predictions:
    :param sliced_image_path:
    :return:
    """
    local_coordinates_paths = []
    for image_name, annotation in predictions.items():

        temp_path = local_coordinates_to_wgs84(image_name,
                                               sliced_image_path,
                                               annotation,
                                               detection_model_category_names)
        if temp_path is not None:
            local_coordinates_paths.append(temp_path)

    return local_coordinates_paths




if __name__ == '__main__':
    base_path = Path("/home/christian/Downloads/")
    image_name = "DJI_0023.JPG"
    # sliced_prediction(base_path=base_path, image_name=image_name, yolov5_model_path="/home/christian/Downloads/6a8a89a0ed19acc1e99273c6585842af.pt")
    path, y = sliced_prediction(base_path=base_path, image_name=image_name,
                                yolov5_model_path="./weights/e3a137ee2ea035a7d462a34715c1d52a.pt")

    print(f"path: {path}")
    print(f"y: {y}")
