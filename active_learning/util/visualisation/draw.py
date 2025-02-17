from pathlib import Path

import PIL

from PIL import Image, ImageDraw, ImageFont

from active_learning.util.Annotation import project_point_to_crop
from active_learning.util.image_manipulation import create_box_around, crop_out_images_v3


def draw_text(
    image: PIL.Image.Image,
    text: str,
    position: tuple,
    font_size: int = 20,
    )-> PIL.Image.Image:

    draw = ImageDraw.Draw(image)
    # see https://github.com/Alexandre-Delplanque/HerdNet/pull/8/files
    try:
        font = ImageFont.truetype("segoeui.ttf", size=font_size)
    except IOError:
        font = ImageFont.load_default()

    l, t, r, b = draw.textbbox(position, text)
    draw.rectangle((l-5, t-5, r+5, b+5), fill='white')
    draw.text(position, text, font=font, fill='black')

    return image


def draw_thumbnail(df, i, suffix, images_path, box_size):
    ts_path = images_path.parent / f"thumbnails_{suffix}"
    ts_path.mkdir(exist_ok=True)

    if len(df) > 0:
        box_polygons = [create_box_around(point, box_size, box_size) for point in df.geometry]
        df_fp_list = df.to_dict(orient="records")
        crops = crop_out_images_v3(image=PIL.Image.open(i), rasters=box_polygons)
        # TODO add every point to theses boxes, some might not be in the center.
        projected_points = [project_point_to_crop(point, crop_box)
                            for point, crop_box in zip(df.geometry, box_polygons)]

        for idx, (crop, point) in enumerate(zip(crops, projected_points)):
            crop = draw_text(crop, f"{df_fp_list[idx].get('species', '')} | {df_fp_list[idx].get('scores')}%",
                             position=(10, 5), font_size=int(0.08 * box_size))
            crop.save(ts_path / f"{Path(i.name).stem}_{suffix}_{idx}.JPG")
            # visualise_image(image=crop, show=True)