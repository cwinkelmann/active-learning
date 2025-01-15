import PIL

from PIL import Image, ImageDraw, ImageFont


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