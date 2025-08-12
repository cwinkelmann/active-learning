

def get_image_id(filename):
    """
    @ has moved to flight_image_capturing_sim/helper/image.py
    generate an id from the image itself which can be used to find images which are exactly the same
    @param filename:
    @return:
    """

    import hashlib
    with open(filename, "rb") as f:
        bytes = f.read()  # read entire file as bytes
        readable_hash = hashlib.sha256(bytes).hexdigest()
    return readable_hash


def get_mosaic_slice_name(mission_name, creation_date):
    return f"{mission_name}_{creation_date}_m"