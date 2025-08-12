from loguru import logger


def xmp_metadata(img_path):
    """
    retrieve XMP Metadata
    @deprecated there is a better version in the flight_image_capturing_sim repo
    :param img_path:
    :return:
    """
    metadata = {}

    try:
        from libxmp import XMPFiles, consts
        xmpfile = XMPFiles(file_path=str(img_path), open_forupdate=True)
        xmp = xmpfile.get_xmp()
        metadata["format"] = xmp.get_property(consts.XMP_NS_DC, 'format')

        for xmp_key in [
            "drone-dji:GpsLatitude", "drone-dji:GpsLongitude",
            "drone-dji:GimbalYawDegree", "drone-dji:GimbalRollDegree",
            "drone-dji:GimbalPitchDegree",  # the pitch is the inclination with -90 == NADIR and 0 is horizontal
            "drone-dji:AbsoluteAltitude", "drone-dji:RelativeAltitude",
            "drone-dji:FlightRollDegree", "drone-dji:FlightYawDegree", "drone-dji:FlightPitchDegree"
        ]:
            metadata[xmp_key] = float(xmp.get_property("http://www.dji.com/drone-dji/1.0/", xmp_key))

    except Exception as e:
        logger.error(
            f"Problems with XMP library. Check https://python-xmp-toolkit.readthedocs.io/en/latest/installation.html, {e}")

    return metadata