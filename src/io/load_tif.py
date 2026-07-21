import rasterio


def load_tif(path):
    """
    Load a GeoTIFF image.

    Returns
    -------
    image : numpy.ndarray
        RGB image (H,W,3)

    profile : rasterio profile
        GeoTIFF metadata
    """

    with rasterio.open(path) as src:

        image = src.read([1, 2, 3])

        profile = src.profile

    image = image.transpose(1, 2, 0)

    return image, profile