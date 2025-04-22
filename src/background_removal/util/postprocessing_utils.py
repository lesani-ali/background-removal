import numpy as np
from PIL import Image


def add_missed_info(
    depth_map: np.ndarray,
    saliency_img: Image.Image,
    image: Image.Image,
    threshold: int = 140
) -> np.ndarray:
    """
    Add missed information from the original image to the saliency image based on the depth map.

    :param depth_map: Depth map of the image.
    :param saliency_img: Saliency image.
    :param image: Original image.
    :param threshold: Depth threshold to determine recoverable information.
    :return: Updated saliency image with recovered information.
    """
    saliency_img = np.array(saliency_img)
    image = np.asarray(image)

    mask = depth_map > threshold

    saliency_img[mask] = image[mask]

    return saliency_img


def remove_farther_objects(
    depth_map: np.ndarray,
    saliency_img: Image.Image,
    threshold: int = 70
) -> np.ndarray:
    """
    Remove information from the original image based on the depth map.

    :param depth_map: Depth map of the image.
    :param saliency_img: Saliency image.
    :param threshold: Depth threshold to determine negligible information.
    :return: Updated saliency image with recovered information.
    """

    saliency_img = np.array(saliency_img)

    mask = depth_map < threshold

    saliency_img[mask] = 255

    return saliency_img