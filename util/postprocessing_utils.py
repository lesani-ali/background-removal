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
    boxes: np.ndarray,
    threshold: int
) -> np.ndarray:
    """
    Remove objects that are farther away based on the depth map.

    :param depth_map: Depth map of the image.
    :param image: Image to remove objects from.
    :param threshold: Depth threshold to determine objects to remove.
    :return: Image with farther objects removed.
    """
    keep_row = np.ones(len(boxes), dtype=bool)
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Crop region
        roi = depth_map[y1:y2, x1:x2]

        if np.mean(roi) < threshold:
            keep_row[idx] = False

    return boxes[keep_row]