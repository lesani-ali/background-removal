from typing import Dict, Any
from PIL import Image
from background_removal.models.saliency import SaliencyDetectionModel
from background_removal.models.depth import DepthModel
from util.postprocessing_utils import (
    add_missed_info,
    remove_farther_objects
)


class BackgroundRemovalPipeline(object):
    def __init__(self, config: Dict[str, Any]):
        self.saliency_model = SaliencyDetectionModel(
            config.models['saliency']
        )

        self.depth_model = DepthModel(
            config.models['depth']
        )

        self.config = config

    def __call__(self, image: Image.Image) -> Image.Image:
        return self.predict(image)

    def predict(self, image: Image.Image) -> Image.Image:

        # Step 1: Saliency Detection
        saliency_img = self.saliency_model(image)

        # Step 2: Depth Estimation
        depth_map = self.depth_model(image)

        # Step 3: Add missed information
        no_background_img = add_missed_info(
            depth_map, saliency_img, image,
            self.config.recover_info_threshold
        )

        # Step 4: Remove further objects
        no_background_img = remove_farther_objects(
            depth_map, no_background_img,
            self.config.ignore_info_threshold
        )

        return no_background_img
