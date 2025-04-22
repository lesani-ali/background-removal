# Introduction

This project provides a tool for background removal by combining saliency masking and depth information.
We use two third-party models: (1) [Transparent Background](https://github.com/plemeri/transparent-background) for generating saliency masks and (2) [Depth Anything](https://github.com/LiheYoung/Depth-Anything) for estimating the relative depth of the scene from a single image. The depth information helps the Transparent Background model to recover missed regions of the foreground, and remove farther objects that are likely part of the background. This improves background removal accuracy, especially in challenging scenes where the saliency mask alone may not be sufficient.

## Third party models used

- **[Transparent Background](https://github.com/plemeri/transparent-background)**: Generates a saliency mask to separate foreground from background.
- **[Depth Anything](https://github.com/LiheYoung/Depth-Anything)**: Predicts relative depth from a monocular image to assist in identifying background objects.

## Installation

1. Create environment:
    ```bash
    conda create --name background_removal python=3.8
    conda activate background_removal
    ```
2. Install background removal:
    ```bash
    git clone git@github.com:lesani-ali/background-removal.git && cd background-removal
    ```
    **Install GPU version:**

        ```bash
        pip install --extra-index-url https://download.pytorch.org/whl/cu118 -e .
        ```

    **Install CPU version:**

        ```bash
        pip install -e .
        ```


## Usage

2. Run the tool:
    ```bash
    python main.py --input-img-dir "/path/to/input/images" --output-img-dir "/path/to/output/images"" 
    
    ```

## Note
If you are not using a GPU, please set the device type to CPU in the configuration file located at: `config/config.yaml`.