import sys
import os
sys.path.insert(0, './')
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{current_dir}/third_party_models/Depth-Anything')

import argparse
import logging
import warnings
from types import SimpleNamespace
from pipeline.pipeline import BackgroundRemovalPipeline
from util.experiment_utils import setup_logger, load_config
from util.inference import run_inference

sys.path.insert(0, './')

warnings.filterwarnings("ignore")  # Suppress all warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run segmentation pipeline.')
    parser.add_argument(
        '--config-dir', type=str, default='./config/config.yaml',
        help='Path to the configuration file.'
    )
    parser.add_argument(
        '--input-img-dir', type=str, default='./data/input_images',
        help='Path to the input image.'
    )
    parser.add_argument(
        '--output-img-dir', type=str, default='./data/output_images',
        help='Path to save the output image.'
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    config = load_config(args.config_dir)
    config = SimpleNamespace(**config)

    setup_logger(config.log_dir)

    # Ensure the all directories exists
    os.makedirs(args.output_img_dir, exist_ok=True)

    logging.info('Starting background removal pipeline...')
    logging.info(f'Configuration file: {args.config_dir}')
    logging.info(f'Input image directory: {args.input_img_dir}')
    logging.info(f'Output image directory: {args.output_img_dir}')

    logging.info('\nInstantiating the background removal pipeline object...')
    model = BackgroundRemovalPipeline(config)

    run_inference(
        model,
        args.input_img_dir,
        args.output_img_dir,
    )

    logging.info('\nBackground removal pipeline completed successfully.')


if __name__ == '__main__':
    args = parse_args()
    main(args)
