"""
Data processing script for NILM datasets.

Usage:
    python data.py --dataset plegma --appliance boiler
    python data.py --dataset refit --appliance dishwasher
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from src_pytorch.config import (
    get_appliance_params,
    get_dataset_config,
    get_dataset_split
)

from dataset_management.plegma.plegma_parser import PlegmaParser
from dataset_management.refit.refit_parser import RefitParser


def process_data(dataset_name: str, appliance_name: str, raw_path: str = './') -> None:
    """
    Main function to orchestrate the data processing pipeline.

    Args:
        dataset_name: 'refit' or 'plegma'
        appliance_name: Name of appliance to process
        raw_path: Base path for raw data (default: current directory)
    """
    print(f"Starting data processing pipeline...")
    print(f"Dataset: {dataset_name}")
    print(f"Appliance: {appliance_name}")

    # Get config from Python config file
    dataset_config = get_dataset_config(dataset_name)
    appliance_params = get_appliance_params(dataset_name, appliance_name)
    split = get_dataset_split(dataset_name, appliance_name)

    # Create output directory
    output_dir = Path(raw_path) / 'processed' / dataset_name / appliance_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processed data will be saved to: {output_dir}")

    # Get data location
    data_location = Path(raw_path) / dataset_config['location']

    # Select and instantiate the correct parser
    if dataset_name == 'plegma':
        parser = PlegmaParser(
            appliance=appliance_name,
            data_location=data_location,
            output_dir=output_dir,
            split=split,
            cutoff=appliance_params['cutoff'],
            aggregate_cutoff=dataset_config['aggregate_cutoff'],
            sampling_rate=dataset_config['sampling']
        )
    elif dataset_name == 'refit':
        parser = RefitParser(
            appliance=appliance_name,
            data_location=data_location,
            output_dir=output_dir,
            split=split,
            cutoff=appliance_params['cutoff'],
            aggregate_cutoff=dataset_config['aggregate_cutoff'],
            sampling_rate=dataset_config['sampling']
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    parser.process()
    print("Data processing complete.")


def main():
    parser = argparse.ArgumentParser(description='Process NILM datasets')
    parser.add_argument('--dataset', '-d', type=str, required=True,
                        choices=['refit', 'plegma'],
                        help='Dataset to process')
    parser.add_argument('--appliance', '-a', type=str, required=True,
                        help='Appliance to extract')
    parser.add_argument('--raw-path', '-r', type=str, default='./',
                        help='Base path for raw data (default: ./)')

    args = parser.parse_args()
    process_data(args.dataset, args.appliance, args.raw_path)


if __name__ == "__main__":
    main()
