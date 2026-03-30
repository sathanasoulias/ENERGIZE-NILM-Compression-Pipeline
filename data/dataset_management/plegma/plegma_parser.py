import pandas as pd
from pathlib import Path
import numpy as np
import glob
import os
from typing import List, Dict, Optional


class PlegmaParser:
    """
    Parses the PLEGMA dataset.

    This class handles loading raw data for specified houses, merging daily CSV files,
    resampling the data to a consistent frequency, and saving the processed
    train, validation, and test splits to disk.
    """
    def __init__(
        self,
        appliance: str,
        data_location: Path,
        output_dir: Path,
        split: Dict[str, List[int]],
        cutoff: int,
        aggregate_cutoff: int = 10000,
        sampling_rate: str = '10s',
        training_file: str = 'training_.csv',
        validation_file: str = 'validation_.csv',
        test_file: str = 'test_.csv'
    ):
        """
        Initializes the PlegmaParser.

        Args:
            appliance: Name of the appliance to extract
            data_location: Path to raw PLEGMA dataset
            output_dir: Directory where processed data files will be saved
            split: Dictionary with 'train', 'val', 'test' house lists
            cutoff: Maximum power value for the appliance (W)
            aggregate_cutoff: Maximum power value for aggregate (W)
            sampling_rate: Resampling frequency (default '10s')
            training_file: Name of training output file
            validation_file: Name of validation output file
            test_file: Name of test output file
        """
        self.appliance = appliance
        self.data_location = Path(data_location)
        self.output_dir = Path(output_dir)
        self.split = split
        self.cutoff = cutoff
        self.aggregate_cutoff = aggregate_cutoff
        self.sampling_rate = sampling_rate
        self.training_file = training_file
        self.validation_file = validation_file
        self.test_file = test_file

        print(f"Data location: {self.data_location}")
        print(f"Data location exists: {self.data_location.exists()}")

    def _merge_house_files(self, house_id: int) -> pd.DataFrame | None:
        """
        Merges all daily electrical data CSVs for a single house into one DataFrame.
        """
        # Handle both House_1 and House_01 naming conventions
        electric_path = self.data_location / f'House_{house_id:02d}/Electric_data'
        print(f"  Trying path: {electric_path} (exists: {electric_path.exists()})")
        if not electric_path.exists():
            electric_path = self.data_location / f'House_{house_id}/Electric_data'
            print(f"  Trying path: {electric_path} (exists: {electric_path.exists()})")
            if not electric_path.exists():
                print(f"Warning: Directory not found for House_{house_id}. Skipping.")
                return None

        # Find all CSV files, excluding metadata files
        csv_files = sorted(glob.glob(os.path.join(electric_path, "*.csv")))
        valid_files = [f for f in csv_files if "metadata" not in os.path.basename(f)]
        if not valid_files:
            print(f"Warning: No valid data files found for House_{house_id}. Skipping.")
            return None

        # Concatenate all valid files into a single DataFrame
        df = pd.concat((pd.read_csv(f) for f in valid_files), ignore_index=True)
        return df

    def _load_house_data(self, house_id: int) -> pd.DataFrame | None:
        """
        Loads, preprocesses, and resamples data for a single house.
        """
        df = self._merge_house_files(house_id)
        if df is None or self.appliance not in df.columns:
            print(f"Warning: Appliance '{self.appliance}' not found in House_{house_id}. Skipping.")
            return None

        # Select relevant columns and handle timestamps
        df = df[['timestamp', 'P_agg', self.appliance, 'issues']]
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

        # Remove data points with known issues
        df = df.drop(index=df[df['issues'] == 1].index, axis=0)
        df = df.rename(columns={'P_agg': 'aggregate'})
        df = df[['aggregate', self.appliance]]

        # Resample to a consistent frequency and forward-fill missing values
        df = df.resample(self.sampling_rate).mean().fillna(method='ffill', limit=30)
        return df.dropna().copy()

    def process(self):
        """
        The main processing pipeline for the PLEGMA dataset.
        """
        # --- Process Training Data ---
        train_dfs = [self._load_house_data(h) for h in self.split['train']]
        df_train = pd.concat([df for df in train_dfs if df is not None])
        df_train = self._clean_and_clip(df_train)

        # Calculate normalization stats ONLY from the training data
        agg_mean = df_train['aggregate'].mean()
        agg_std = df_train['aggregate'].std()
        print(f"Calculated training stats for '{self.appliance}': Mean={agg_mean:.8f}, Std={agg_std:.8f}")

        # Normalize aggregate and scale appliance data, then save
        df_train['aggregate'] = (df_train['aggregate'] - agg_mean) / agg_std
        df_train[self.appliance] /= self.cutoff
        df_train.to_csv(self.output_dir / self.training_file, index=False)
        print(f"Saved training data to {self.output_dir / self.training_file}")

        # --- Process Validation and Test Data ---
        for split_name, houses, filename in [
            ('validation', self.split['val'], self.validation_file),
            ('test', self.split['test'], self.test_file)
        ]:
            split_dfs = [self._load_house_data(h) for h in houses]
            if not split_dfs or all(df is None for df in split_dfs):
                print(f"Warning: No data found for {split_name} split. Skipping.")
                continue

            df_split = pd.concat([df for df in split_dfs if df is not None])
            df_split = self._clean_and_clip(df_split)

            # Normalize using the training set's mean and std
            df_split['aggregate'] = (df_split['aggregate'] - agg_mean) / agg_std
            df_split[self.appliance] /= self.cutoff

            # Save the processed split
            df_split.to_csv(self.output_dir / filename, index=False)
            print(f"Saved {split_name} data to {self.output_dir / filename}")

    def _clean_and_clip(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies cleaning and clipping to the power data.
        """
        df = df[df['aggregate'] > 0]
        df[df < 5] = 0  # Set values below 5W to 0 to reduce noise
        df['aggregate'] = df['aggregate'].clip(upper=self.aggregate_cutoff)
        df[self.appliance] = df[self.appliance].clip(upper=self.cutoff)
        return df
