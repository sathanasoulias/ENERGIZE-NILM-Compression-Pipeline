import pandas as pd
from pathlib import Path
import numpy as np
from typing import List, Dict


class RefitParser:
    """
    Parses the REFIT dataset.
    """
    def __init__(
        self,
        appliance: str,
        data_location: Path,
        output_dir: Path,
        split: Dict[str, List[int]],
        cutoff: int,
        aggregate_cutoff: int = 10000,
        sampling_rate: str = '8s',
        training_file: str = 'training_.csv',
        validation_file: str = 'validation_.csv',
        test_file: str = 'test_.csv'
    ):
        """
        Initializes the REFIT parser.

        Args:
            appliance: Name of the appliance to extract
            data_location: Path to raw REFIT dataset (containing Data/ and Labels/ folders)
            output_dir: Directory where processed data files will be saved
            split: Dictionary with 'train', 'val', 'test' house lists
            cutoff: Maximum power value for the appliance (W)
            aggregate_cutoff: Maximum power value for aggregate (W)
            sampling_rate: Resampling frequency (default '8s')
            training_file: Name of training output file
            validation_file: Name of validation output file
            test_file: Name of test output file
        """
        self.appliance = appliance
        self.output_dir = Path(output_dir)
        self.split = split
        self.cutoff = cutoff
        self.aggregate_cutoff = aggregate_cutoff
        self.sampling_rate = sampling_rate
        self.training_file = training_file
        self.validation_file = validation_file
        self.test_file = test_file

        self.data_location = Path(data_location) / 'Data'
        self.labels_location = Path(data_location) / 'Labels'

    def _load_house_data(self, house_idx: int) -> pd.DataFrame | None:
        """Loads and preprocesses data for a single house."""
        house_data_loc = self.data_location / f'CLEAN_House{house_idx}.csv'
        label_loc = self.labels_location / f'House{house_idx}.txt'

        if not (house_data_loc.exists() and label_loc.exists()):
            print(f"Warning: Data not found for House {house_idx}. Skipping.")
            return None

        with open(label_loc) as f:
            house_labels = ['Time', 'Unix'] + f.readline().strip().split(',')

        if self.appliance not in house_labels:
            print(f"Warning: Appliance '{self.appliance}' not found in House {house_idx}. Skipping.")
            return None

        appliance_col_index = house_labels.index(self.appliance)
        issues_col_index = house_labels.index('issues')

        df = pd.read_csv(house_data_loc, usecols=[0, 1, 2, appliance_col_index, issues_col_index], header=0)
        df.columns = ['Time', 'Unix', 'Aggregate', self.appliance, 'Issues']
        df = df.rename(columns={'Aggregate': 'aggregate', 'Issues': 'issues'})

        df['Unix'] = pd.to_datetime(df['Unix'], unit='s')
        df = df.set_index('Unix')
        df = df.drop(columns=['Time'])

        idx_to_drop = df[df['issues'] == 1].index
        df = df.drop(index=idx_to_drop, axis=0)

        df = df.resample(self.sampling_rate).mean().fillna(method='ffill', limit=30)
        return df.dropna().copy()

    def process(self):
        """
        Main processing function for the REFIT dataset.
        """
        # Process Training Data
        train_dfs = [self._load_house_data(h) for h in self.split['train']]
        df_train = pd.concat([df for df in train_dfs if df is not None])
        df_train = self._clean_and_clip(df_train)

        agg_mean = df_train['aggregate'].mean()
        agg_std = df_train['aggregate'].std()
        print(f"Calculated training stats for '{self.appliance}': Mean={agg_mean:.8f}, Std={agg_std:.8f}")

        df_train['aggregate'] = (df_train['aggregate'] - agg_mean) / agg_std
        df_train[self.appliance] /= self.cutoff
        df_train[['aggregate', self.appliance]].to_csv(self.output_dir / self.training_file, index=False)
        print(f"Saved training data to {self.output_dir / self.training_file}")

        # Process Validation and Test Data
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

            df_split['aggregate'] = (df_split['aggregate'] - agg_mean) / agg_std
            df_split[self.appliance] /= self.cutoff

            df_split[['aggregate', self.appliance]].to_csv(self.output_dir / filename, index=False)
            print(f"Saved {split_name} data to {self.output_dir / filename}")

    def _clean_and_clip(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies cleaning and clipping to the dataframe."""
        df = df[df['aggregate'] > 0]
        df[df < 5] = 0

        df['aggregate'] = df['aggregate'].clip(upper=self.aggregate_cutoff)
        df[self.appliance] = df[self.appliance].clip(upper=self.cutoff)
        return df
