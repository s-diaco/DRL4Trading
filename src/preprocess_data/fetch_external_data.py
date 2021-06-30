"""
Get csv data from [multiple] dirs
"""
import logging
import pathlib
from halo.halo import Halo

import pandas as pd


class ExternalData:
    """Tools to get data from third party sources"""

    def __init__(
        self,
        first_date: str,
        last_date: str,
    ):
        """
        Initialize class variables.

        Args:
                first_date (str): Strat of data (%Y-%m-%d)
                last_date (str): End of the data (%Y-%m-%d)
        """
        self.first_day = first_date
        self.last_day = last_date

    def fetch_from_csv(
        self, csv_dirs: list,
        ticker: str,
        field_mappins: list = None,
        date_column: str = "date"
    ) -> pd.DataFrame:
        """
        Fetch data from csv files

        Args:
                csv_dirs (list): Path to CSV dirs
                ticker (str): ticker name
                field_mappins (list): Mappings to rename csv field names
                date_column (str): Name of the date column in csv file

        Returns:
                pd.DataFrame: Data from csv file
        """
        with Halo(text='Gettng dirs', spinner='arrow3') as halo_log:
            csv_dfs = []
            halo_log.text=f'Fetching data for {ticker}'
            for csv_dir in csv_dirs:
                halo_log.text=f'Fetching data for {ticker} -> checking: {csv_dir}'
                price_file_name = f'{ticker}.csv'
                try:
                    csv_df = self._get_single_csv(
                        file_name=csv_dir/pathlib.Path(price_file_name),
                        date_column=date_column,
                        field_mappins=field_mappins
                    )
                except Exception as e:
                    logging.error(e)
                if not csv_df.empty:
                    csv_dfs.append(csv_df)
        if csv_dfs:
            concat_df = pd.concat(csv_dfs)
        else:
            raise ValueError(f'No csv data found for {ticker}')
        return concat_df

    def _get_single_csv(
        self,
        file_name: pathlib.Path,
        date_column: str = "date",
        field_mappins: list = None
    ) -> pd.DataFrame:
        """
        Fetch data from a csv file

        Args:
                file_name (pathlib.Path): Path to CSV file
                date_column (str): Name of the date column in csv file
                field_mappins (list): Mappings to rename csv field names

        Returns:
                pd.DataFrame: Data from csv file
        """
        csv_df = pd.read_csv(
            file_name,
            index_col=date_column,
            parse_dates=[date_column],
            header=0,
            date_parser=lambda x: pd.to_datetime(x, format="%Y-%m-%d"),
        )
        csv_df = csv_df.loc[self.first_day:self.last_day]
        if field_mappins:
            csv_df = csv_df.rename(columns=field_mappins)
        csv_df.index.names = ['date']
        return csv_df

    def fetch_baseline_from_csv(
        self, file_name: str,
        date_column: str = "date",
        field_mappins: list = None
    ) -> pd.DataFrame:
        """
        Fetch baseline data from csv file

        Args:
                file_name (pathlib.Path): Path to baseline file
                date_column (str): Name of the date column
                field_mappins (list): Mappings to rename field names

        Returns:
                pd.DataFrame: Baseline data from csv file
        """
        logging.info(f"fetching baseline {file_name}.")
        baseline_full_path = pathlib.Path(file_name)
        bl_df = self._get_single_csv(
            baseline_full_path,
            date_column,
            field_mappins
        )
        if bl_df.empty:
            raise ValueError(f'Can not load baseline data from "{file_name}"')
        return bl_df
