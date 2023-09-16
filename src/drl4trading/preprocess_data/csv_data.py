import pathlib
import string
import json

import pandas as pd
import jdatetime

from src.drl4trading.preprocess_data import add_technical_indicator, fetch_external_data
from src.drl4trading.preprocess_data.config import csvconfig as cfg


class CSVData:
    """Provides methods for retrieving daily stock data
    from csv files

    Attributes
    ----------
        start_date : str
            start date of the data (modified from config.py)
        end_date : str
            end date of the data (modified from config.py)
        ticker_list : list
            a list of stock tickers (modified from config.py)

    Methods
    -------
    fetch_data()
        Fetches data from csv source

    """

    def __init__(
        self,
        start_date: str, 
        end_date: str,
        ticker_list: list = None,
        csv_dirs: list = None,
        baseline_file_name: str = None,
        has_daily_trading_limit: bool = False,
        use_baseline_data: bool = False,
        baseline_filed_mappings = None,
        baseline_date_column_name: str = "date"):
        """
        get price data [and add custom]
        """
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list
        self._csv_dirs = csv_dirs
        self.has_daily_trading_limit = has_daily_trading_limit
        self.use_baseline_data = use_baseline_data
        if self.use_baseline_data:
            self.baseline_df = self._get_baseline(
                file_name=baseline_file_name,
                field_mappings=baseline_filed_mappings,
                baseline_date_column_name=baseline_date_column_name)

    def _get_baseline(self, file_name, field_mappings, baseline_date_column_name) -> pd.DataFrame:
        """
        get tse index as a dataframe
        """

        baseline_fetcher = fetch_external_data.ExternalData(
            self.start_date,
            self.end_date,
        )
        bline_df = baseline_fetcher.fetch_baseline_from_csv(
            file_name=file_name, date_column=baseline_date_column_name,
            field_mappins=field_mappings
        )
        bline_df["tic"] = "BLine"
        return bline_df

    def process_single_tic(self,
                           filenames: list,
                           field_mappings: dict,
                           date_column: str) -> pd.DataFrame:
        """
        process a single ticker and return the dataframe
        """
        data_fetcher = fetch_external_data.ExternalData(
            first_date=self.start_date,
            last_date=self.end_date,
        )
        df = data_fetcher.fetch_from_csv(
            csv_dirs=self._csv_dirs,
            filenames=filenames,
            field_mappins=field_mappings,
            date_column=date_column
        )
        df = df.reindex(self.baseline_df.index)
        df['index_close'] = self.baseline_df['close']
        df['index_volume'] = self.baseline_df['volume']
        df['index_open'] = self.baseline_df['open']
        df["stopped"] = df["open"].isnull()
        if self.has_daily_trading_limit:
            if not "yesterday" in df:
                raise ValueError(
                    f'Market has daily trading limit; \
                        "yesterday" column is required.'
                )
            df["b_queue"] = (df["high"] == df["low"]) & (
                df["low"] > df["yesterday"])
            df["s_queue"] = (df["high"] == df["low"]) & (
                df["high"] < df["yesterday"])
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')
        df = df.reset_index()
        # create day of the week column (monday = 0+2)
        df["day"] = (pd.to_datetime(df["date"]).dt.dayofweek + 2) % 7
        return df

    def fetch_data(
        self, field_mappings, date_column) -> pd.DataFrame:
        """
        get ticker data
        """
        out_dir = cfg.CSV_DIR
        exp_filename = cfg.EXP_FILE_NAME
        path = pathlib.Path.cwd()
        out_dir_full = path / out_dir
        combined_frame = pd.DataFrame()
        for tic in self.ticker_list:
            temp_df = self.process_single_tic(
                ticker=tic,
                field_mappings=field_mappings,
                date_column=date_column
            )
            if not temp_df.empty:
                combined_frame = combined_frame.append(temp_df)

        combined_frame = combined_frame.sort_values(by=["date", "tic"]).reset_index(drop=True)
        out_dir_full.mkdir(parents=True, exist_ok=True)
        full_name = out_dir_full / exp_filename
        combined_frame.to_csv(full_name, index=1, encoding="utf-8")
        return combined_frame

    @staticmethod
    def dl_baseline(index: string = "32097828799138957", base_path: string = "baseline_data/") -> pd.DataFrame:
        """
        Get a TSE index historical data as a dataframe
        """

        index_name = index
        index_code = index
        with open('preprocess_data/config/tse_indexes.json', 'r', encoding="utf8") as json_data:
            tse_indexes = json.load(json_data)
            if index.isnumeric():
                for index_info in tse_indexes:
                    if tse_indexes[index_info]['index'] == index:
                        index_name = index_info
            else:
                index_code = tse_indexes[index]['index']

        url = cfg.TSE_INDEX_DATA_ADDRESS.format(index_code)
        storage_options = {'User-Agent': 'Mozilla/5.0'}
        names = ['j_date', 'index_val']
        index_df = pd.read_csv(url,
                               lineterminator=";",
                               names=names,
                               storage_options=storage_options
                               )
        # index_df.date = pd.to_datetime(df.date, format="%Y%m%d")
        index_df[['j_Y', 'j_M', 'j_D']] = index_df['j_date'].str.split(
            '/', 2, expand=True).astype(int)
        index_df['date'] = index_df.apply(lambda x: jdatetime.date(
            x['j_Y'], x['j_M'], x['j_D'], locale='fa_IR').togregorian(), axis=1)
        # pylint: disable=E1101
        index_df = index_df.drop(columns=['j_Y', 'j_M', 'j_D'])
        index_df.to_csv(
            f'{base_path}/{index_name}.csv',
            index=False
        )
        return index_df
        
