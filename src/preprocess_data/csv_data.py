import pathlib
import pandas as pd
import get_tse_data.tse_config.tse_config as cfg
from . import fetch_external_data


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
        ticker_list: list,
        csv_dirs: list,
        baseline_file_name: str = None,
        baseline_dir: str = None,
        date_column_name: str = "date",
        has_daily_trading_limit: bool = False,
        use_baseline_data: bool = False):
        """
        get price data [and add custom data to it]
        """
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list
        self._date_column = date_column_name
        self._csv_dirs = csv_dirs
        self.has_daily_trading_limit = has_daily_trading_limit
        self.use_baseline_data = use_baseline_data
        self.baseline_df = self._get_baseline(baseline_file_name, baseline_dir)

    def _get_baseline(self, file_name, csv_dir) -> pd.DataFrame:
        """
        get tse index as a dataframe
        """

        baseline_fetcher = fetch_external_data.ExternalData(
            self.start_date,
            self.end_date
        )
        bline_df = baseline_fetcher.fetch_baseline_from_csv(
            csv_dir,
            file_name
        )
        bline_df["tic"] = "BLine"
        return bline_df

    def process_single_tic(self,
                           ticker,
                           field_mappings
    ) -> pd.DataFrame:
        """
        process a single ticker and return the dataframe
        """
        data_fetcher = fetch_external_data.ExternalData(
            first_date=self.start_date,
            last_date=self.end_date,
        )
        df = data_fetcher.fetch_from_csv(
            csv_dirs=self._csv_dirs,
            ticker=ticker,
            field_mappins=field_mappings
        )

        df = df.reindex(self.baseline_df.index)
        df["tic"] = ticker
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

    def fetch_data(self, field_mappings) -> pd.DataFrame:
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
                field_mappings=field_mappings
            )
            if not temp_df.empty:
                combined_frame = combined_frame.append(temp_df)

        combined_frame = combined_frame.sort_values(by=["date", "tic"]).reset_index(drop=True)
        out_dir_full.mkdir(parents=True, exist_ok=True)
        full_name = out_dir_full / exp_filename
        combined_frame.to_csv(full_name, index=1, encoding="utf-8")
        return combined_frame
