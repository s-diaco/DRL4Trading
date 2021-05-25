import glob
import pathlib
import pandas as pd
import get_tse_data.tse_config.tse_config as cfg
import logging
from finrl.trade.backtest import get_baseline
from . import fetch_external_data


class tse_data:
    """Provides methods for retrieving daily stock data from
    TSE (Tehran stock exchange)

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
        Fetches data from yahoo API

    """

    def __init__(self, start_date: str, end_date: str, ticker_list: list = []):

        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list
        self.index_df = self._get_tse_index()
        self.baseline_df = self.index_df.reset_index()

    def _get_tse_index(self) -> pd.DataFrame:
        """
        get tse index as a dataframe
        """
        logging.info(f"Adding TSE Index.")

        path = pathlib.Path.cwd()
        # tsei_dir = cfg.CSV_DIR
        # tsei_file_name = cfg.TSEI
        tsei_dir = cfg.IN_DIR
        tsei_file_name = cfg.TSEI
        df = pd.read_csv(
            path / tsei_dir / "adjusted" / tsei_file_name,
            parse_dates=["date"],
            header=0,
            date_parser=lambda x: pd.to_datetime(x, format="%Y-%m-%d"),
        )
        cols = ["date", "open", "high", "low", "close", "volume", "tic"]
        df = df[cols]
        # create day of the week column (monday = 0)
        df["day"] = pd.to_datetime(df["date"]).dt.dayofweek
        df["tic"] = "TSEI"
        df = df.set_index('date')
        df = df.loc[self.start_date:self.end_date]
        logging.info(f"Added TSE Index.")
        return df

    def process_single_tic(self,
                           ticker,
                           include_client_types,
                           adjusted,
                           path,
                           in_dir) -> pd.DataFrame:
        """
        prepare a single ticker and return the df
        """
        logging.info(f"Adding file: {ticker}")
        # using data from tseclient_v2
        data_fetcher = fetch_external_data.ExternalData(
            ticker=ticker,
            first_date=self.start_date,
            last_date=self.end_date,
            csv_dir= path / in_dir
        )
        df = data_fetcher.fetch_data(
            include_client_types=include_client_types,
            adjusted_price=adjusted
        )

        df = df.reindex(self.index_df.index)
        df["tic"] = ticker
        df['index_close'] = self.index_df['close']
        df['index_volume'] = self.index_df['volume']
        df["stopped"] = df["open"].isnull()
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

    def fetch_data(self, adjusted=True, include_client_types=True) -> pd.DataFrame:
        """
        get ticker data
        """
        in_dir = cfg.IN_DIR
        out_dir = cfg.CSV_DIR
        exp_filename = cfg.EXP_FILE_NAME
        path = pathlib.Path.cwd()
        out_dir_all = path / out_dir
        li = []
        for tic in self.ticker_list:
            df = self.process_single_tic(
                tic,
                include_client_types,
                adjusted,
                path,
                in_dir
            )
            if not df.empty:
                li.append(df)

        frame = pd.concat(li, axis=0, ignore_index=True)
        frame = frame.sort_values(by=["date", "tic"]).reset_index(drop=True)
        out_dir_all.mkdir(parents=True, exist_ok=True)
        fullname = out_dir_all / exp_filename
        frame.to_csv(fullname, index=1, encoding="utf-8")
        return frame

    # todo: delete
    def combine_csv(self) -> pd.DataFrame:
        logging.basicConfig(
            format="%(asctime)s - %(message)s", level=logging.INFO)
        # %% csv files
        in_dir = cfg.IN_DIR
        out_dir = cfg.CSV_DIR
        exp_filename = "v2_" + cfg.EXP_FILE_NAME
        path = pathlib.Path.cwd()
        all_files = glob.glob(str(path / in_dir) + "/*.csv")
        out_dir_all = path / out_dir

        baseline_df = get_baseline(
            ticker="^DJI", start=self.start_date, end=self.end_date
        )
        li = []
        logging.info(f"{len(all_files)} csv files found.")
        for filename in all_files:
            logging.info(f"Adding file: {filename}.")
            df = pd.read_csv(
                filename,
                index_col="date",
                parse_dates=["date"],
                header=0,
                date_parser=lambda x: pd.to_datetime(x, format="%Y-%m-%d"),
            )
            df = df.drop(["count", "value", "adjClose"], axis=1)

            # make sure there is data for every day to avoid calendar errors
            df = df.resample("1d").pad()
            new_index = pd.to_datetime(baseline_df["date"])
            df = df.reindex(new_index)
            df = df.reset_index()
            df["tic"] = pathlib.Path(filename).stem
            cols = ["date", "open", "high", "low", "close", "volume", "tic"]
            df = df[cols]
            # convert date to standard string format, easy to filter
            df["date"] = df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
            # drop missing data
            df = df.dropna()
            df = df.reset_index(drop=True)
            # create day of the week column (monday = 0)
            df["day"] = pd.to_datetime(df["date"]).dt.dayofweek
            print("Shape of DataFrame: ", df.shape)
            li.append(df)

        frame = pd.concat(li, axis=0, ignore_index=True)

        frame = frame.sort_values(by=["date", "tic"]).reset_index(drop=True)

        out_dir_all.mkdir(parents=True, exist_ok=True)
        fullname = out_dir_all / exp_filename
        frame.to_csv(fullname, index=1, encoding="utf-8")
        frame.head()

        return frame