# %% common library
import glob
import os
import pathlib as pathlb
import numpy as np
import pandas as pd
import get_tse_data.tse_config.tse_config as cfg
import logging
import pytse_client as tse
from pathlib import Path
from shutil import copyfile
from finrl.trade.backtest import get_baseline

# %%


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
        self.baseline_df = self.get_tse_index()

    def get_tse_index(self) -> pd.DataFrame:
        logging.info(f"Adding TSEI.")

        path = pathlb.Path.cwd()
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
        df = df[(df["date"] > self.start_date) & (df["date"] < self.end_date)]
        logging.info(f"Added TSEI.")
        return df

    def tse_downloader(self, tic, base_path):
        """Downloads data  from tsetmc.com
        Parameters
        ----------
        tic : str
            The symbol of TSE ticker
        base_path : str
            Directory to save data file
        """
        tse.download(symbols=tic, write_to_csv=True, base_path=str(base_path))

    def process_single_tic(self, df, ticker, bline_index) -> pd.DataFrame:
        df = df.reindex(bline_index.index)
        df["tic"] = ticker
        df['index_close']= bline_index['close']
        df['index_volume']= bline_index['volume']
        df["stopped"] = df["open"].isnull()
        df["b_queue"] = (df["high"] == df["low"]) & (df["low"] > df["yesterday"])
        df["s_queue"] = (df["high"] == df["low"]) & (df["high"] < df["yesterday"])    
        df=df.fillna(method='ffill')
        df=df.fillna(method='bfill')
        df = df.reset_index()
        # create day of the week column (monday = 0+2)
        df["day"] = (pd.to_datetime(df["date"]).dt.dayofweek + 2) % 7
        return df

    def fetch_data(self, adjusted=True) -> pd.DataFrame:
        in_dir = cfg.IN_DIR
        out_dir = cfg.CSV_DIR
        exp_filename = cfg.EXP_FILE_NAME
        path = pathlb.Path.cwd()
        out_dir_all = path / out_dir
        li = []
        for tic in self.ticker_list:
            logging.info(f"Adding file: {tic}")
            # using data from tseclient_v2
            if adjusted:
                tic_fn = "adjusted/" + tic + "-ت.csv"
            else:
                tic_fn = tic + ".csv"
            tic_fnp = Path(tic_fn)
            # if there is a downloaded csv file, open it; otherwise download and save a csv file for the ticker
            try:
                df = pd.read_csv(
                    path / in_dir / tic_fnp,
                    index_col="date",
                    parse_dates=["date"],
                    header=0,
                    date_parser=lambda x: pd.to_datetime(x, format="%Y-%m-%d"),
                )
            except:
                if adjusted:
                    logging.error(f"No data for {tic}")
                else:
                    logging.info(f"No downloaded data for {tic}. downloading...")
                    self.tse_downloader(tic, path / in_dir)
                    df = pd.read_csv(
                        path / in_dir / tic_fnp,
                        index_col="date",
                        parse_dates=["date"],
                        header=0,
                        date_parser=lambda x: pd.to_datetime(x, format="%Y-%m-%d"),
                    )
            if not df.empty:
                df = self.process_single_tic(df,
                                             tic,
                                             self.baseline_df.set_index('date')
                                             )
                li.append(df)

        frame = pd.concat(li, axis=0, ignore_index=True)
        frame = frame.sort_values(by=["date", "tic"]).reset_index(drop=True)
        out_dir_all.mkdir(parents=True, exist_ok=True)
        fullname = out_dir_all / exp_filename
        frame.to_csv(fullname, index=1, encoding="utf-8")
        return frame

    # todo: delete
    def combine_csv(self) -> pd.DataFrame:
        logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
        # %% csv files
        in_dir = cfg.IN_DIR
        out_dir = cfg.CSV_DIR
        exp_filename = "v2_" + cfg.EXP_FILE_NAME
        path = pathlb.Path.cwd()
        all_files = glob.glob(str(path / in_dir) + "/*.csv")
        out_dir_all = path / out_dir

        baseline_df = get_baseline(
            ticker="^DJI", start=self.start_date, end=self.end_date
        )
        li = []
        # %%
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
            df["tic"] = pathlb.Path(filename).stem
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

        # %%
        frame = pd.concat(li, axis=0, ignore_index=True)

        frame = frame.sort_values(by=["date", "tic"]).reset_index(drop=True)

        # %%
        out_dir_all.mkdir(parents=True, exist_ok=True)
        fullname = out_dir_all / exp_filename
        frame.to_csv(fullname, index=1, encoding="utf-8")
        frame.head()

        return frame


# %%
