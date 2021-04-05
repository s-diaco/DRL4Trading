# %% common library
import glob
import os
import pathlib as pathlb
import numpy as np
import pandas as pd
import get_tse_data.tse_config.tse_config as cfg
import logging
from shutil import copyfile
from finrl.trade.backtest import get_baseline

# %%


class tse_data:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API

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

    def __init__(self, start_date: str, end_date: str, ticker_list: list):

        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def combine_csv(self) -> pd.DataFrame:
        logging.basicConfig(
            format='%(asctime)s - %(message)s', level=logging.INFO)
        # %% csv files
        in_dir = cfg.IN_DIR
        out_dir = cfg.CSV_DIR
        exp_filename = 'v2_'+cfg.EXP_FILE_NAME
        path = pathlb.Path.cwd()
        all_files = glob.glob(str(path / in_dir) + "/*.csv")
        out_dir_all = path / out_dir

        baseline_df = get_baseline(ticker='^DJI',
                                   start=self.start_date,
                                   end=self.end_date)
        li = []
        # %%
        logging.info(f'{len(all_files)} csv files found.')
        for filename in all_files:
            logging.info(f'Adding file: {filename}.')
            df = pd.read_csv(filename, index_col='date',
                             parse_dates=['date'], header=0,
                             date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d'))
            df = df.drop(['count', 'value', 'adjClose'], axis=1)

            # make sure there is data for every day to avoid calendar errors
            df = df.resample('1d').pad()
            new_index = pd.to_datetime(baseline_df['date'])
            df = df.reindex(new_index)
            df = df.reset_index()
            df["tic"] = pathlb.Path(filename).stem
            cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'tic']
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

        frame = frame.sort_values(by=['date', 'tic']).reset_index(drop=True)

        # %%
        out_dir_all.mkdir(parents=True, exist_ok=True)
        fullname = out_dir_all / exp_filename
        frame.to_csv(fullname, index=1, encoding='utf-8')
        frame.head()

        return frame


# %%