"""Contains methods and classes to collect data from csv files."""

from typing import List
import pandas as pd
from tensortrade.data.cdd import CryptoDataDownload
from preprocess_data.csv_data import CSVData


class ReadCSV(CryptoDataDownload):
    """Provides methods for retrieving data from csv files.

    """

    def __init__(self) -> None:
        super().__init__()

    def fetch_tsetmc(self,
                      quote_symbol: str,
                      base_dirs: list) -> pd.DataFrame:
        """Fetches data in csv files downloaded from tsetmc.com that matches the symbol name.

        Parameters
        ----------
        quote_symbol : str
            The quote symbol fo the cryptocurrency pair.
        base_dir : str
            The directory where the csv files are located.

        Returns
        -------
        `pd.DataFrame`
            A open, high, low, close and volume for the specified exchange.
        """
        data_manager = CSVData(
            start_date = "2018-01-01",
            end_date = "2020-12-31",
            csv_dirs = base_dirs,
            baseline_file_name = "tickers_data/tse/adjusted/شاخص كل6.csv",
            has_daily_trading_limit = True,
            use_baseline_data = True,
        )
        file_name = f'{quote_symbol}-ت'
        df = data_manager.process_single_tic(
            file_name,
            None,
            'date'
        )
        # df = pd.read_csv(base_dir + filename, skiprows=0)
        # df = df[::-1]
        df = df.drop(["tic"], axis=1)
        # df = df.rename({base_vc: new_base_vc, quote_vc: new_quote_vc, "Date": "date"}, axis=1)
        # df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        # df = df.set_index("date")
        # df.columns = [name.lower() for name in df.columns]
        # df = df.reset_index()
        return df

    def fetch(self,
              exchange_name: str,
              base_symbol: str,
              quote_symbol: str,
              timeframe: str = "d",
              include_all_volumes: bool = False,
              base_dirs: list = ["tickers_data/tse/"]) -> pd.DataFrame:
        """Fetches data for different exchanges and cryptocurrency pairs.

        Parameters
        ----------
        exchange_name : str
            The name of the exchange.
        base_symbol : str
            The base symbol fo the cryptocurrency pair.
        quote_symbol : str
            The quote symbol fo the cryptocurrency pair.
        timeframe : {"d", "h", "m"}
            The timeframe to collect data from.
        include_all_volumes : bool, optional
            Whether or not to include both base and quote volume.
        base_dir : str, optional
            The directory where the csv files are located.

        Returns
        -------
        `pd.DataFrame`
            A open, high, low, close and volume for the specified exchange and
            cryptocurrency pair.
        """
        if exchange_name.lower() == "tsetmc":
            return self.fetch_tsetmc(quote_symbol, base_dirs)
        return super().fetch(exchange_name,
                                  base_symbol,
                                  quote_symbol,
                                  timeframe,
                                  include_all_volumes=include_all_volumes)
