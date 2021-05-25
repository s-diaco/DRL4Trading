"""
get data from third pary sources
"""
import pandas as pd
import logging
import pytse_client as tse


class ExternalData:
    """
    tools to get data related to a specific ticker
    from third party sources

    Args:
        ticker: a unique string indicating symbol name for the stock 
        baseline_dates: list of dates that we should get data for

    Returns:
        a pandas dataframe with dates, ticker and data
    """

    def __init__(self, ticker: str, first_date: str, last_date: str, csv_dir: str):
        self.first_day = first_date
        self.last_day = last_date
        self.ticker = tse.Ticker(ticker)
        self.csv_dir = csv_dir

    def fetch_volume_data(self) -> pd.DataFrame:
        """
        fetch volume data from the specified ticker

        Returns:
            a dataframe with dates and volumes for a ticker
        """
        logging.info(f"fetch volume data for {self.ticker.title}")
        df = pd.DataFrame(self.ticker.client_types)
        df = df[(df["date"] > self.first_day) & (df["date"] < self.last_day)]
        return df
