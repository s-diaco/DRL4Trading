"""
get data from third pary sources
"""
import pandas as pd
import logging
import pytse_client as tse
import pathlib


class ExternalData:
    """
    tools to get data related to a specific ticker
    from third party sources

    Args:
        ticker: str
            The symbol of TSE ticker
        csv_dir: Path
            Directory to save data file

    Returns:
        a pandas dataframe with dates, ticker and data
    """

    def __init__(self, ticker: str, first_date: str, last_date: str, csv_dir: pathlib.Path):
        self.first_day = first_date
        self.last_day = last_date
        self.ticker = tse.Ticker(ticker)
        self.csv_dir = csv_dir

    def fetch_data(
        self,
        include_client_types: bool = True,
        adjusted_price: bool = True
    ) -> pd.DataFrame:
        """Downloads data  from tsetmc.com
        Parameters:
        ----------
            tic : str
                The symbol of TSE ticker
            base_path : str
                Directory to save data file
        """
        logging.info(f"fetching data for {self.ticker.title}")
        if adjusted_price:
            price_file_name = f'adjusted/{self.ticker.symbol}-ت.csv'
        else:
            price_file_name = f'{self.ticker.symbol}.csv'
        if (self.csv_dir/pathlib.Path(price_file_name)).is_file():
            # price file exist. open and read it
            price_df = pd.read_csv(
                self.csv_dir/price_file_name,
                index_col="date",
                parse_dates=["date"],
                header=0,
                date_parser=lambda x: pd.to_datetime(x, format="%Y-%m-%d"),
            )
        else:
            # price file not exist
            if adjusted_price:
                raise ValueError(
                    f'There is no downloaded file for {self.ticker.symbol} with adjusted prices'
                    )
            else:
                price_df = pd.DataFrame(
                    tse.download(
                        symbols=self.ticker.symbol,
                        write_to_csv=True,
                        base_path=self.csv_dir
                    )[self.ticker.symbol]
                )
                price_df = price_df.set_index('date')
        price_df = price_df.sort_index()
        ret_val = price_df.loc[self.first_day:self.last_day]
        if include_client_types:
            client_types_path = self.csv_dir / "client_types"
            client_types_path.mkdir(parents=True, exist_ok=True)
            client_types_file = f'{self.ticker.symbol}.csv'
            full_path = client_types_path/client_types_file
            if full_path.is_file():
                client_types_df = pd.read_csv(
                    client_types_path/client_types_file,
                    index_col="date",
                    parse_dates=["date"],
                    header=0,
                    date_parser=lambda x: pd.to_datetime(x, format="%Y-%m-%d"),
                )
            else:
                client_types_df = pd.DataFrame(self.ticker.client_types)
                client_types_df.to_csv(
                    client_types_path/f'{self.ticker.symbol}.csv')
                client_types_df = client_types_df.set_index('date')
            client_types_df = client_types_df.sort_index()
            client_types_df = client_types_df.loc[self.first_day:self.last_day]
            ret_val = ret_val.join(client_types_df)
        return ret_val
