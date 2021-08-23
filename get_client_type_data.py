from pytse_client import download_client_types_records

from config import settings

if __name__ == '__main__':
    symbol_list = settings.TSE_TICKER
    records_dict = download_client_types_records(symbols=symbol_list, write_to_csv=True, base_path="cts")