"""
test 'fetch_external_data.py'
"""
import pytest
import pathlib
from . import fetch_external_data


@pytest.fixture
def ext_data():
    TEST_TIC = "بترانس"
    FIRST_DATE = '20180101'
    LAST_DATE = '20180628'
    TEST_CSV_DIR = pathlib.Path().cwd()/'tickers_data'
    external_data = fetch_external_data.ExternalData(
        TEST_TIC, FIRST_DATE, LAST_DATE, csv_dir=TEST_CSV_DIR)
    yield external_data


class TestExternalData:
    """
    test ExternalData class from 'fetch_external_data.py'
    """

    def test_fetch_data(self, ext_data):
        """
        test fetch_data with and without client types
        """
        include_df_false = ext_data.fetch_data(include_client_types=False)
        assert len(include_df_false.columns) == 14, "price table has missing columns"
        include_df_true = ext_data.fetch_data(include_client_types=True)
        assert len(include_df_true.columns) == 32, "volume table has missing columns"
