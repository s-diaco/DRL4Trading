"""
test 'fetch_external_data.py'
"""
import pytest
from . import fetch_external_data


@pytest.fixture
def ext_data():
    TEST_TIC = "وتوشه"
    FIRST_DATE = '20180101'
    LAST_DATE = '20180628'
    TEST_CSV_DIR = "test_dir"
    ext_data = fetch_external_data.ExternalData(
        TEST_TIC, FIRST_DATE, LAST_DATE, csv_dir=TEST_CSV_DIR)
    yield ext_data


class TestExternalData:
    """
    test ExternalData class from 'fetch_external_data.py'
    """

    def test_fetch_volume_data(self, ext_data):
        df = ext_data.fetch_volume_data()
        assert len(df.columns) == 18, "volume table has missing columns"
