"""
testing models.py
"""
from . import models

class TestTradeDRLAgent:
    """
    testing TradeDRLAgent
    """
    def test_get_model_dirs(self):
        """
        testing _get_model_dirs(...)
        """
        root_dir = 'root_dir'
        str_1, str_2, str_3 = models.TradeDRLAgent()._get_model_dirs(root_dir=root_dir)
        assert isinstance(str_1, str), "dir is not a string"
        assert isinstance(str_2, str), "dir is not a string"
        assert isinstance(str_3, str), "dir is not a string"
