
import os
from pprint import pprint
from finrl.trade.backtest import backtest_stats
from finrl.model.models import DRLAgent
from env_tse.env_stocktrading_tse_stoploss import StockTradingEnvTSEStopLoss
from finrl.preprocessing.data import data_split
import matplotlib.pyplot as plt
import matplotlib
import backtest_tse.tse_backtest_plot as bt_plt
import logging

def backtest_tse_trades(df_account_value, baseline, start_date, end_date):
    # ## 7.1 Backtest Stats
    logging.info("==============Backtest Results===========")
    perf_stats_all = backtest_stats(
        account_value=df_account_value, value_col_name="total_assets"
    )

    # ## 7.2 Backtest Plot
    logging.info("==============Compare to baseline===========")
    get_ipython().run_line_magic("matplotlib", "inline")

    bt_plt.backtest_plot(
        df_account_value,
        baseline_ticker="^TSEI",
        baseline_start=start_date,
        baseline_end=end_date,
        value_col_name="total_assets",
    )
