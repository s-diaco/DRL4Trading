import logging

from finrl.trade.backtest import backtest_stats
from IPython import get_ipython

import backtest.backtest_plot as bt_plt


def backtest_trades(df_account_value, baseline):
    # ## 7.1 Backtest Stats
    logging.info("==============Backtest Results===========")
    _ = backtest_stats(
        account_value=df_account_value, value_col_name="total_assets"
    )

    # ## 7.2 Backtest Plot
    logging.info("==============Compare to baseline===========")
    ipy = get_ipython()
    if ipy is not None:
        ipy.run_line_magic('matplotlib', 'inline')
    # get_ipython().run_line_magic("matplotlib", "inline")

    bt_plt.backtest_plot(
        df_account_value,
        baseline_df=baseline,
        value_col_name="total_assets",
    )
