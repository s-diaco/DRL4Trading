# DRL4Trading
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

DRL4Trading is an open source Python framework for building, training and evaluating trading algorithms using deep reinforcement learning. This framework uses numpy, pandas, tensorflow (v2) and tf-agents.

## Features and Limitations

- **Add your own indicator.** Users can add any calculated feature to the data by implementing a simple calss. For example a technical indicator or a fundamental parameter can be added by implementing "CustomColumn" class as seen in "preprocess_data/custom_columns.py".

- **Multiple csv files.** You can use several csv files for one stock. for example you can load "price data" for each ticker (open, high, low, close ...) in a directory and "fundamental data" in another directory. Just add the list of dirs in settings file (config/settins.py) in section "CSV_FILE_SETTINGS->dir_list".

- **Only csv files.** As of today, the framework work just with csv data. you should provide separate csv files for every single ticker.

- **Supports DTL.** In some markets there is a [daily trading limit](https://www.investopedia.com/terms/d/daily_trading_limit.asp) (A maximum or minimum price per day). The envoirement supports daily trading limits and stopped tickers.

- **Select columns to train.** You can set wich columns to use for the training in the settings file (config/settings.py) under "DATA_COLUMNS" section.

## Bug reports

You can post **bug reports and feature requests** in [GitHub issues](https://github.com/s-diaco/DRL4Trading/issues).

## Contributors

Contributions are encouraged and welcomed. This project is meant to grow as the community around it grows. Let me know if there is anything that you would like to see in the future, or if there is anything you feel is missing.

**Working on your first Pull Request?** You can learn how from this _free_ series [How to Contribute to an Open Source Project on GitHub](https://egghead.io/series/how-to-contribute-to-an-open-source-project-on-github)
