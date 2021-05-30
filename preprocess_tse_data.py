from finrl.preprocessing.data import data_split
from finrl.preprocessing.preprocessors import FeatureEngineer
from config import config
import numpy as np
import pandas as pd
from get_tse_data.tse_data import tse_data
from typing import Tuple
import logging

# matplotlib.use('Agg')
# get_ipython().run_line_magic('matplotlib', 'inline')

def preprocess_data() -> Tuple[pd.DataFrame, pd.DataFrame]:

    #if not os.path.exists("./" + config.DATA_SAVE_DIR):
    #    os.makedirs("./" + config.DATA_SAVE_DIR)
    #if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    #    os.makedirs("./" + config.TRAINED_MODEL_DIR)
    #if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    #    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
    #if not os.path.exists("./" + config.RESULTS_DIR):
    #    os.makedirs("./" + config.RESULTS_DIR)

    # from config.py start_date is a string
    logging.info(f'Start date: {config.START_DATE}')
    # from config.py end_date is a string
    logging.info(f'End date: {config.END_DATE}')
    logging.info(f'Tickers: {config.TSE_TICKER_30}')
    df = tse_data(start_date=config.START_DATE,
                end_date=config.END_DATE,
                ticker_list=config.TSE_TICKER_30).fetch_data()

    # 4.Preprocess Data
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
        use_turbulence=False,
        user_defined_feature=False)

    processed = fe.preprocess_data(df)

    # processed['log_volume'] = np.log(processed.volume*processed.close)
    processed['change'] = (processed.close-processed.open)/processed.close
    processed['daily_variance'] = (processed.high-processed.low)/processed.close
    processed['volume_ma_ratio'] = processed.volume_10_sma/processed.volume_30_sma
    logging.info(f'Preprocessed data: \n {processed.head()}')
    
    # 5.Design Environment
    train = data_split(processed, config.START_DATE, config.START_TRADE_DATE)
    trade = data_split(processed, config.START_TRADE_DATE, config.END_DATE)
    logging.info(f'Training sample size: {len(train)}')
    logging.info(f'Trading sample size: {len(trade)}')
    logging.info(f'Training column names: {train.columns}')

    return train, trade
