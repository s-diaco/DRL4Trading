from finrl.preprocessing.data import data_split
from finrl.preprocessing.preprocessors import FeatureEngineer
import numpy as np
from numpy.core.multiarray import where
from numpy.core.numeric import zeros_like
from config import config
import pandas as pd
from get_tse_data.tse_data import TSEData
from typing import Tuple
import logging

# matplotlib.use('Agg')
# get_ipython().run_line_magic('matplotlib', 'inline')

def preprocess_data(tic_list = config.TSE_TICKER_5, add_custom_columns=True) -> Tuple[pd.DataFrame, pd.DataFrame]:

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
    logging.info(f'Tickers: {tic_list}')
    tse_data = TSEData(start_date=config.START_DATE,
                end_date=config.END_DATE,
                ticker_list=tic_list)
    raw_df = tse_data.fetch_data()

    # 4.Preprocess Data
    feat_eng = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
        use_turbulence=False,
        user_defined_feature=False)

    processed = feat_eng.preprocess_data(raw_df)

    if add_custom_columns:
        # processed['log_volume'] = np.log(processed.volume*processed.close)
        processed['change'] = np.divide(
            ((processed.close-processed.open)/processed.close),
            ((processed.index_close-processed.index_yesterday)/processed.index_close),
            out=np.ones_like(processed.index_close),
            where=(processed.index_close-processed.index_yesterday)!=0)
        processed['daily_variance'] = np.divide(
            processed.high-processed.low,
            processed.close,
            out=zeros_like(processed.close),
            where=processed.close!=0)
        processed['volume_ma_ratio'] = np.divide(
            processed.volume_5_sma,
            processed.volume_30_sma,
            out=np.ones_like(processed.close),
            where=processed.volume_30_sma!=0)
        processed['count_ma_ratio'] = np.divide(
            processed.count_5_sma,
            processed.count_30_sma,
            out=np.ones_like(processed.close),
            where=processed.count_30_sma!=0)
        processed['ma_ratio'] = np.divide(
            processed.close_5_sma,
            processed.close_30_sma,
            out=np.ones_like(processed.close),
            where=processed.close_30_sma!=0)
        processed['indv_buy_sell_ratio'] = np.divide(
            processed.individual_buy_count,
            processed.individual_sell_count,
            out=np.zeros_like(processed.close),
            where=processed.individual_sell_count!=0)
        processed['corp_buy_sell_ratio'] = np.divide(
            processed.corporate_buy_count,
            processed.corporate_sell_count,
            out=np.zeros_like(processed.close),
            where=processed.corporate_sell_count!=0)
        processed['ind_corp_buy_ratio'] = np.divide(
            processed.individual_buy_vol,
            processed.corporate_buy_vol,
            out=np.zeros_like(processed.close),
            where=processed.corporate_buy_vol!=0)
    logging.info(f'Preprocessed data (tail): \n {processed.tail()}')
    
    # 5.Design Environment
    train = data_split(processed, config.START_DATE, config.START_TRADE_DATE)
    trade = data_split(processed, config.START_TRADE_DATE, config.END_DATE)
    logging.info(f'Training sample size: {len(train)}')
    logging.info(f'Trading sample size: {len(trade)}')
    logging.info(f'Training column names: {train.columns}')

    return train, trade
