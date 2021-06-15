import logging

import numpy as np
import pandas as pd
from config import config
from numpy.core.multiarray import where
from numpy.core.numeric import zeros_like


def add_user_defined_features(processed) -> pd.DataFrame:
    user_added_feature_list = config.USER_DEFINED_FEATURES
    # processed['log_volume'] = np.log(processed.volume*processed.close)
    processed['change'] = np.divide(
        ((processed.close-processed.open)/processed.close),
        ((processed.index_close-processed.index_open)/processed.index_close),
        out=np.ones_like(processed.index_close),
        where=(processed.index_close-processed.index_open)!=0)
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

    return processed
