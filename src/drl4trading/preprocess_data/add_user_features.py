import logging

from tensortrade.feed.core.base import IterableStream

from src.drl4trading.preprocess_data import user_features


def col_from_cls(feat_class, data):
    '''
    Create "series" from a given function and dataframe

            Parameters:
                    client_func (callable): function used to create the column
                    data (pd.DataFrame): data used to calculate new columns

            Returns:
                    column (pd.Series): calculated column

            Raises:
                    TypeError: if the column type is not pd.Series
    '''
    # TODO check if there are any Nan or inf values in new column
    column_cls = feat_class(data)
    if isinstance(column_cls, IterableStream):
        return column_cls
    else:
        raise TypeError('Feature class has to inherit "IterableStream"')


def add_features(symbol, data, user_cols) -> list:
    '''
    Add data from functions in 'user_calculated_columns.py'.

            Parameters:
                    data (pd.DataFrame): data used to calculate new columns
                    user_cols (list): user class names to use

            Returns:
                    data (pd.DataFrame): the updated dataframe
    '''
    logging.info('Adding custom columns')
    feature_streams = []
    for col_cls in IterableStream.__subclasses__():
        cls_name = col_cls.__name__
        if(col_cls.__module__ == user_features.__name__):

            if cls_name in user_cols:
                try:
                    # add new column to dataframe
                    new_feature = col_from_cls(
                        col_cls, data
                        ).rename(cls_name + '/' + symbol)
                    logging.info(f'Add feature "{str(new_feature)}" ✅')
                    feature_streams.append(new_feature)
                except Exception as e:
                    logging.info(f'Add column from user class "{cls_name}" '
                                 f'❌: {str(e)}')
    return feature_streams
