# %% common library
import glob
import os
import pathlib as pathlb
import numpy as np
import pandas as pd
import config.config as cfg

# %% csv files
in_dir = cfg.IN_DIR
out_dir = cfg.CSV_DIR
exp_filename = 'v2_'+cfg.EXP_FILE_NAME
path = pathlb.Path.cwd()
all_files = glob.glob(str(path / in_dir) + "/*.csv")
out_dir_all = path / out_dir

# %%
li = []
# %%
for filename in all_files:
    df = pd.read_csv(filename, index_col='date', parse_dates=[
                     'date'], header=0, date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d'))
    df = df.drop(['count', 'value', 'adjClose'], axis=1)

    # make sure there is data for every day to avoid calendar errors
    df = df.resample('1d').pad()
    df = df.reset_index()
    df["tic"] = pathlb.Path(filename).stem
    cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'tic']
    df = df[cols]
    # create day of the week column (monday = 0)
    df["day"] = df["date"].dt.dayofweek
    # convert date to standard string format, easy to filter
    df["date"] = df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
    # drop missing data
    df = df.dropna()
    df = df.reset_index(drop=True)
    print("Shape of DataFrame: ", df.shape)
    li.append(df)

# %%
frame = pd.concat(li, axis=0, ignore_index=True)

frame = frame.sort_values(by=['date', 'tic']).reset_index(drop=True)


# %%
out_dir_all.mkdir(parents=True, exist_ok=True)
fullname = out_dir_all / exp_filename
frame.to_csv(fullname, index=1, encoding='utf-8')
frame.head()


# %%
