# %% common library
import glob
import os
import pathlib as pathlb
import numpy as np
import pandas as pd

# %% csv files
in_dir = 'import_csv'
out_dir = 'result_csv'
path = pathlb.Path.cwd()
all_files = glob.glob(str(path / in_dir) + "/*.csv")
exp_filename = "combined_tse.csv"
out_dir_all = path / out_dir
turb_csv = path / 'reference_csv' / 'dow30_turbulence_index.csv'

# %%
li = []
turb_df = pd.read_csv(turb_csv, parse_dates=['datadate'], index_col='datadate',
                      date_parser=lambda x: pd.to_datetime(x, format='%Y%m%d'))

# %%
for filename in all_files:
    df = pd.read_csv(filename, index_col='date', parse_dates=[
                     'date'], header=0, date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d'))
    df = df.drop(['count', 'value', 'adjClose'], axis=1)

    # make sure there is data for every day to avoid calendar errors
    df = df.resample('1d').pad()
    df = df.reindex(turb_df.index)
    df = df.reset_index()

    df = df.rename(columns=({'date': 'datadate',
                             'open': 'prcod',
                             'high': 'prchd',
                             'low': 'prcld',
                             'volume': 'cshtrd',
                             'close': 'prccd'}))

    # df['datadate'] = df['datadate'].map(lambda x: x.replace('-', ''))
    df['datadate'] = df['datadate'].dt.strftime('%Y%m%d')
    df["ajexdi"] = 1.0
    df["tic"] = pathlb.Path(filename).stem
    cols = ['datadate', 'tic', 'prccd', 'ajexdi',
            'prcod', 'prchd', 'prcld', 'cshtrd']
    df = df[cols]
    li.append(df)

# %%
frame = pd.concat(li, axis=0, ignore_index=True)

# %%
out_dir_all.mkdir(parents=True, exist_ok=True)

fullname = out_dir_all / exp_filename
frame.to_csv(fullname, index=1, encoding='utf-8')
frame.head()

# %%
