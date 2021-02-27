# %% common library
import pandas as pd
import numpy as np
import pandas as pd
import glob
import os

# %% csv files
in_dir = 'import_csv'
out_dir = 'result_csv'
path = os.getcwd()
all_files = glob.glob(os.path.join(path, in_dir) + "/*.csv")
exp_filename = "combined_tse.csv"
outdir = os.path.join(path, out_dir)

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    df.head()
    df = df.drop(['count', 'value', 'adjClose'], axis=1)
    df = df.rename(columns=({'date': 'datadate',
                             'open': 'prcod',
                             'high': 'prchd',
                             'low': 'prcld',
                             'volume': 'cshtrd',
                             'close': 'prccd'}))
    df['datadate'] = df['datadate'].map(lambda x: x.replace('-', ''))
    df["ajexdi"] = 1.0
    df["tic"] = os.path.split(filename)[1]
    cols = ['datadate', 'tic', 'prccd', 'ajexdi',
            'prcod', 'prchd', 'prcld', 'cshtrd']
    df = df[cols]
    df = df[df['datadate'] >= '20090101']
    li.append(df)

# %%
frame = pd.concat(li, axis=0, ignore_index=True)
if not os.path.exists(outdir):
    os.mkdir(outdir)

fullname = os.path.join(outdir, exp_filename)
frame.to_csv(fullname, index=1, encoding='utf-8')
frame.head()

# %%
