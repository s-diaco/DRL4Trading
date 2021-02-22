# %% common library
import pandas as pd
import numpy as np
import pandas as pd
import glob
import os
  
# %% csv files
path = os.getcwd()
all_files = glob.glob(path + "/*.csv")
exp_filename = "combined_tse.csv"

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    df.head()
    df = df.drop(['count', 'value', 'adjClose'], axis = 1)
    df = df.rename(columns=({'date':'datadate', 
                            'open':'prcod',
                            'high':'prchd',
                            'low':'prcld',
                            'volume':'cshtrd',
                            'close':'prccd'}))
    df['datadate'] = df['datadate'].map(lambda x: x.replace('-',''))
    df["ajexdi"] = 1.0
    df["tic"] = os.path.split(filename)[1]
    cols = ['datadate', 'tic', 'prccd', 'ajexdi', 'prcod', 'prchd', 'prcld', 'cshtrd'] 
    df = df[cols]
    df = df[df['datadate'] >= '20090101']
    li.append(df)

#%%
frame = pd.concat(li, axis=0, ignore_index=True)
frame.to_csv('all_csvs', index=1, encoding='utf-8')
frame.head()

# %%
