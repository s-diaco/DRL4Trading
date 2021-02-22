# %% common library
import pandas as pd
import numpy as np
import pandas as pd
  
#%% csv file name 
import_filename = "بترانس"
exp_filename = "tse"
#filename = "dow_30_2009_2020"
#import_filename = "test"
fileext = "csv"

# %% read the file
df = pd.read_csv(import_filename+"."+fileext) 
df.head()
      
# %% change and remove some columns
df = df.drop(['count', 'value', 'adjClose'], axis = 1)
df = df.rename(columns=({'date':'datadate', 
                        'open':'prcod',
                        'high':'prchd',
                        'low':'prcld',
                        'volume':'cshtrd',
                        'close':'prccd'}))
df['datadate'] = df['datadate'].map(lambda x: x.replace('-',''))
df["ajexdi"] = 1.0
df["tic"] = import_filename
cols = ['datadate', 'tic', 'prccd', 'ajexdi', 'prcod', 'prchd', 'prcld', 'cshtrd'] 
df = df[cols]
df = df[df['datadate'] >= '20090101']
df = df.reset_index(drop=True)
df.head()
# %% save to the file
df.to_csv(exp_filename+'.'+fileext, index=1, encoding='utf-8')
# %%
