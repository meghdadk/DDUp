"""Dataset registrations."""
import os

import numpy as np

import common
import pandas as pd


def LoadTPCH(filename="",batch_num=None,finetune=False):
    csv_file = '/home/meghdad/DDUp/tpch_data/{}'.format(filename)
    cols = ['order_date','segment','nation']

    df = pd.read_csv(csv_file,usecols=cols, sep = ',')
    #df = df[cols]
    df = df.dropna(axis=1, how='all')
	
    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    df.replace('', np.nan, inplace=True)

    print (df.shape)

    return common.CsvTable('imdb', df, cols)
