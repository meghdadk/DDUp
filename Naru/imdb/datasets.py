"""Dataset registrations."""
import os

import numpy as np

import common
import pandas as pd


def LoadIMDB(filename="title_join_mc_mii_b2010.csv",batch_num=None,finetune=False):
    csv_file = '/home/meghdad/DDUp/job_data/{}'.format(filename)
    cols = ['kind_id','production_year','info_type_id','company_type_id']

    df = pd.read_csv(csv_file,usecols=cols, sep = ',')
    #df = df[cols]
    df = df.dropna(axis=1, how='all')
	
    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    df.replace('', np.nan, inplace=True)
    #df.dropna(inplace=True)
    if batch_num!=None:
        landmarks = int(len(df)*10/12) + np.linspace(0, int((len(df)*10/12)*0.2), 6, dtype=np.int)
        df = df.iloc[:landmarks[batch_num]]

    if finetune:
        landmarks = int(len(df)*10/12) + np.linspace(0, int((len(df)*10/12)*0.2), 6, dtype=np.int)
        return common.CsvTable('imdb', df, cols), landmarks


    #landmarks = int(len(df)*10/12) + np.linspace(0, int((len(df)*10/12)*0.2), 6, dtype=np.int)
    #df = df.iloc[:landmarks[5]]
    print (df.shape)


    return common.CsvTable('imdb', df, cols)



if __name__=="__main__":
    LoadPermutedIMDB(permute=False)
    #LoadPartlyPermutedIMDB()
