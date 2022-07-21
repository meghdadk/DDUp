"""Dataset registrations."""
import os

import numpy as np

import common
import pandas as pd

def LoadDmv(filename='Vehicle__Snowmobile__and_Boat_Registrations.csv', batch_num=None, finetune=False): 
    #csv_file = '../data/DMV/{}'.format(filename)
    csv_file = './permuted_dataset.csv'
    cols = [
        'Record Type', 'Registration Class', 'State', 'County', 'Body Type',
        'Fuel Type', 'Reg Valid Date', 'Color', 'Scofflaw Indicator',
        'Suspension Indicator', 'Revocation Indicator'
    ]
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    type_casts = {'Reg Valid Date': np.datetime64}
    df = pd.read_csv(csv_file,usecols=cols, sep = ',')

    df = df.dropna(axis=1, how='all')
	
    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    df.replace('', np.nan, inplace=True)
    df.dropna(inplace=True)
    if batch_num!=None:
        landmarks = int(len(df)*10/12) + np.linspace(0, int((len(df)*10/12)*0.2), 6, dtype=np.int)
        df = df.iloc[:landmarks[batch_num]]

    if finetune:
        landmarks = int(len(df)*10/12) + np.linspace(0, int((len(df)*10/12)*0.2), 6, dtype=np.int)
        return common.CsvTable('DMV', df, cols), landmarks


    #landmarks = int(len(df)*10/12) + np.linspace(0, int((len(df)*10/12)*0.2), 6, dtype=np.int)
    #df = df.iloc[:landmarks[5]]
    print (df.shape)


    return common.CsvTable('DMV', df, cols, type_casts)


def LoadPermutedDmv(filename='Vehicle__Snowmobile__and_Boat_Registrations.csv', permute=True):
    csv_file = '../data/DMV/{}'.format(filename)
    cols = [
        'Record Type', 'Registration Class', 'State', 'County', 'Body Type',
        'Fuel Type', 'Reg Valid Date', 'Color', 'Scofflaw Indicator',
        'Suspension Indicator', 'Revocation Indicator'
    ]
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    type_casts = {'Reg Valid Date': np.datetime64}
    df = pd.read_csv(csv_file,usecols=cols, sep = ',')
    print (df.shape)
    df = df.dropna(axis=1, how='all')
	
    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    df.replace('', np.nan, inplace=True)
    df.dropna(inplace=True)

   
    if permute:
        columns_to_sort = df.columns

        sorted_columns = pd.concat([df[col].sort_values(ignore_index=True).reset_index(drop=True) for col in columns_to_sort], axis=1, ignore_index=True)
        sorted_columns.columns = df.columns
        update_sample = sorted_columns.sample(frac=0.2)
    else:
        update_sample = df.sample(frac=0.2)

    if permute:    
        del sorted_columns


    landmarks = len(df) + np.linspace(0, len(update_sample), 2, dtype=np.int)

    data = pd.concat([df,update_sample])
    del df
    del update_sample

    data.to_csv('permuted_dataset.csv', sep=',', index=None)
    return common.CsvTable('DMV', data, cols=data.columns, type_casts=type_casts), landmarks

def LoadPartlyPermutedDmv(filename='Vehicle__Snowmobile__and_Boat_Registrations.csv', num_of_sorted_cols=1):
    csv_file = '../data/DMV/{}'.format(filename)
    cols = [
        'Record Type', 'Registration Class', 'State', 'County', 'Body Type',
        'Fuel Type', 'Reg Valid Date', 'Color', 'Scofflaw Indicator',
        'Suspension Indicator', 'Revocation Indicator'
    ]
    assert num_of_sorted_cols < len(cols)

    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    type_casts = {'Reg Valid Date': np.datetime64}
    df = pd.read_csv(csv_file,usecols=cols, sep = ',')
    print (df.shape)
    df = df.dropna(axis=1, how='all')
    
    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    df.replace('', np.nan, inplace=True)
    df.dropna(inplace=True)

    if num_of_sorted_cols==0:
        update_sample = df.sample(frac=0.2)
        landmarks = len(df) + np.linspace(0, len(update_sample), 2, dtype=np.int)

        data = pd.concat([df,update_sample])
        del df
        del update_sample
        data.to_csv('permuted_dataset.csv', sep=',', index=None)
        return common.CsvTable('DMV', df, cols=df.columns, type_casts=type_casts)


    columns_to_sort = [df.columns[i] for i in range(num_of_sorted_cols)]
    columns_not_sort = [df.columns[i] for i in range(num_of_sorted_cols, len(df.columns))]

    sorted_columns = pd.concat(([df[col].sort_values(ignore_index=True).reset_index(drop=True) for col in columns_to_sort]+[df[col] for col in columns_not_sort]), axis=1, ignore_index=True)
    sorted_columns.columns = df.columns
   
    update_sample = sorted_columns.sample(frac=0.2)
    landmarks = len(df) + np.linspace(0, len(update_sample), 2, dtype=np.int)

    data = pd.concat([df,update_sample])
    del df
    del update_sample

    data.to_csv('permuted_dataset.csv', sep=',', index=None)
    return common.CsvTable('DMV', data, cols=data.columns, type_casts=type_casts)

if __name__=="__main__":
    LoadPermutedDmv(permute=True)
    #LoadPartlyPermutedDmv()
