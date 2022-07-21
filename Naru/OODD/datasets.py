"""Dataset registrations."""
import os
import numpy as np
import common
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(root, filename, cols):
    csv_file = os.path.join(root, filename)

    df = pd.read_csv(csv_file,usecols=cols, sep = ',')
    df = df.dropna(axis=1, how='all')
    
    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    df.replace('', np.nan, inplace=True)

    train = df
    #train, test = train_test_split(df, test_size=0.1)
    test = GetPermutedData(train, 0.2, permute=False)
    #permuted = GetPermutedData(train, 0.2, permute=True)
    #permuted = GetPartlyPermutedData(train, 0.2, 2)

    permuted = pd.DataFrame()
    for i in range(1, 6):
         part = GetPartlyPermutedData(train, 0.2/5, i)
         permuted = pd.concat([permuted, part])



    print ("train set size: {}".format(train.shape))
    print ("test set size: {}".format(test.shape))

    df.to_csv('data/main.csv', sep=',', index=None)
    train.to_csv('data/train.csv', sep=',', index=None)
    test.to_csv('data/test_IND.csv', sep=',', index=None)
    permuted.to_csv('data/test_OOD.csv', sep=',', index=None)

def LoadData(root, filename, cols):
    csv_file = os.path.join(root, filename)
    df = pd.read_csv(csv_file,usecols=cols, sep = ',')
    df = df.dropna(axis=1, how='all')
	
    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    df.replace('', np.nan, inplace=True)

    return common.CsvTable('data', df, cols)

def GetPermutedData(data, frac, permute=True):
    df = data
   
    if permute:
        columns_to_sort = df.columns

        sorted_columns = pd.concat([df[col].sort_values(ignore_index=True).reset_index(drop=True) for col in columns_to_sort], axis=1, ignore_index=True)
        sorted_columns.columns = df.columns
        update_sample = sorted_columns.sample(frac=frac)
    else:
        update_sample = df.sample(frac=frac)



    return update_sample

def GetPartlyPermutedData(data, frac, num_of_sorted_cols):
    assert num_of_sorted_cols < len(data.columns)
    df = data

    if num_of_sorted_cols==0:
        update_sample = df.sample(frac=frac)
        return update_sample

    columns_to_sort = [df.columns[i] for i in range(num_of_sorted_cols)]
    columns_not_sort = [df.columns[i] for i in range(num_of_sorted_cols, len(df.columns))]

    sorted_columns = pd.concat(([df[col].sort_values(ignore_index=True).reset_index(drop=True) for col in columns_to_sort]+[df[col] for col in columns_not_sort]), axis=1, ignore_index=True)
    sorted_columns.columns = df.columns
   
    update_sample = sorted_columns.sample(frac=frac)


    return update_sample



if __name__=="__main__":

    prepare_data(root="/home/meghdad/data/census/", filename="census.csv",
             cols=[
                'age','workclass','fnlwgt','education',
                'marital-status','occupation','relationship',
                'race','sex','capital-gain','capital-loss',
                'hours-per-week','native-country']
            )
    """
    prepare_data(root="/home/meghdad/data/forest/", filename="forest.csv",
                 cols = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology',
                'Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways',
                'Hillshade_9am','Hillshade_Noon','Hillshade_3pm',
                'Horizontal_Distance_To_Fire_Points']
            )

    prepare_data(root="/home/meghdad/data/DMV/", filename="Vehicle__Snowmobile__and_Boat_Registrations.csv",
                 cols = [
                'Record Type', 'Registration Class', 'State', 'County', 'Body Type',
                'Fuel Type', 'Reg Valid Date', 'Color', 'Scofflaw Indicator',
                'Suspension Indicator', 'Revocation Indicator']
            )
    """
    #LoadPermutedData(permute=False)
    #LoadPartlyPermutedIMDB()
