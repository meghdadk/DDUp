from ctgan import TVAESynthesizer
from ctgan import load_demo
from ctgan import data
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time


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

def prepare_driftdata(csv_filename, metafile):

    if csv_filename is None:
        transfer_set, discrete_columns = data.read_csv(csv_filename='data/transfer_set.csv', meta_filename=metafile)
        update_sample, _ = data.read_csv(csv_filename='data/update_batch.csv', meta_filename=metafile)

        return transfer_set, update_sample, discrete_columns

    census_data, discrete_columns = data.read_csv(csv_filename=csv_filename,meta_filename=metafile)
    
    test_IND = GetPermutedData(census_data, 0.5, False)

    test_OOD = pd.DataFrame()
    for i in range(1, 6):
         part = GetPartlyPermutedData(census_data, 0.5/5, i)
         test_OOD = pd.concat([test_OOD, part])

    transfer_set = census_data.sample(frac=0.1)


    test_IND.to_csv('data/update_batch_noperm.csv',sep=',', index=None)
    test_OOD.to_csv('data/update_batch_grperm.csv',sep=',', index=None)
    transfer_set.to_csv('data/transfer_set.csv',sep=',', index=None)




def prepare_updatedata(csv_filename, metafile, permute=True):

    if csv_filename is None:
        transfer_set, discrete_columns = data.read_csv(csv_filename='data/transfer_set.csv', meta_filename=metafile)
        update_sample, _ = data.read_csv(csv_filename='data/update_batch.csv', meta_filename=metafile)

        return transfer_set, update_sample, discrete_columns

    census_data, discrete_columns = data.read_csv(csv_filename=csv_filename,meta_filename=metafile)

    update_sample = GetPermutedData(census_data, 0.5, permute)

    transfer_set = census_data.sample(frac=0.1)


    new_df = pd.concat([census_data,update_sample])
    new_df.to_csv(csv_filename.replace('.csv','_updated.csv'),sep=',', header=True, index=None)
    del census_data
    del new_df

    update_sample.to_csv('data/update_batch.csv',sep=',', index=None)
    transfer_set.to_csv('data/transfer_set.csv',sep=',', index=None)


    return transfer_set, update_sample, discrete_columns


def update(pre_model, datafile, metafile, _type, dataset):

    model = None
    with open(pre_model,'rb') as inp:
        model = pickle.load(inp)
    
    if not model:
        print ("failed loading the previous model {}!".format(pre_model))
        return None

 
    transfer_set, update_sample, discrete_columns = prepare_updatedata(csv_filename=datafile, metafile=metafile)

    train_data = update_sample#pd.concat([update_sample,transfer_set]).sample(frac=1).reset_index(drop=True)


    if _type=="distill":
        train_data = pd.concat([update_sample, transfer_set]).sample(frac=1).reset_index(drop=True)
        t1 = time.time()
        model.update_decoder(train_data, alpha=0.2,discrete_columns=discrete_columns)
        t2 = time.time()
        print ("updating via distillation took {} seconds".format(t2-t1))
        with open('models/'+dataset+'_updated.pkl', 'wb') as outp:
            pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)

    
    elif _type=="finetune":
        t1 = time.time()
        model.finetune(train_data,discrete_columns=discrete_columns)
        t2 = time.time()
        print ("updating via finetune took {} seconds".format(t2-t1))
        with open('models/'+dataset+'_finetuned.pkl', 'wb') as outp:
            pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)

    elif _type=='retrain':
        train_data = pd.concat([update_sample,transfer_set]).sample(frac=1).reset_index(drop=True)
        #train_data, discrete_columns = data.read_csv(csv_filename='data/covtype_updated.csv',meta_filename=metafile)
        t1 = time.time()
        model.fit(train_data,discrete_columns=discrete_columns,retrain=True)
        t2 = time.time()
        print ('retraining on {} took {} seconds'.format(datafile, t2-t1))
        with open('models/'+dataset+'_retrain_updatebatch_and_transferset.pkl', 'wb') as outp:
            pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)


    del model
    del transfer_set
    del update_sample



def update_multiple(pre_model, datafile, metafile, _type, dataset):
 
    transfer_set, update_sample, discrete_columns = prepare_updatedata(csv_filename=datafile, metafile=metafile)


    batchs = np.array_split(update_sample, 5 ,axis=0)
    for i, batch in enumerate(batchs):
        batch.to_csv('models/update_batch{}.csv'.format(i),header=True, index=None, sep=',')

    for i, update_batch in enumerate(batchs):

        model = None
        with open(pre_model,'rb') as inp:
            model = pickle.load(inp)
        
        if not model:
            print ("failed loading the previous model {}!".format(pre_model))
            return None

        

        if _type=="distill":
            train_data = pd.concat([update_batch, transfer_set]).sample(frac=1).reset_index(drop=True)
            t1 = time.time()
            model.update_decoder(train_data, alpha=0.5,discrete_columns=discrete_columns,epochs=None)
            t2 = time.time()
            print ("updating via distillation took {} seconds".format(t2-t1))
            next_model_name = "models/"+dataset+'_updated'+str(i)+'.pkl'
            with open(next_model_name, 'wb') as outp:
                pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)

        
        elif _type=="finetune":
            t1 = time.time()
            model.finetune(update_batch,discrete_columns=discrete_columns)
            t2 = time.time()
            print ("updating via finetune took {} seconds".format(t2-t1))
            next_model_name = "models/"+dataset+'_finetuned'+str(i)+'.pkl'
            with open(next_model_name, 'wb') as outp:
                pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)

        elif _type=='retrain':
            train_data = pd.concat([update_batch,transfer_set]).sample(frac=1).reset_index(drop=True)
            #train_data, discrete_columns = data.read_csv(csv_filename='data/covtype_updated.csv',meta_filename=metafile)
            t1 = time.time()
            model.fit(train_data,discrete_columns=discrete_columns,retrain=True)
            t2 = time.time()
            print ('retraining on {} took {} seconds'.format(datafile, t2-t1))
            next_model_name = "models/"+dataset+'_AggTrain'+str(i)+'.pkl'
            with open(next_model_name, 'wb') as outp:
                pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)


        new_small_tr = update_batch.sample(frac=0.1)
        transfer_set = pd.concat([transfer_set,new_small_tr]).sample(frac=1)
        pre_model = next_model_name


    del model
    del transfer_set
    del update_sample


def distill(pre_model, datafile, metafile, dataset):

    model = None
    with open(pre_model,'rb') as inp:
        model = pickle.load(inp)
    
    if not model:
        print ("failed loading the previous model {}!".format(pre_model))
        return None

    transfer_set, _, discrete_columns = prepare_updatedata(csv_filename=datafile, metafile=metafile)

    t1 = time.time()
    model.distill_decoder(epochs=10000)
    #model.fit(transfer_set, discrete_columns, retrain=True)
    t2 = time.time()

    print ("distilling took {} seconds".format(t2-t1))
    with open('models/'+dataset+'_distilled_ce.pkl', 'wb') as outp:
        pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)

 
def train(csv_filename, meta_filename, dataset):

    census_data, discrete_columns = data.read_csv(csv_filename=csv_filename,meta_filename=meta_filename)


    tvae = TVAESynthesizer()
    t1 = time.time()
    tvae.fit(census_data, discrete_columns)
    t2 = time.time()


    print ("Training took {} seconds".format(t2-t1))
    with open('models/'+dataset+'.pkl', 'wb') as outp:
        pickle.dump(tvae, outp, pickle.HIGHEST_PROTOCOL)
    

    del tvae



if __name__ == "__main__":
    train(csv_filename="data/job_data/title_until_part1.csv", meta_filename="data/job_data/imdb.json", dataset='imdb')
    #update(pre_model="models/dmv_00.pkl", datafile=None, metafile= "data/dmv.json",_type='retrain', dataset='dmv')
    #update_multiple(pre_model="forest_results/models/forest_vae_00.pkl", datafile=None, metafile= "data/covtype.json",_type='distill', dataset='forest')
    #distill(pre_model="models/dmv_00.pkl", datafile=None, metafile="data/dmv.json",dataset='dmv')
    #prepare_driftdata(csv_filename='data/dmv_small.csv', metafile='data/dmv.json')
