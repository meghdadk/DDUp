from ctgan import CTGANSynthesizer
from ctgan import load_demo
from ctgan import data
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics

def prepare_data(real, synthetic, target, cat_cols, num_cols):
    encoder = ce.BinaryEncoder(cols=cat_cols,return_df=True)
    encoded_df = encoder.fit_transform(real[cat_cols])
    encoded_synth = encoder.transform(synthetic[cat_cols])

    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(real[num_cols])
    x_scaled_synth = min_max_scaler.transform(synthetic[num_cols])
    normalized_df= pd.DataFrame(x_scaled,columns=num_cols)
    normalized_synth = pd.DataFrame(x_scaled_synth, columns=num_cols)



    X = pd.concat([encoded_df,normalized_df], axis=1)   #order: [category columns , numerical columns]
    X_synth = pd.concat([encoded_synth, normalized_synth], axis=1)

    enc = LabelEncoder()
    target_encoded = enc.fit_transform(real[target].values)
    target_encoded_synth = enc.transform(synthetic[target].values)


    X_train, X_test, y_train, y_test = train_test_split(X.values, target_encoded, test_size=0.3)


    return (X_train, y_train), (X_test, y_test), (X_synth.values, target_encoded_synth)




def benchmarking(modelpath, datapath, metapath):


    df, cat_cols = data.read_csv(csv_filename=datapath,meta_filename=metapath)   
    num_cols = [col for col in df.columns if col not in cat_cols]
    cat_cols.remove('label')



    real_data_acc = {'adaboost':[],'xgboost':[]}
    real_data_f1 = {'adaboost':[],'xgboost':[]}
    syn_data_acc = {'adaboost':[],'xgboost':[]}
    syn_data_f1 = {'adaboost':[],'xgboost':[]}

    for i in range(3):
        print (i)      
        synthetic = sample(modelpath,int(len(df)*0.7))


        (X_train, y_train), (X_test, y_test), (X_synth, y_synth) = prepare_data(df, synthetic, 'label', cat_cols, num_cols)



        abc = AdaBoostClassifier(n_estimators=50,learning_rate=1)
        model = abc.fit(X_train, y_train)

        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
            max_depth=1, random_state=0).fit(X_train, y_train)


        y_pred_adaboost = model.predict(X_test)
        y_pred_xgboost = clf.predict(X_test)

        f1_ada = f1_score(y_test, y_pred_adaboost)
        acc_ada = metrics.accuracy_score(y_test, y_pred_adaboost)

        f1_xg = f1_score(y_test, y_pred_xgboost)
        acc_xg = metrics.accuracy_score(y_test, y_pred_xgboost)
        
        real_data_acc['adaboost'].append(acc_ada)
        real_data_acc['xgboost'].append(acc_xg)
        real_data_f1['adaboost'].append(f1_ada)
        real_data_f1['xgboost'].append(f1_xg)



        del synthetic
        del clf
        del abc



        abc = AdaBoostClassifier(n_estimators=50,learning_rate=1)
        model = abc.fit(X_synth, y_synth)

        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
            max_depth=1, random_state=0).fit(X_synth, y_synth)


        y_pred_adaboost = model.predict(X_test)
        y_pred_xgboost = clf.predict(X_test)

        f1_ada = f1_score(y_test, y_pred_adaboost)
        acc_ada = metrics.accuracy_score(y_test, y_pred_adaboost)

        f1_xg = f1_score(y_test, y_pred_xgboost)
        acc_xg = metrics.accuracy_score(y_test, y_pred_xgboost)

        syn_data_acc['adaboost'].append(acc_ada)
        syn_data_acc['xgboost'].append(acc_xg)
        syn_data_f1['adaboost'].append(f1_ada)
        syn_data_f1['xgboost'].append(f1_xg)


    print ("Accuracy results:")
    print ("model   \t\t\treal data\t\tsynthetic data")
    print ("Adaboost\t\t\t{:.4f}\t\t{:.4f}".format(np.mean(real_data_acc['adaboost']),np.mean(syn_data_acc['adaboost'])))
    print ("Xgboost \t\t\t{:.4f}\t\t{:.4f}".format(np.mean(real_data_acc['xgboost']),np.mean(syn_data_acc['xgboost'])))



    print ("\n\nf1-measure results:")
    print ("model   \t\t\treal data\t\tsynthetic data")
    print ("Adaboost\t\t\t{:.4f}\t\t{:.4f}".format(np.mean(real_data_f1['adaboost']),np.mean(syn_data_f1['adaboost'])))
    print ("Xgboost \t\t\t{:.4f}\t\t{:.4f}".format(np.mean(real_data_f1['xgboost']),np.mean(syn_data_f1['xgboost'])))

    print (real_data_f1)
    print (syn_data_f1)
    del df
    del clf
    del abc


def prepare_updatedata(csv_filename, permute=True):
    census_data, discrete_columns = data.read_csv(csv_filename=csv_filename,meta_filename=meta_filename)

    if permute:
        columns_to_sort = census_data.columns

        sorted_columns = pd.concat([census_data[col].sort_values(ignore_index=True).reset_index(drop=True) for col in columns_to_sort], axis=1, ignore_index=True)
        sorted_columns.columns = census_data.columns

        update_sample = sorted_columns.sample(frac=0.2)
        del sorted_columns
    else:
        update_sample = data.sample(frac=0.2) 
    

    transfer_set = census_data.sample(frac=0.1)


    new_df = pd.concat([census_data,update_sample])
    new_df.to_csv('data/census_updated.csv',sep=',', header=True, index=None)
    del census_data
    del new_df

    update_sample.to_csv('data/update_batch.csv',sep=',', index=None)
    transfer_set.to_csv('data/transfer_set.csv',sep=',', index=None)


    return transfer_set, update_sample, discrete_columns


def update(pre_model, datafile):

    model = None
    with open(pre_model,'rb') as inp:
        model = pickle.load(inp)
    
    if not model:
        print ("failed loading the previous model {}!".format(pre_model))
        return None

 
    transfer_set, update_sample, discrete_columns = prepare_updatedata(csv_filename=datafile)

    train_data = pd.concat([update_sample,transfer_set]).sample(frac=1).reset_index(drop=True)

    #model.update(train_data, alpha=0.15,discrete_columns=discrete_columns,epochs=100)
    model.finetune(train_data,discrete_columns=discrete_columns,epochs=50)

    with open('models/census_01.pkl', 'wb') as outp:
        pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)

    del model
    del train_data



def distill(pre_model, datafile):

    model = None
    with open(pre_model,'rb') as inp:
        model = pickle.load(inp)
    
    if not model:
        print ("failed loading the previous model {}!".format(pre_model))
        return None

    #transfer_set, _, discrete_columns = prepare_updatedata(csv_filename=datafile)

    #model.distill2(transfer_set, 0.5, discrete_columns, epochs=100)
    model.distill0(epochs=5000)

    with open('models/census_distilled.pkl', 'wb') as outp:
        pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)



 
def train(csv_filename, meta_filename):

    census_data, discrete_columns = data.read_csv(csv_filename=csv_filename,meta_filename=meta_filename)


    ctgan = CTGANSynthesizer(epochs=50)
    ctgan.fit(census_data, discrete_columns)

    with open('models/census_00.pkl', 'wb') as outp:
        pickle.dump(ctgan, outp, pickle.HIGHEST_PROTOCOL)
    

    del ctgan


def sample(modelname, num_sample):
    model = None
    with open(modelname,'rb') as inp:
        model = pickle.load(inp)
    
    if model:
        #sample = model.sample(num_sample,condition_column='relationship', condition_value=[' Husband'])
        sample = model.sample(num_sample)
        return sample
    else:
        print ("failed loading the model!")
        return None


def compare_histograms(modelname, datafile, attributes, num_sample=30000):
    fsample = sample(modelname, num_sample)

    if fsample is None:
        return
    
    dataset, _ = data.read_csv(datafile)
    update_batch, _ = data.read_csv('data/update_batch.csv')
    dataset = pd.concat([dataset, update_batch])
    
    rsample = dataset.sample(num_sample)

    
    i=1
    for att in attributes:
        plt.subplot(len(attributes),2,i)
        plt.hist(rsample[att], bins=rsample[att].nunique())
        plt.title(att+'_real_sample')

        i += 1

        plt.subplot(len(attributes),2,i)
        plt.hist(fsample[att], bins=fsample[att].nunique())
        plt.title(att+'_fake_sample')

        i += 1


    plt.savefig('data/census_hist_compare.png')



if __name__ == "__main__":
    csv_filename="data/census.csv"
    meta_filename="data/census.json"
    #train(csv_filename, meta_filename)
    update(pre_model="models/census_00.pkl", datafile=csv_filename)
    #distill(pre_model="models/census_00.pkl", datafile=csv_filename)
    #samples = sample("models/census_00.pkl",45000)
    #samples.to_csv('data/census_generated.csv', sep=',', index=None, header=True)
    #compare_histograms('models/census_01.pkl','examples/csv/census.csv',['workclass','education','age'])
    #benchmarking('models/census_01.pkl', datapath='data/census_updated.csv', metapath='data/census.json')
