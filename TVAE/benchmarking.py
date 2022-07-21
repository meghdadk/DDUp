import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from ctgan import TVAESynthesizer
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import cross_entropy
import torch
from ctgan import load_demo
from ctgan import data
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time
import os
pd.options.mode.chained_assignment = None
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

def prepare_classification_data(real, synthetic, target, cat_cols, num_cols):
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

def prepare_regression_data(real, synthetic, target, cat_cols, num_cols):
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


    y_scaled = real[target].values.reshape(-1,1)
    y_scaled_synth = synthetic[target].values.reshape(-1,1)


    X_train, X_test, y_train, y_test = train_test_split(X.values, y_scaled, test_size=0.3)


    return (X_train, y_train), (X_test, y_test), (X_synth.values, y_scaled_synth)


def benchmarking(modelpath, datapath, metapath,num_try, label):


    df, cat_cols = data.read_csv(csv_filename=datapath,meta_filename=metapath)   
    num_cols = [col for col in df.columns if col not in cat_cols]
    cat_cols.remove(label)



    real_data_acc = {'adaboost':[],'xgboost':[]}
    real_data_f1 = {'adaboost':[],'xgboost':[]}
    syn_data_acc = {'adaboost':[],'xgboost':[]}
    syn_data_f1 = {'adaboost':[],'xgboost':[]}
    real_data_rocauc = {'adaboost':[],'xgboost':[]}
    syn_data_rocauc = {'adaboost':[],'xgboost':[]}

    for i in range(num_try):    
        print (i)
        synthetic = sample(modelpath,int(len(df)*0.7))
        (X_train, y_train), (X_test, y_test), (X_synth, y_synth) = prepare_classification_data(df, synthetic, label, cat_cols, num_cols)


        #Evaluate real data
        
        """
        abc = AdaBoostClassifier(n_estimators=50,learning_rate=1)
        model = abc.fit(X_train, y_train)
        y_pred_adaboost = model.predict(X_test)
        f1_ada = f1_score(y_test, y_pred_adaboost, average='micro')
        acc_ada = metrics.accuracy_score(y_test, y_pred_adaboost)
        real_data_acc['adaboost'].append(acc_ada)
        real_data_f1['adaboost'].append(f1_ada)
        """

        #clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, 
        #    max_depth=1, random_state=0).fit(X_train, y_train) 

        clf = XGBClassifier()
        clf.fit(X_train, y_train)

        y_pred_xgboost = clf.predict(X_test)
        y_pred_xgboost = [round(value) for value in y_pred_xgboost]
        #y_prob = clf.predict_proba(X_test)
        f1_xg = f1_score(y_test, y_pred_xgboost, average='micro')
        #acc_xg = metrics.accuracy_score(y_test, y_pred_xgboost)
        #rocauc = metrics.roc_auc_score(y_test, y_prob, multi_class='ovr')

        #real_data_acc['xgboost'].append(acc_xg)
        real_data_f1['xgboost'].append(f1_xg)
        #real_data_rocauc['xgboost'].append(rocauc)




        #Evaluate synthetic data
        """
        abc = AdaBoostClassifier(n_estimators=50,learning_rate=1)
        model = abc.fit(X_synth, y_synth)
        y_pred_adaboost = model.predict(X_test)
        f1_ada = f1_score(y_test, y_pred_adaboost, average='micro')
        acc_ada = metrics.accuracy_score(y_test, y_pred_adaboost)
        syn_data_acc['adaboost'].append(acc_ada)
        syn_data_f1['adaboost'].append(f1_ada)
        """


        clf = XGBClassifier()
        clf.fit(X_synth, y_synth)

        y_pred_xgboost = clf.predict(X_test)
        y_pred_xgboost = [round(value) for value in y_pred_xgboost]


        f1_xg = f1_score(y_test, y_pred_xgboost, average='micro')
        #acc_xg = metrics.accuracy_score(y_test, y_pred_xgboost)
        #rocauc = metrics.roc_auc_score(y_test, y_prob, multi_class='ovr')


        #syn_data_acc['xgboost'].append(acc_xg)
        syn_data_f1['xgboost'].append(f1_xg)
        #syn_data_rocauc['xgboost'].append(rocauc)




    """
    print ("Accuracy results:")
    print ("model   \t\t\treal data\t\tsynthetic data")
    print ("Adaboost\t\t\t{:.4f}\t\t{:.4f}".format(np.mean(real_data_acc['adaboost']),np.mean(syn_data_acc['adaboost'])))
    print ("Xgboost \t\t\t{:.4f}\t\t{:.4f}".format(np.mean(real_data_acc['xgboost']),np.mean(syn_data_acc['xgboost'])))

    """
    print (real_data_f1)
    print (syn_data_f1)

    print ("\n\nXgboost results:")
    print ("model     \t\t\treal data\t\tsynthetic data")
    #print ("Adaboost\t\t\t{:.4f}\t\t{:.4f}".format(np.mean(real_data_f1['adaboost']),np.mean(syn_data_f1['adaboost'])))
    print ("f1 score  \t\t\t{:.2f},{:.2f}\t\t{:.2f},{:.2f}".format(
           np.mean(real_data_f1['xgboost']),np.var(real_data_f1['xgboost']),
           np.mean(syn_data_f1['xgboost']), np.var(syn_data_f1['xgboost'])))
    #print ("auc_roc   \t\t\t{:.4f}\t\t{:.4f}".format(np.mean(real_data_rocauc['xgboost']),np.mean(syn_data_rocauc['xgboost'])))


def benchmark_multiple(models_path, metapath, num_try, label):
    models = []
    batches = []
    for file in os.listdir(models_path):
        if file.endswith('.pkl'):
            models.append(os.path.join(models_path,file))
        elif file.endswith('csv'):
            batches.append(os.path.join(models_path,file))
    models = np.sort(models)
    batches = np.sort(batches)


    assert len(models)==len(batches)

    print (models)
    print (batches)


    df = pd.DataFrame()

    for model, batch in zip(models, batches):
        df1, _ = data.read_csv(batch, metapath)
        df = pd.concat([df,df1])
        df.to_csv('temp.csv', header=True, index=None, sep=',')

        benchmarking(model, datapath='temp.csv', metapath=metapath, label=label, num_try=num_try)


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
    #update_batch, _ = data.read_csv('data/update_batch.csv')
    #dataset = pd.concat([dataset, update_batch])
    
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



def _loss_function(recon_x, x, sigmas, mu, logvar, output_info, factor):
    st = 0
    loss = []
    for column_info in output_info:
        for span_info in column_info:

            if span_info.activation_fn != 'softmax':
                ed = st + span_info.dim
                std = sigmas[st]
                eq = x[:, st] - torch.tanh(recon_x[:, st])
                loss.append((eq ** 2 / 2 / (std ** 2)).sum())
                loss.append(torch.log(std) * x.size()[0])
                st = ed

            else:
                ed = st + span_info.dim
                loss.append(cross_entropy(
                    recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
                st = ed

    assert st == recon_x.size()[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return sum(loss) * factor / x.size()[0], KLD / x.size()[0]



def Drift_Detection(model, old_data, update_batch, metafile, bootstrap=500, sample_size=64):

    with open(model, 'rb') as d:
        m = pickle.load(d)


    def offline_phase(m, s_old, sample_size, simulations, fig):
        t1 = time.time()
        avg_loss = []
        for i in range(simulations):

            df_sample = s_old.sample(n=sample_size,replace=True)

            train_data = m.transformer.transform(df_sample)
            real = torch.Tensor(torch.from_numpy(train_data.astype('float32'))).to(m._device)

            mu, std, logvar = m.encoder(real)
            eps = torch.randn_like(std)
            emb = eps * std + mu
            rec, sigmas = m.decoder(emb)
            loss_1, loss_2 = _loss_function(
                rec, real, sigmas, mu, logvar,
                m.transformer.output_info_list, m.loss_factor
            )
            loss = loss_1 + loss_2

            
            avg_loss.append(loss.item())
        
        t2 = time.time()

        if fig is not None:
            fig.hist(avg_loss, bins=int(len(avg_loss)/10), label='original')

        
        return np.mean(avg_loss), 2*np.var(avg_loss), t2-t1, fig

        
    def online_phase_sample(m, d_new, mean, threshold, batch, fig, label):
        t1 = time.time()


        losses = []
        for i in range(50):
            df_sample = d_new.sample(n=batch,replace=True)

            train_data = m.transformer.transform(df_sample)
            real = torch.Tensor(torch.from_numpy(train_data.astype('float32'))).to(m._device)


            mu, std, logvar = m.encoder(real)
            eps = torch.randn_like(std)
            emb = eps * std + mu
            rec, sigmas = m.decoder(emb)
            loss_1, loss_2 = _loss_function(
                rec, real, sigmas, mu, logvar,
                m.transformer.output_info_list, m.loss_factor
            )
            loss = loss_1 + loss_2
            
            loss = loss.item()
            losses.append(loss)


        if fig is not None:
            fig.hist(losses, bins=int(len(losses)/10), label=label)

        ind = []
        ood = []
        for loss in losses:
            stat = np.abs(loss - mean)
            if stat > threshold:
                ood.append(loss)
            else:
                ind.append(loss)


        print("mean: {:0.4f}".format(mean), "threshold: {:0.4f}".format(threshold))
        print("number of all samples: {}".format(len(losses)))
        print("number of ood detected: {}".format(len(ood)))
        print("number of ind detected: {}".format(len(ind)))
        if label == "ind":
            fpr = len(ood)/len(losses)
            print ("FPR = {:0.4f}".format(fpr))
            return fig, fpr
        elif label == "ood":
            fnr = len(ind)/len(losses)
            print ("FNR = {:0.4f}".format(fnr))
            return fig, fnr




    previous_data, discrete_columns = data.read_csv(csv_filename=old_data, meta_filename=metafile)
    test_OOD, discrete_columns = data.read_csv(csv_filename=update_batch, meta_filename=metafile)
    test_IND = previous_data.sample(n=len(test_OOD))
    print (previous_data.shape, test_IND.shape)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 16})

    fig, ax = plt.subplots()
    mean, threshold, time_off, ax = offline_phase(m, previous_data, sample_size=sample_size, simulations=bootstrap, fig=ax)
    fprs = []
    fnrs = []
    num_tuples = [i for i in range(1, 2000, 100)]
    for n in num_tuples:
        ax, rate = online_phase_sample(m, test_IND, mean, threshold, n, ax, 'ind')
        fprs.append(rate)
        ax, rate = online_phase_sample(m, test_OOD, mean, threshold, n, ax, 'ood')
        fnrs.append(rate)

    ax.legend(loc="upper left")
    fig.savefig('histograms.png')
    #print ("Offline time = {}\nonline time={}".format(time_off, time_on))

    plt.clf()
    fig, ax = plt.subplots()
    xi = list(range(len(fprs)))
    ax.plot(num_tuples, fprs, marker='o', linestyle='-', label='false positive rate')
    ax.plot(num_tuples, fnrs, marker='*', linestyle='--', label='false negative rate')
    ax.legend(loc="upper right")
    ax.set_xlabel('batch size',fontsize=20, fontweight='bold', fontfamily='Times New Roman')
    ax.set_ylabel('FPR/FNR rate',fontsize=20, fontweight='bold', fontfamily='Times New Roman')
    plt.title('TVAE | DMV',fontsize=22, fontweight='bold', fontfamily='Times New Roman')
    #ax.set_xticks(num_tuples)
    #ax.set_xticklabels(num_tuples, rotation=45)
    fig.savefig('errors.png')



if __name__ == "__main__":
    #compare_histograms('models/census_vae_00.pkl','data/census.csv',['workclass','education','age','marital-status'])
    #s = sample('models/forest_vae_00.pkl',num_sample=10)
    #print (s)
    #benchmarking('models/forest_vae_updated.pkl', datapath='data/covtype_updated.csv', metapath='data/covtype.json', label='label', num_try=5)
    Drift_Detection('dmv_results/models/dmv_00.pkl', old_data='data/dmv_small.csv', update_batch='data/update_batch_grperm.csv', metafile='data/dmv.json')
    #benchmark_multiple('models/','data/covtype.json', label='label',num_try=1)
