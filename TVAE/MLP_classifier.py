import torch as T
from torch import nn
import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score 
import random

#random.seed(0)
#np.random.seed(0)
#T.manual_seed(0)


device = T.device("cuda" if T.cuda.is_available() else "cpu")

class Net(T.nn.Module):
    def __init__(self, inputSize):
        super(Net, self).__init__()
        # Number of input features is 12.
        self.layer_1 = nn.Linear(inputSize, 128) 
        self.layer_2 = nn.Linear(128, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, 1) 
    
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.relu(self.layer_3(x))
        x = self.batchnorm3(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x



class Classifier():
    def __init__(self):
        self.model = None
        self.encoder = None
        self.normalizer = None
        self.cat_cols = None
        self.num_cols = None
        self.label_enc = None
        self.bs = 256
        self.lr = 1e-2
        self.epochs = 100
        self.num_worker = 2


    def prepare_data(self,data, target, cat_cols, num_cols):

        encoder = ce.BinaryEncoder(cols=cat_cols,return_df=True)
        encoded_df = encoder.fit_transform(data[cat_cols])

        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(data[num_cols])
        normalized_df= pd.DataFrame(x_scaled,columns=num_cols)



        data = pd.concat([encoded_df,normalized_df], axis=1)   #order: [category columns , numerical columns]


        enc = LabelEncoder()
        target_encoded = enc.fit_transform(target)



        x_tensor = T.tensor(data.values)
        y_tensor = T.tensor(target_encoded.reshape(-1,1))




        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.encoder = encoder
        self.normalizer = min_max_scaler
        self.label_enc = enc


        del data
        del normalized_df
        del target_encoded

        return x_tensor.float(), y_tensor.float()

    def fit(self, X, y, cat_ix, num_ix, batch_size=None, learning_rate=None, num_epoch=None):

        self.bs = batch_size if batch_size else self.bs
        self.lr = learning_rate if learning_rate else self.lr
        self.epochs = epoch if num_epoch else self.epochs
 

        x_tensor, y_tensor = self.prepare_data(X, y, cat_ix, num_ix)


        train_dataset = T.utils.data.TensorDataset(
            x_tensor, y_tensor
        )  
        dataloader = T.utils.data.DataLoader(
            train_dataset,
            batch_size=self.bs,
            shuffle=True,
            num_workers=self.num_worker,
        )


        self.model = Net(inputSize=x_tensor.shape[1])

        criterion = nn.BCEWithLogitsLoss()
        optimizer = T.optim.Adam(self.model.parameters(), lr=self.lr)
        decay_rate = 0.96
        scheduler = T.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=decay_rate
            )

        
        self.model = self.model.to(device)
        self.model.train()
        _iter = 0
        for epoch in range(self.epochs):
            epoch_loss = 0
            epoch_acc = 0
            for mini_batch, labels in dataloader:
                mini_batch = mini_batch.to(device)
                labels = labels.to(device)
                self.model.zero_grad()

                outputs = self.model(mini_batch)

                loss = criterion(outputs, labels)
                acc = self.binary_acc(outputs, labels)

                loss.backward()
                optimizer.step()
                 

                epoch_loss += loss
                epoch_acc += acc
                _iter += 1

            print(f'Epoch: {epoch+0:03} | Iter: {_iter+0:04} | Loss: {epoch_loss/len(dataloader):.9f} | Acc: {epoch_acc/len(dataloader):.3f}')
            
            #scheduler.step()

        self.model.eval()

        return self



    def eval(self, X,y, cat_ix, num_ix, batch_size=None,):

        self.bs = batch_size if batch_size else self.bs

        x_tensor, y_tensor = self.prepare_data(X,y, cat_ix, num_ix)

        test_dataset = T.utils.data.TensorDataset(
            x_tensor, y_tensor
        )  
        dataloader = T.utils.data.DataLoader(
            test_dataset,
            batch_size=self.bs,
            shuffle=True,
            num_workers=self.num_worker,
        )

        self.model = self.model.to(device)
       
        predictions = []
        self.model.eval()
        accuracy = 0
        f1 = 0
        with T.no_grad():
            for mini_batch, labels in dataloader:
                mini_batch = mini_batch.to(device)
                labels = labels.to(device)

                outputs = self.model(mini_batch)
                outputs = T.sigmoid(outputs)
                pred_labels = T.round(outputs)
                predictions.append(pred_labels.cpu().numpy())

                accuracy += self.binary_acc(pred_labels, labels)
                f1 += self.binary_f1(pred_labels, labels)

        
        print ("accuracy = {}, f1 = {}".format(accuracy/len(dataloader), f1/len(dataloader)))

        return accuracy/len(dataloader), f1/len(dataloader)


    def binary_acc(self,y_pred, y_true):
        y_pred_tag = T.round(T.sigmoid(y_pred))

        correct_results_sum = (y_pred_tag == y_true).sum().float()
        acc = correct_results_sum/y_true.shape[0]
        acc = T.round(acc * 100)
        
        return acc.item()


    def binary_f1(self,y_pred, y_true): 
        f1 = f1_score(y_true.squeeze().tolist(), y_pred.squeeze().tolist())
        return f1


def load_dataset(full_path, sep, label, train_path=None, test_path=None):
    if full_path is not None:
        df = pd.read_csv(full_path,sep=sep)
        #df1 = pd.read_csv('data/update_batch.csv',sep=sep)
        #df = pd.concat([df,df1],ignore_index=True, axis=0)
        df = df.dropna()
        msk = np.random.rand(len(df)) < 0.8
        train, test = df[msk].reset_index().drop(["index"], axis=1), df[~msk].reset_index().drop(["index"], axis=1)
        train.to_csv('data/census_train.csv',sep=',', index=None, header=True)
        test.to_csv('data/census_test.csv',sep=',', index=None, header=True)
    if train_path is not None:
        train = pd.read_csv(train_path, sep=sep)
        train.dropna()
    if test_path is not None:
        test = pd.read_csv(test_path, sep=sep)
        test.dropna()


    x_columns = list(train.columns)
    x_columns.remove(label)

    X_train, y_train, X_test, y_test = train[x_columns], train[label], test[x_columns], test[label]
    cat_ix = X_train.select_dtypes(include=['object', 'bool']).columns
    num_ix = X_train.select_dtypes(include=['int64', 'float64']).columns


    return X_train, y_train, X_test, y_test, cat_ix, num_ix


if __name__=="__main__":
    accuracies = []
    f1s = []
    for i in range(5):
        X_train, y_train, X_test, y_test, cat_ix, num_ix = load_dataset(full_path=None, train_path='data/census_generated.csv',sep=',',label='label',test_path='data/census_test.csv')
        #X_train, y_train, X_test, y_test, cat_ix, num_ix = load_dataset(full_path='data/census.csv',sep=',',label='label',)
        clf = Classifier()
        clf.fit(X_train,y_train, cat_ix, num_ix)
        acc, f1 = clf.eval(X_test, y_test, cat_ix, num_ix)
        accuracies.append(acc)
        f1s.append(f1)

    print (accuracies)
    print (f1s)

