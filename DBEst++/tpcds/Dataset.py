import pandas as pd
import numpy as np
from itertools import groupby
from collections import Counter
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class Normalizer:
    def __init__(self):
        self.mean = None
        self.width = None
        self.min = None
        self.max = None
        
    def get_mean(self):
        return self.mean
    
    def get_width(self):
        return self.width

    def get_min(self):
        return self.min

    def get_max(self):
        return self.max

class Data:
    def __init__(self, data_path, train_file, test_file, update_path, transfer_set, header=[], x_attributes=[], y_attributes=[], sep=','):
        """
        data_path: the root directory where data files are located
        train_file: the name of the main file used to build the base model
        test_file: the test set to evaluate log likelihood. If not applicable set to None
        
        """
        self.data_path = data_path
        self.train_file = train_file
        self.test_file = test_file
        self.update_path = update_path
        self.delimiter = sep
        self.header = [h.tolower() for h in header]
        self.x_attributes = [x.lower() for x in x_attributes]
        self.y_attributes = [y.lower() for y in y_attributes]
        self.encoders = {}
        self.normalizing_stats = {}
        self.FTs = {}
        self.transfer_set = transfer_set

    def train_update_split(self,table, haveheader, permute, convert_to_timestamp=False, percentage=0.2, num_update_chunks=10):
        path = os.path.join(self.data_path, table)
        if haveheader:
            data = pd.read_csv(path, sep=self.delimiter)
            cols = [i.lower() for i in list(data.columns)]
            data.columns = cols
        else:
            if self.header==[]:
                print ("header list is not provided")
                return
            data = pd.read_csv(path, header=None, sep=self.delimiter, names=self.header)

        data = data[self.x_attributes+self.y_attributes]
        data = data.dropna()
        if convert_to_timestamp:
            data[sort]=pd.to_datetime(data[sort]).astype(int) / 10**9

        if permute:
            columns_to_sort = data.columns

            sorted_columns = pd.concat([data[col].sort_values(ignore_index=True).reset_index(drop=True) for col in columns_to_sort], axis=1, ignore_index=True)
            sorted_columns.columns = data.columns
   
            update_sample = sorted_columns.sample(frac=percentage) 
            print ("permuted dataset:")
            print (update_sample)
        else:
            update_sample = data.sample(frac=percentage) #to shuffle

        data = pd.concat([data,update_sample])

        self.train_file = os.path.join(self.data_path,'train_set.csv')
        self.test_file = os.path.join(self.data_path,'test_set.csv')
        self.transfer_set = os.path.join(self.data_path,'transfer_set.csv')
        self.update_path = os.path.join(self.data_path, "update_batches")        

        split_index = int(data.shape[0]-update_sample.shape[0])            
        train_data = data.iloc[:split_index+1, :]
        update_data = data.iloc[split_index:, :]
        train_data = train_data.sample(frac=1) #to shuffle
        train_set, test_set = train_test_split(train_data, test_size=0.1)
        #transfer_data = stratified_sample(train_set, self.x_attributes, size=5000, seed=123, keep_index= True)#train_set.sample(frac=0.9)

        update_chunks = np.array_split(update_data,num_update_chunks)
        
        train_set.to_csv(self.train_file,sep=self.delimiter, index=None)
        test_set.to_csv(self.test_file,sep=self.delimiter, index=None) 
        #transfer_data.to_csv(self.transfer_set, sep=self.delimiter, index=None)

        if not os.path.exists(self.update_path):
            os.mkdir(os.path.join(self.update_path))

        for i,chunk in enumerate(update_chunks):
            train, test = train_test_split(chunk, test_size=0.1)
            train.to_csv(os.path.join(self.update_path,'update{}.csv'.format(str(i+1).zfill(2))), sep=self.delimiter, index=None)
            test.to_csv(os.path.join(self.update_path,'update{}_test.csv'.format(str(i+1).zfill(2))), sep=self.delimiter, index=None)

    def read_data(self, file, haveheader):
        if haveheader:
            data = pd.read_csv(file, sep=self.delimiter)
        else:
            if self.header==[]:
                print ("header list is not provided")
                return
            data = pd.read_csv(file, header=None, sep=self.delimiter, names=self.header)

        data.columns = [x.lower() for x in data.columns]
        x_values = {}
        y_values = {}
        for att in self.x_attributes:
            if data[att].dtype!=np.str:
                data[att] = data[att].astype(np.str)
            x_values[att] = data[att].str.lower().tolist()
        for att in self.y_attributes:
            y_values[att] = data[att].tolist()

        return x_values, y_values

    def create_encoders(self, data):
        
        for key in data.keys():
            encoder = OneHotEncoder(categories='auto')        
            encoder.fit(np.asarray(data[key]).reshape(-1,1))
            self.encoders[key] = encoder

    def create_frequency_tables(self, data):
        for key in data.keys():
            ft = Counter(data[key])
            self.FTs[key] = ft

    def get_normalizing_stats(self, data=None, min_max=None):
        if min_max is not None:
            for key in min_max:
                _mean = min_max[key][0]
                _width = min_max[key][1]
                norm = Normalizer()
                norm.mean = _mean
                norm.width = _width
                norm.min = min_max[key][0]
                norm.max = min_max[key][1]
                self.normalizing_stats[key] = norm
                
        else:
            for key in data.keys():
                _mean = (np.min(data[key]) + np.max(data[key]))/2
                _width = (np.max(data[key]) - np.min(data[key]))
                norm = Normalizer()
                norm.mean = _mean
                norm.width = _width
                norm.min = np.min(data[key])
                norm.max = np.max(data[key])
                self.normalizing_stats[key] = norm
        
    def normalize(self, data):
        normalized_data = {}
        for key in data.keys():
            s = self.normalizing_stats[key]
            normed = [(x-s.mean)/s.width * 2 for x in data[key]]
            normalized_data[key] = normed
            
        return normalized_data

    def denormalize(self, data):
        denormalized_data = {}
        for key in data.keys():
            s = self.normalizing_stats[key]
            denormed = [0.5 * s.width * x + s.mean for x in data[key]]
            denormalized_data[key] = denormed
            
        return normalized_data

    



# the functions:
def stratified_sample(df, strata, size=None, seed=None, keep_index= True):
    '''
    It samples data from a pandas dataframe using strata. These functions use
    proportionate stratification:
    n1 = (N1/N) * n
    where:
        - n1 is the sample size of stratum 1
        - N1 is the population size of stratum 1
        - N is the total population size
        - n is the sampling size
    Parameters
    ----------
    :df: pandas dataframe from which data will be sampled.
    :strata: list containing columns that will be used in the stratified sampling.
    :size: sampling size. If not informed, a sampling size will be calculated
        using Cochran adjusted sampling formula:
        cochran_n = (Z**2 * p * q) /e**2
        where:
            - Z is the z-value. In this case we use 1.96 representing 95%
            - p is the estimated proportion of the population which has an
                attribute. In this case we use 0.5
            - q is 1-p
            - e is the margin of error
        This formula is adjusted as follows:
        adjusted_cochran = cochran_n / 1+((cochran_n -1)/N)
        where:
            - cochran_n = result of the previous formula
            - N is the population size
    :seed: sampling seed
    :keep_index: if True, it keeps a column with the original population index indicator
    
    Returns
    -------
    A sampled pandas dataframe based in a set of strata.
    Examples
    --------
    >> df.head()
        id  sex age city 
    0    123 M   20  XYZ
    1    456 M   25  XYZ
    2    789 M   21  YZX
    3    987 F   40  ZXY
    4    654 M   45  ZXY
    ...
    # This returns a sample stratified by sex and city containing 30% of the size of
    # the original data
    >> stratified = stratified_sample(df=df, strata=['sex', 'city'], size=0.3)
    Requirements
    ------------
    - pandas
    - numpy
    '''
    population = len(df)
    size = __smpl_size(population, size)
    tmp = df[strata]
    tmp['size'] = 1
    tmp_grpd = tmp.groupby(strata).count().reset_index()
    tmp_grpd['samp_size'] = round(size/population * tmp_grpd['size']).astype(int)

    # controlling variable to create the dataframe or append to it
    first = True 
    for i in range(len(tmp_grpd)):
        # query generator for each iteration
        qry=''
        for s in range(len(strata)):
            stratum = strata[s]
            value = tmp_grpd.iloc[i][stratum]
            n = tmp_grpd.iloc[i]['samp_size']

            if type(value) == str:
                value = "'" + str(value) + "'"
            
            if s != len(strata)-1:
                qry = qry + stratum + ' == ' + str(value) +' & '
            else:
                qry = qry + stratum + ' == ' + str(value)
        
        # final dataframe
        if first:
            stratified_df = df.query(qry).sample(n=int(n), random_state=seed).reset_index(drop=(not keep_index))
            first = False
        else:
            tmp_df = df.query(qry).sample(n=int(n), random_state=seed).reset_index(drop=(not keep_index))
            stratified_df = stratified_df.append(tmp_df, ignore_index=True)
    
    return stratified_df



def stratified_sample_report(df, strata, size=None):
    '''
    Generates a dataframe reporting the counts in each stratum and the counts
    for the final sampled dataframe.
    Parameters
    ----------
    :df: pandas dataframe from which data will be sampled.
    :strata: list containing columns that will be used in the stratified sampling.
    :size: sampling size. If not informed, a sampling size will be calculated
        using Cochran adjusted sampling formula:
        cochran_n = (Z**2 * p * q) /e**2
        where:
            - Z is the z-value. In this case we use 1.96 representing 95%
            - p is the estimated proportion of the population which has an
                attribute. In this case we use 0.5
            - q is 1-p
            - e is the margin of error
        This formula is adjusted as follows:
        adjusted_cochran = cochran_n / 1+((cochran_n -1)/N)
        where:
            - cochran_n = result of the previous formula
            - N is the population size
    Returns
    -------
    A dataframe reporting the counts in each stratum and the counts
    for the final sampled dataframe.
    '''
    population = len(df)
    size = __smpl_size(population, size)
    tmp = df[strata]
    tmp['size'] = 1
    tmp_grpd = tmp.groupby(strata).count().reset_index()
    tmp_grpd['samp_size'] = round(size/population * tmp_grpd['size']).astype(int)
    return tmp_grpd


def __smpl_size(population, size):
    '''
    A function to compute the sample size. If not informed, a sampling 
    size will be calculated using Cochran adjusted sampling formula:
        cochran_n = (Z**2 * p * q) /e**2
        where:
            - Z is the z-value. In this case we use 1.96 representing 95%
            - p is the estimated proportion of the population which has an
                attribute. In this case we use 0.5
            - q is 1-p
            - e is the margin of error
        This formula is adjusted as follows:
        adjusted_cochran = cochran_n / 1+((cochran_n -1)/N)
        where:
            - cochran_n = result of the previous formula
            - N is the population size
    Parameters
    ----------
        :population: population size
        :size: sample size (default = None)
    Returns
    -------
    Calculated sample size to be used in the functions:
        - stratified_sample
        - stratified_sample_report
    '''
    if size is None:
        cochran_n = round(((1.96)**2 * 0.5 * 0.5)/ 0.02**2)
        n = round(cochran_n/(1+((cochran_n -1) /population)))
    elif size >= 0 and size < 1:
        n = round(population * size)
    elif size < 0:
        raise ValueError('Parameter "size" must be an integer or a proportion between 0 and 0.99.')
    elif size >= 1:
        n = size
    return n





