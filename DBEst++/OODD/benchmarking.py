from MDN import DenMDN
from MDN import MDN
from Dataset import Data
from sqlParser import Parser
from multiprocessing import Process, Queue, Lock, Manager

import scipy
import numpy as np
import pandas as pd
import pandasql as ps
import torch.nn as nn
import torch
import math
import dill
import os
import time
import shutil
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
dill._dill._reverse_typemap['ClassType'] = type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


lock = Lock()


class Benchmark:
    def __init__(self, category_att, range_att, num_queries=1000):
        self.num_queries = num_queries
        self.queries = []
        self.category_att = category_att
        self.range_att = range_att
        self.results = {}


    def generate_queries(self, range, categories, type, agg, range1=None):
        template = "select {}({}) from {} where {} >= {} and {} <= {} and {} = '{}'"
        i = 0
        while i < self.num_queries:
            
            if type=="olddata":
                i1 = int(np.random.uniform(range[0], range[1]))
                i2 = int(np.random.uniform(range[0], range[1]))
            elif type=="alldata":
                i1 = int(np.random.uniform(range[0], range[1]))
                i2 = int(np.random.uniform(range[0], range[1]))
            elif type=="newdata":
                i1 = int(np.random.uniform(range[0], range[1]))
                i2 = int(np.random.uniform(range[0], range[1]))
                
                
            if abs(i1 - i2) > 1:
                ub = np.max([i1,i2])
                lb = np.min([i1,i2])
                cat = np.random.choice(categories)
                q = template.format(agg, self.range_att, 'XXXTOBEREPLACEDXXX', self.range_att, lb, self.range_att, ub, self.category_att, cat)
                self.queries.append(q)
                i = i+1
       
    def calculate_groundtruth(self, DATADF):

        if self.queries==None:
            print ("No query found!")
            return

        q = Queue()
        for i, query in enumerate(self.queries):
            q.put((i,query))

        manager = Manager()
        self.results = manager.dict()
        
        workers = []
        for i in range(30):
            p = Process(target=self.run_query, args=(q, DATADF))
            p.start()
            workers.append(p)
            #q = q.replace('XXXTOBEREPLACEDXXX','DATADF')
            #re = ps.sqldf(q,locals())
            #results[i] = re.iloc[0][0]
        
        for p in workers:
            p.join()
        print ("threads finished")

        return self.results

    def run_query(self, q, DATADF):
        while not q.empty():
            i, query = q.get()
            query = query.replace('XXXTOBEREPLACEDXXX','DATADF')
            re = ps.sqldf(query,locals())
            result = re.iloc[0][0]
            with lock:
                self.results[i] = result

    def relative_error(self,gt, pred):

        errors = []
        for key in gt.keys():
            y1 = gt[key]
            y2 = pred[key]
            err = abs(y2-y1)/y1
            errors.append(err)
        
        relative_error = np.mean(errors)
        print (relative_error)
        return relative_error


def cal_count(probs, step, frequencies):

    sub_areas = probs * step

    integral = np.sum(sub_areas[:, 1:-1], axis=1)

    integral = np.add(integral, sub_areas[:,0]*0.5)
    integral = np.add(integral, sub_areas[:,-1]*0.5)
    
    _count = integral * np.asarray(frequencies[list(frequencies.keys())[0]]).reshape(-1,1)

    return _count

def cal_sum(probs, regs, step, frequencies):
    product = np.multiply(probs, regs)

    sub_areas = product * step

    integral = np.sum(sub_areas[:, 1:-1], axis=1)
    integral = np.add(integral, sub_areas[:,0]*0.5)
    integral = np.add(integral, sub_areas[:,-1]*0.5)
    
    _sum = integral * np.asarray(frequencies[list(frequencies.keys())[0]]).reshape(-1,1)

    return _sum

def cal_avg(probs, regs, step, frequencies):
    _count = cal_count(probs, step, frequencies)
    _sum = cal_sum(probs, regs, step, frequencies)
    _avg = _sum/_count

    return _avg

def cal_err(reals, predictions, _type, agg, model_num):
    result_file = os.path.join("benchmark/result_"+ agg + str(model_num).zfill(2)+'.csv')
    if _type == "q-error":
        err = lambda r, p: np.max([r/p,p/r])
    if _type == "relative-error":
        err = lambda r, p: 100 * abs((r-p)/r)
    with open(result_file, 'w') as res:
        res.write("err, estimated, true\n")
        selected_queries = []
        if model_num == 0:
            errors = []
            for query_num, key in enumerate(reals.keys()):
                if reals[key]>0:
                    e = err(reals[key], predictions[key])
                    #print (reals[key], predictions[key], e)
                    errors.append(e)
                    res.write(str(e)+','+str(predictions[key])+','+str(reals[key])+'\n')
                    selected_queries.append(query_num)
            print ("mean {} = {:.4f}, median = {:.4f}, 95th={:.4f}, 99th={:.4f}, max={:.4f}".format(_type,
                np.mean(errors),np.median(errors), np.percentile(np.array(errors),95),
                np.percentile(np.array(errors),99),np.max(errors))
            )
            print ("{:.4f}".format(np.mean(errors)))
            print ("{:.4f}".format(np.median(errors)))
            print ("{:.4f}".format(np.percentile(np.array(errors),95)))
            print ("{:.4f}".format(np.percentile(np.array(errors),99)))
            print ("{:.4f}".format(np.max(errors)))
        else:
            errors = []
            for query_num, key in enumerate(reals.keys()):
                if query_num in selected_queries:
                    e = err(reals[key], predictions[key])
                    errors.append(e)
                    res.write(str(e)+','+str(predictions[key])+','+str(reals[key])+'\n')

            print ("mean {} = {}, median = {}, 95th={}, 99th={}, max={}".format(_type,
                np.mean(errors),np.median(errors), np.percentile(np.array(errors),95),
                np.percentile(np.array(errors),99),np.max(errors))
            )

def gaussian_probability(sigma, mu, data):
    data = data.unsqueeze(1).expand_as(sigma)
    ret = (
        1.0
        / math.sqrt(2 * math.pi)
        * torch.exp(-0.5 * ((data - mu) / sigma) ** 2)
        / sigma
    )
    return torch.prod(ret, 2)

def denormalize(x_point, mean, width):
    return 0.5 * width * x_point + mean

def LL(pi, sigma, mu, target, device, reduction='mean'):
    """Calculates the error, given the MoG parameters and the target
    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """
    softmax = nn.Softmax(dim=1)
    pi = softmax(pi)
    
    prob = pi * gaussian_probability(sigma, mu, target)

    ll = torch.log(torch.sum(prob, dim=1)).to(device)
    if reduction == None:
        return ll
    else:
        return torch.mean(ll)

def test_for_drift(model, main_data, test_IND, test_OOD, simulations, sample):
    
    def offline_phase(m, xs, ys, bootstraps, sample_size, fig):
        tensor_xs = torch.from_numpy(xs[m.dataset.x_attributes[0]].astype(np.float32)) 
        y_points = np.asarray(ys[m.dataset.y_attributes[0]]).reshape(-1,1)
        tensor_ys = torch.from_numpy(y_points.astype(np.float32))
        
        t1 = time.time()
        avg_lls = []
        m.model = m.model.to(device)
        for i in range(bootstraps):
            idx = np.random.randint(len(tensor_xs), size=sample_size)
            idx_tensor = torch.from_numpy(idx)
            sample_x = torch.index_select(tensor_xs,0,idx_tensor)
            sample_y = torch.index_select(tensor_ys,0,idx_tensor)
            sample_x = sample_x.to(device)
            sample_y = sample_y.to(device)
            pi, sigma, mu = m.model(sample_x)
            ll = LL(pi, sigma, mu, sample_y, device)
            avg_lls.append(ll.item())
        mean = np.mean(avg_lls)
        threshold = mean - 2*np.var(avg_lls)
        t2 = time.time()

        if fig is not None:
            height, bins, patches = fig.hist(avg_lls, bins=int(len(avg_lls)/10), label='sampling distribution')
            #fig.fill_betweenx([0, height.max()], mean-threshold, mean+threshold, color='gray', alpha=0.9)

        return mean, threshold, t2-t1, fig

    def online_phase_sample(m, xs, ys, mean, threshold, batch, label, fig):

        tensor_xs = torch.from_numpy(xs[m.dataset.x_attributes[0]].astype(np.float32)) 
        y_points = np.asarray(ys[m.dataset.y_attributes[0]]).reshape(-1,1)
        tensor_ys = torch.from_numpy(y_points.astype(np.float32))

        losses = []
        for i in range(1000):
            idx = np.random.randint(len(tensor_xs), size=batch)
            idx_tensor = torch.from_numpy(idx)
            sample_x = torch.index_select(tensor_xs,0,idx_tensor)
            sample_y = torch.index_select(tensor_ys,0,idx_tensor)
            sample_x = sample_x.to(device)
            sample_y = sample_y.to(device)
            pi, sigma, mu = m.model(sample_x)
            ll = LL(pi, sigma, mu, sample_y, device)
            losses.append(ll.item())

        if fig is not None:
            bins = int(len(losses)/10)
            if bins == 0:
                bins = 1
            fig.hist(losses, bins=bins, label=label)

        ind = []
        ood = []
        for loss in losses:
            stat = loss#np.mean(loss - mean)
            if stat < threshold:
                ood.append(loss)
            else:
                ind.append(loss)


        print("mean: {:0.4f}".format(mean), "threshold: {:0.4f}".format(threshold))
        print("number of all samples: {}".format(len(losses)))
        print("number of ood detected: {}".format(len(ood)))
        print("number of ind detected: {}".format(len(ind)))
        if label == "IND":
            fpr = len(ood)/len(losses)
            print ("FPR = {:0.4f}".format(fpr))
            return fpr, fig
        elif label == "OOD":
            fnr = len(ind)/len(losses)
            print ("FNR = {:0.4f}".format(fnr))
            return fnr, fig

    def online_phase_stream(m, xs, ys, mean, threshold, batch, label, fig):
        tensor_xs = torch.from_numpy(xs[m.dataset.x_attributes[0]].astype(np.float32)) 
        y_points = np.asarray(ys[m.dataset.y_attributes[0]]).reshape(-1,1)
        tensor_ys = torch.from_numpy(y_points.astype(np.float32))
        dataset = torch.utils.data.TensorDataset(tensor_xs, tensor_ys)
        loader = torch.utils.data.DataLoader(dataset,batch_size=batch, shuffle=False)
        losses = []
        m.model = m.model.to(device)
        for i, (x,y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            pi, sigma, mu = m.model(x)
            ll = LL(pi, sigma, mu, y, device)
            losses.append(ll.item())


        if fig is not None:
            bins = int(len(losses)/10)
            if bins == 0:
                bins = 1
            fig.hist(losses, bins=bins, label=label)

        ind = []
        ood = []
        for loss in losses:
            stat = loss#np.mean(loss - mean)
            if stat < threshold:
                ood.append(loss)
            else:
                ind.append(loss)


        print("mean: {:0.4f}".format(mean), "threshold: {:0.4f}".format(threshold))
        print("number of all samples: {}".format(len(losses)))
        print("number of ood detected: {}".format(len(ood)))
        print("number of ind detected: {}".format(len(ind)))
        if label == "IND":
            fpr = len(ood)/len(losses)
            print ("FPR = {:0.4f}".format(fpr))
            return fpr, fig
        elif label == "OOD":
            fnr = len(ind)/len(losses)
            print ("FNR = {:0.4f}".format(fnr))
            return fnr, fig

    m = None
    with open(model, 'rb') as d:
        m = dill.load(d)

    x_main, y_main = m.dataset.read_data(main_data, haveheader=True, to_dict=True)
    x_ind, y_ind = m.dataset.read_data(test_IND, haveheader=True, to_dict=True) 
    x_ood, y_ood = m.dataset.read_data(test_OOD, haveheader=True, to_dict=True)

    x_main_encoded = {}
    for key in x_main.keys():
        x_main_encoded[key] = m.dataset.encoders[key].transform(np.asarray(x_main[key]).reshape(-1,1)).toarray()
    y_main_normalized = m.dataset.normalize(y_main)

    x_ind_encoded = {}
    for key in x_ind.keys():
        x_ind_encoded[key] = m.dataset.encoders[key].transform(np.asarray(x_ind[key]).reshape(-1,1)).toarray()
    y_ind_normalized = m.dataset.normalize(y_ind)

    x_ood_encoded = {}
    for key in x_ood.keys():
        x_ood_encoded[key] = m.dataset.encoders[key].transform(np.asarray(x_ood[key]).reshape(-1,1)).toarray()
    y_ood_normalized = m.dataset.normalize(y_ood)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 16})

    fig, ax = plt.subplots()
    mean, threshold, offtime, ax = offline_phase(m, x_main_encoded, y_main_normalized, simulations, sample, ax)
    fprs = []
    fnrs = []
    num_tuples = [i for i in range(1, 2000, 100)]
    for n in num_tuples:
        rate, ax = online_phase_sample(m, x_ind_encoded, y_ind_normalized, mean, threshold, n, 'IND', ax)
        fprs.append(rate)
        rate, ax = online_phase_sample(m, x_ood_encoded, y_ood_normalized, mean, threshold, n, 'OOD', ax)
        fnrs.append(rate)

    ax.legend(loc="upper left", prop={'size': 14})
    fig.savefig('histograms.png')

    plt.clf()
    fig, ax = plt.subplots()
    xi = list(range(len(fprs)))
    ax.plot(num_tuples, fprs, marker='o', linestyle='-', label='false positive rate')
    ax.plot(num_tuples, fnrs, marker='*', linestyle='--', label='false negative rate')
    ax.legend(loc="upper right")
    ax.set_xlabel('batch size',fontsize=20, fontweight='bold', fontfamily='Times New Roman')
    ax.set_ylabel('FPR/FNR rate',fontsize=20, fontweight='bold', fontfamily='Times New Roman')
    plt.title('DBEst++ | forest',fontsize=22, fontweight='bold', fontfamily='Times New Roman')
    #ax.set_xticks(num_tuples)
    #ax.set_xticklabels(num_tuples, rotation=45)
    fig.savefig('errors.png')




if __name__=="__main__":
    
    test_for_drift(model='models/forest.dill', main_data='data/train_set.csv', 
                   test_IND='data/update_batches/update_permFalse01.csv', 
                   test_OOD='data/update_batches/update_permTrue01.csv', 
                   simulations=10000, sample=32)
