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
import shutil
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
dill._dill._reverse_typemap['ClassType'] = type
np.seterr(divide='ignore', invalid='ignore')


lock = Lock()


class Benchmark:
    def __init__(self, category_att, range_att, num_queries=500):
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

def LL(pi, sigma, mu, target, device):
    """Calculates the error, given the MoG parameters and the target
    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """
    softmax = nn.Softmax(dim=1)
    pi = softmax(pi)
    
    prob = pi * gaussian_probability(sigma, mu, target)

    ll = torch.log(torch.sum(prob, dim=1)).to(device)
    return torch.mean(ll)

def log_likelihood_test(basemodel, testset, updatemodels, update_testsets):
    models = [basemodel]
    
    for file in np.sort(os.listdir(updatemodels)):
        if file.startswith('update') and file.endswith('.dill'):
            models.append(os.path.join(updatemodels,file))
    
    testfiles = [testset]
    for file in np.sort(os.listdir(update_testsets)):
        if file.startswith('update') and file.endswith('_test.csv'):
            testfiles.append(os.path.join(update_testsets,file))

    if len(models) != len(testfiles):
        print ('number of models and test_sets does not match!')
        return
    else:
        print ("\n\nThe files and their corresponding testfile:\n")
        for i, itm in enumerate(models):
            print (itm,'\t',testfiles[i])


    lls_modelupdated = []
    lls_modelfixed = []
    all_testsets = pd.DataFrame()
    for i, name in enumerate(models):
        m_fix = None
        with open(models[0], 'rb') as d:
            m_fix = dill.load(d)

        
        m_update = None
        with open(name, 'rb') as d:
            m_update = dill.load(d)


        x_values, y_values = m_update.dataset.read_data(testfiles[i], haveheader=True) 
        df = pd.DataFrame.from_dict({**x_values, **y_values})
        all_testsets = pd.concat([all_testsets,df])
        print (all_testsets.shape)
        xs = {}
        ys = {}
        for key in x_values.keys():
            xs[key] = all_testsets[key].tolist()
        for key in y_values.keys():
            ys[key] = all_testsets[key].tolist()
        del x_values
        del y_values

        x_encoded = {}
        for key in xs.keys():
            x_encoded[key] = m_update.dataset.encoders[key].transform(np.asarray(xs[key]).reshape(-1,1)).toarray()
        y_normalized = m_update.dataset.normalize(ys)
        
        tensor_xs = torch.from_numpy(x_encoded[m_update.dataset.x_attributes[0]].astype(np.float32)) 
        y_points = np.asarray(y_normalized[m_update.dataset.y_attributes[0]]).reshape(-1,1)
        tensor_ys = torch.from_numpy(y_points.astype(np.float32))

        # move variables to cuda
        tensor_xs = tensor_xs.to('cpu')
        tensor_ys = tensor_ys.to('cpu')
        
        pi, sigma, mu = m_fix.model(tensor_xs)
        ll = LL(pi, sigma, mu, tensor_ys, 'cpu')
        lls_modelfixed.append(ll)

        pi, sigma, mu = m_update.model(tensor_xs)
        ll = LL(pi, sigma, mu, tensor_ys, 'cpu')
        lls_modelupdated.append(ll)
    

    
    lls_modelfixed = [t.item() for t in lls_modelfixed]
    lls_modelupdated = [t.item() for t in lls_modelupdated]

    print (lls_modelfixed)
    print (lls_modelupdated)

    x = list(range(len(lls_modelfixed)))
    fig, (ax1, ax2) = plt.subplots(2)

    ax1.plot(x, lls_modelupdated, marker='', color='blue', linewidth=4,label='updated')
    ax1.plot(x, lls_modelfixed, marker='', color='olive', linewidth=2,label='fixed')
    ax1.legend()
    ax1.set_xlabel("step")
    ax1.set_ylabel("log likelihood")
    #ax1.set_ylim([0,1])
    
    ax2.plot(x, lls_modelupdated, marker='o', markerfacecolor='blue', markersize=10, color='skyblue', linewidth=4,label='updated')
    ax2.plot(x, lls_modelfixed, marker='', color='olive', linewidth=2,label='fixed')
    ax2.legend()
    ax2.set_xlabel("step")
    ax2.set_ylabel("likelihood")
    #ax2.set_ylim([0,1])


    plt.savefig('ll.png')

def query_gt_setup(main_train_file, updates_dir, benchmark_dir, cat_att, range_att, sep, type, agg):
    datasets = []
    for file in np.sort(os.listdir(updates_dir)):
        if 'only' in file and not('test' in file):
            datasets.append(os.path.join(updates_dir,file)) 
    datasets = [main_train_file] + datasets
    datasets = ['../../job_data/title_only_part1.csv','../../job_data/title_only_part2.csv',
                '../../job_data/title_only_part3.csv','../../job_data/title_only_part4.csv',
                '../../job_data/title_only_part5.csv']
    
    if type=='olddata':
        df = pd.read_csv(datasets[0],sep=sep)
        _range = (df[range_att].min(), df[range_att].max())

        categories = list(set(df[cat_att].tolist()))
        b = Benchmark(cat_att, range_att)


        print ("Start generating random queries ...")
        b.generate_queries(range=_range,categories=categories,type=type, agg=agg)
        
        queries = os.path.join(benchmark_dir, agg + '00.sql')
        with open(queries, 'w') as f:
            for q in b.queries:
                f.write(q)
                f.write('\n')
        

        for i, dataset in enumerate(datasets):

            print ("Start generating ground-truth ...")
            df = pd.read_csv(dataset,sep=sep)
            results = b.calculate_groundtruth(df)


            if i>0:
                re = pd.read_csv(os.path.join(benchmark_dir, agg+str(i-1).zfill(2)+'.csv'), header=None)[0].to_dict()
                for key in results.keys():
                    results[key] = results[key] + re[key]
            
            ground_truth = os.path.join(benchmark_dir, agg+str(i).zfill(2)+'.csv')
            with open(ground_truth, 'w') as g:
                for key in results.keys():
                    g.write(str(results[key]))
                    g.write('\n')


        for i, dataset in enumerate(datasets):
            try:
                shutil.copyfile(queries, os.path.join(benchmark_dir, agg + str(i).zfill(2)+'.sql'))
            except:
                pass


    elif type=='newdata':

        b = None
        for i, dataset in enumerate(datasets):

            df = pd.read_csv(dataset,sep=sep)
          
            _range = (df[range_att].min(), df[range_att].max())

            categories = list(set(df[cat_att].tolist()))


            print ("Start generating random queries ...")

            b = Benchmark(cat_att, range_att)
            b.generate_queries(range=_range,categories=categories,type=type, agg=agg)


            print ("Start generating ground-truth ...")
            results = b.calculate_groundtruth(df)

            del df
            queries = os.path.join(benchmark_dir, agg + str(i).zfill(2)+'.sql')

            with open(queries, 'w') as f:
                for q in b.queries:
                    f.write(q)
                    f.write('\n')
                    
            ground_truth = os.path.join(benchmark_dir, agg + str(i).zfill(2)+'.csv')
            keys = np.sort(list(results.keys()))
            with open(ground_truth, 'w') as g:
                for key in keys:
                    g.write(str(results[key]))
                    g.write('\n')


    elif type=='alldata':
        b = None
        print (datasets)
        for i, dataset in enumerate(datasets):
            range1 = np.inf
            df = pd.DataFrame()
            for j in range(i+1):
                df1 = pd.read_csv(datasets[j],sep=sep)
                df = pd.concat([df,df1])
                if j==0:
                    range1 = df[range_att].max()

            _range = (df[range_att].min(), df[range_att].max())

            categories = list(set(df[cat_att].tolist()))


            print ("Start generating random queries ...")
            
            if True:#i==0:
                b = Benchmark(cat_att, range_att)
                b.generate_queries(range=_range, range1=range1,categories=categories,type=type, agg=agg)


            print ("Start generating ground-truth ...")
            results = b.calculate_groundtruth(df)

            del df
            queries = os.path.join(benchmark_dir, agg + str(i).zfill(2)+'.sql')
            with open(queries, 'w') as f:
                for q in b.queries:
                    f.write(q)
                    f.write('\n')
                    
            ground_truth = os.path.join(benchmark_dir, agg + str(i).zfill(2)+'.csv')
            keys = np.sort(list(results.keys()))
            with open(ground_truth, 'w') as g:
                for key in keys:
                    g.write(str(results[key]))
                    g.write('\n')



    else:
        print ("Not implemented!")
        return

def run_queries(basemodel, modelsdir, benchmarkdir, _type, integral_points=100):
    models = []
    for file in np.sort(os.listdir(modelsdir)):
        if file.endswith('.dill'):
            models.append(os.path.join(modelsdir,file)) 

    if not os.path.exists(basemodel):
        print ("Model not found!")
        return


    relative_errors = []
    selected_queries = []
    for model_num, model in enumerate(models):
        ground_truth = os.path.join(benchmarkdir, str(model_num).zfill(2)+'.csv')
        query_file = os.path.join(benchmarkdir, str(model_num).zfill(2)+'.sql')

        M = None
        with open(model, 'rb') as d:
            M = dill.load(d)

        queries = []
        with open(query_file,'r') as file:
            for line in file:
                qs = line.split(';')
                for itm in qs:
                    if itm.lower().startswith('select'):
                        queries.append(itm)
                    elif len(itm)>1:
                        print ('query not recognized \n {}'.format(itm))

        reals = {}
        with open(ground_truth,'r') as file:
            for i, line in enumerate(file):
                try:
                    reals[i] = float(line.split(',')[0])
                except:
                    reals[i] = 0

        predictions = {}
        freqs = {}
        equalities = {}
        agg = ""
        for i, query in enumerate(queries):
            parser = Parser()
            succ, conditions, agg = parser.parse(query, M.dataset.x_attributes + M.dataset.y_attributes)
            if not succ:
                print ("error in {}".format(query))
            

            x_values = {}
            y_values = {}
            lb = ub = step = 0
            for key in conditions.keys():
                if conditions[key].equalities:
                    x_values[key] = conditions[key].equalities
                else:
                    lb = M.dataset.normalizing_stats[key].min
                    ub = M.dataset.normalizing_stats[key].max
                    if conditions[key].lb is not None:
                        lb = conditions[key].lb
                    if conditions[key].ub is not None:
                        ub = conditions[key].ub
                    y_values[key], step = list(np.linspace(lb,ub,integral_points, retstep=True))


            frequencies = {}        
            x_encoded = {}
            for key in x_values.keys():
                x_encoded[key] = M.dataset.encoders[key].transform(np.asarray(x_values[key]).reshape(-1,1)).toarray()
                frequencies[key] = [M.dataset.FTs[key][cat] for cat in x_values[key]]                
            y_normalized = M.dataset.normalize(y_values)

            if agg == "count":
                probs = M.predict(x_points = x_encoded[list(x_values.keys())[0]] ,y_points=y_normalized[list(y_values.keys())[0]])
                probs = probs / M.dataset.normalizing_stats[list(y_values.keys())[0]].width * 2
                answer = cal_count(probs, step, frequencies)

            elif agg == "sum":
                probs = M.predict(x_points = x_encoded[list(x_values.keys())[0]] ,y_points=y_normalized[list(y_values.keys())[0]])
                probs = probs / M.dataset.normalizing_stats[list(y_values.keys())[0]].width * 2
                regs = y_values[regs]
                answer = cal_sum(probs, regs, step, frequencies)

            elif agg == "avg":
                probs = M.predict(x_points = x_encoded[list(x_values.keys())[0]] ,y_points=y_normalized[list(y_values.keys())[0]])
                probs = probs / M.dataset.normalizing_stats[list(y_values.keys())[0]].width * 2
                regs = y_values[regs]
                answer = cal_avg(probs, regs, step, frequencies)

 

            freqs[i] =  np.asarray(frequencies[list(frequencies.keys())[0]]).reshape(-1,1)
            predictions[i] = np.floor(answer[0][0]) if np.floor(answer[0][0])>0 else 1
            equalities[i] = list(x_values.values())[0]



        cal_err(reals, predictions, _type, agg, model_num)

def run_single_file(model, query_file, ground_truth, _type, integral_points=100):

    M = None
    with open(model, 'rb') as d:
        M = dill.load(d)


    queries = []
    with open(query_file,'r') as file:
        for line in file:
            qs = line.split(';')
            for itm in qs:
                if itm.lower().startswith('select'):
                    queries.append(itm)
                elif len(itm)>1:
                    print ('query not recognized \n {}'.format(itm))

    reals = {}
    with open(ground_truth,'r') as file:
        for i, line in enumerate(file):
            try:
                reals[i] = float(line.split(',')[0])
            except:
                reals[i] = 0
            


    predictions = {}
    freqs = {}
    equalities = {}
    agg = ""
    for i, query in enumerate(queries):
        parser = Parser()
        succ, conditions, agg = parser.parse(query, M.dataset.x_attributes + M.dataset.y_attributes)
        if not succ:
            print ("error in {}".format(query))
        

        x_values = {}
        y_values = {}
        lb = ub = step = 0
        for key in conditions.keys():
            if conditions[key].equalities:
                x_values[key] = conditions[key].equalities
            else:
                lb = M.dataset.normalizing_stats[key].min
                ub = M.dataset.normalizing_stats[key].max
                if conditions[key].lb is not None:
                    lb = conditions[key].lb
                if conditions[key].ub is not None:
                    ub = conditions[key].ub
                y_values[key], step = list(np.linspace(lb,ub,integral_points, retstep=True))


        frequencies = {}        
        x_encoded = {}
        for key in x_values.keys(): 
            x_encoded[key] = M.dataset.encoders[key].transform(np.asarray(x_values[key]).reshape(-1,1)).toarray()
            frequencies[key] = [M.dataset.FTs[key][cat] for cat in x_values[key]]
            
        y_normalized = M.dataset.normalize(y_values)

        if agg == "count":
            probs= M.predict(x_points = x_encoded[list(x_values.keys())[0]] ,y_points=y_normalized[list(y_values.keys())[0]])
            probs = probs / M.dataset.normalizing_stats[list(y_values.keys())[0]].width * 2
            answer = cal_count(probs, step, frequencies)

        elif agg == "sum":
            probs = M.predict(x_points = x_encoded[list(x_values.keys())[0]] ,y_points=y_normalized[list(y_values.keys())[0]])
            probs = probs / M.dataset.normalizing_stats[list(y_values.keys())[0]].width * 2
            regs = y_values[list(y_values.keys())[0]]
            answer = cal_sum(probs, regs, step, frequencies)

        elif agg == "avg":
            probs = M.predict(x_points = x_encoded[list(x_values.keys())[0]] ,y_points=y_normalized[list(y_values.keys())[0]])
            probs = probs / M.dataset.normalizing_stats[list(y_values.keys())[0]].width * 2
            regs = y_values[list(y_values.keys())[0]]
            answer = cal_avg(probs, regs, step, frequencies)



        freqs[i] =  np.asarray(frequencies[list(frequencies.keys())[0]]).reshape(-1,1)
        predictions[i] = np.floor(answer[0][0]) if np.floor(answer[0][0])>0 else 1
        equalities[i] = list(x_values.values())[0]

        #print (reals[i], predictions[i], freqs[i])


    cal_err(reals, predictions, _type, agg, 0)

def plot(model, input):
    M = None
    with open(model, 'rb') as d:
        M = dill.load(d)

    x_encoded = {}

    x_encoded['country'] = M.dataset.encoders['country'].transform(np.asarray([input.lower()]).reshape(-1,1)).toarray()
    
    tensor_xs = torch.from_numpy(x_encoded['country'].astype(np.float32))    
    tensor_xs = tensor_xs.to('cpu')
    pis, sigmas, mus = M.model(tensor_xs)

    softmax = nn.Softmax(dim=1)

    pis = softmax(pis)
    pis = pis.cpu()
    sigmas = sigmas.cpu()
    mus = mus.cpu()

    samples = MoGsampling(pis, sigmas,mus,10000,'cpu')

    results = [denormalize(i.item(), M.dataset.normalizing_stats['invoicedate'].mean, M.dataset.normalizing_stats['invoicedate'].width) for i in samples[0]]

    q = pd.DataFrame(data={'y':results})
    ax = q.y.hist(bins=q.y.nunique(),color='blue')
    fig = ax.get_figure()
    fig.savefig('samples.png')



if __name__=="__main__":
    
    #log_likelihood_test(basemodel='DenMDN_census.dill', testset='data/test_set.csv', updatemodels='.', update_testsets='data/update_batches')
    #for agg in ['count', 'sum', 'avg']:
    #    query_gt_setup(main_train_file="../../job_data/title_until_part1.csv", updates_dir='../../job_data/',
    #                   benchmark_dir='benchmark/alldata', cat_att="info_type_id", range_att="production_year", 
    #                   sep=',', type='alldata', agg=agg)
    models = ['distill01.dill','distill02.dill','distill03.dill','distill04.dill']
    #models = ['stale01.dill', 'stale02.dill', 'stale03.dill', 'stale04.dill']
    #models = ['finetune01.dill', 'finetune02.dill', 'finetune03.dill', 'finetune04.dill']
    #models = ['retrain01.dill', 'retrain02.dill', 'retrain03.dill', 'retrain04.dill']
    for i,model in enumerate(models):
        j = i+1
        run_single_file(model=model,query_file='benchmark/alldata/count0{}.sql'.format(j), ground_truth='benchmark/alldata/count0{}.csv'.format(j), _type='q-error')
        run_single_file(model=model,query_file='benchmark/alldata/sum0{}.sql'.format(j), ground_truth='benchmark/alldata/sum0{}.csv'.format(j), _type='relative-error')
        run_single_file(model=model,query_file='benchmark/alldata/avg0{}.sql'.format(j), ground_truth='benchmark/alldata/avg0{}.csv'.format(j), _type='relative-error')
    #run_single_file(model='retrain02.dill',query_file='benchmark/alldata/count02.sql', ground_truth='benchmark/alldata/count02.csv',_type='q-error')
    #plot('DenMDN_flight.dill', 'united kingdom')


