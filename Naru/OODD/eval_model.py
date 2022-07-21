"""Evaluate estimators (Naru or others) on queries."""
import argparse
import collections
import glob
import os
import pickle
import re
import time

import numpy as np
import pandas as pd
import random
import torch
import pickle
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.manifold import TSNE

import common
import datasets
import estimators as estimators_lib
import made
import transformer

# For inference speed.
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASETS = ['imdb', 'imdb_updated']
print('Device', DEVICE)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='cardinality', help='cardinality or likelihood.')
parser.add_argument('--rlth', type=float, default=-1, help='threshold to hit during re-learn')
parser.add_argument('--inference-opts',
                    action='store_true',
                    help='Tracing optimization for better latency.')

parser.add_argument('--num-queries', type=int, default=20, help='# queries.')
parser.add_argument('--dataset', type=str, default='dmv-tiny', help='Dataset.')
parser.add_argument('--err-csv',
                    type=str,
                    default='results.csv',
                    help='Save result csv to what path?')
parser.add_argument('--glob',
                    type=str,
                    help='Checkpoints to glob under models/.')
parser.add_argument('--blacklist',
                    type=str,
                    help='Remove some globbed checkpoint files.')
parser.add_argument('--psample',
                    type=int,
                    default=2000,
                    help='# of progressive samples to use per query.')
parser.add_argument(
    '--column-masking',
    action='store_true',
    help='Turn on wildcard skipping.  Requires checkpoints be trained with '\
    'column masking.')
parser.add_argument('--order',
                    nargs='+',
                    type=int,
                    help='Use a specific order?')

# MADE.
parser.add_argument('--fc-hiddens',
                    type=int,
                    default=128,
                    help='Hidden units in FC.')
parser.add_argument('--layers', type=int, default=4, help='# layers in FC.')
parser.add_argument('--residual', action='store_true', help='ResMade?')
parser.add_argument('--direct-io', action='store_true', help='Do direct IO?')
parser.add_argument(
    '--inv-order',
    action='store_true',
    help='Set this flag iff using MADE and specifying --order. Flag --order'\
    'lists natural indices, e.g., [0 2 1] means variable 2 appears second.'\
    'MADE, however, is implemented to take in an argument the inverse '\
    'semantics (element i indicates the position of variable i).  Transformer'\
    ' does not have this issue and thus should not have this flag on.')
parser.add_argument(
    '--input-encoding',
    type=str,
    default='binary',
    help='Input encoding for MADE/ResMADE, {binary, one_hot, embed}.')
parser.add_argument(
    '--output-encoding',
    type=str,
    default='one_hot',
    help='Iutput encoding for MADE/ResMADE, {one_hot, embed}.  If embed, '
    'then input encoding should be set to embed as well.')

# Transformer.
parser.add_argument(
    '--heads',
    type=int,
    default=0,
    help='Transformer: num heads.  A non-zero value turns on Transformer'\
    ' (otherwise MADE/ResMADE).'
)
parser.add_argument('--blocks',
                    type=int,
                    default=2,
                    help='Transformer: num blocks.')
parser.add_argument('--dmodel',
                    type=int,
                    default=32,
                    help='Transformer: d_model.')
parser.add_argument('--dff', type=int, default=128, help='Transformer: d_ff.')
parser.add_argument('--transformer-act',
                    type=str,
                    default='gelu',
                    help='Transformer activation.')

# Estimators to enable.
parser.add_argument('--run-sampling',
                    action='store_true',
                    help='Run a materialized sampler?')
parser.add_argument('--run-maxdiff',
                    action='store_true',
                    help='Run the MaxDiff histogram?')
parser.add_argument('--run-bn',
                    action='store_true',
                    help='Run Bayes nets? If enabled, run BN only.')

# Bayes nets.
parser.add_argument('--bn-samples',
                    type=int,
                    default=200,
                    help='# samples for each BN inference.')
parser.add_argument('--bn-root',
                    type=int,
                    default=0,
                    help='Root variable index for chow liu tree.')
# Maxdiff
parser.add_argument(
    '--maxdiff-limit',
    type=int,
    default=30000,
    help='Maximum number of partitions of the Maxdiff histogram.')

args = parser.parse_args()


def InvertOrder(order):
    if order is None:
        return None
    # 'order'[i] maps nat_i -> position of nat_i
    # Inverse: position -> natural idx.  This it the "true" ordering -- it's how
    # heuristic orders are generated + (less crucially) how Transformer works.
    nin = len(order)
    inv_ordering = [None] * nin
    for natural_idx in range(nin):
        inv_ordering[order[natural_idx]] = natural_idx
    return inv_ordering


def Entropy(name, data, bases=None):
    import scipy.stats
    s = 'Entropy of {}:'.format(name)
    ret = []
    for base in bases:
        assert base == 2 or base == 'e' or base is None
        e = scipy.stats.entropy(data, base=base if base != 'e' else None)
        ret.append(e)
        unit = 'nats' if (base == 'e' or base is None) else 'bits'
        s += ' {:.4f} {}'.format(e, unit)
    print(s)
    return ret

def RunEpoch(split,
             model,
             opt,
             train_data,
             lr_scheduler=None,
             val_data=None,
             batch_size=100,
             upto=None,
             epoch_num=None,
             verbose=False,
             log_every=10,
             return_losses=False,
             return_embed=False,
             table_bits=None):
    torch.set_grad_enabled(split == 'train')
    model.train() if split == 'train' else model.eval()
    dataset = train_data if split == 'train' else val_data
    losses = []


    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=(split == 'train'))

    # How many orderings to run for the same batch?
    nsamples = 1
    if hasattr(model, 'orderings'):
        nsamples = len(model.orderings)


    for step, xb in enumerate(loader):

        xb = xb.to(DEVICE).to(torch.float32)

        # Forward pass, potentially through several orderings.
        xbhat = None
        model_logits = []
        num_orders_to_forward = 1
        if split == 'test' and nsamples > 1:
            # At test, we want to test the 'true' nll under all orderings.
            num_orders_to_forward = nsamples

        for i in range(num_orders_to_forward):
            if hasattr(model, 'update_masks'):
                # We want to update_masks even for first ever batch.
                model.update_masks()

            model_out = model(xb)
            model_logits.append(model_out)

            if xbhat is None:
                xbhat = torch.zeros_like(model_out)
            xbhat += model_out

        if xbhat.shape == xb.shape:
            if mean:
                xb = (xb * std) + mean
            loss = F.binary_cross_entropy_with_logits(
                xbhat, xb, size_average=False) / xbhat.size()[0]
        else:
            if model.input_bins is None:
                # NOTE: we have to view() it in this order due to the mask
                # construction within MADE.  The masks there on the output unit
                # determine which unit sees what input vars.
                xbhat = xbhat.view(-1, model.nout // model.nin, model.nin)
                # Equivalent to:
                loss = F.cross_entropy(xbhat, xb.long(), reduction='none') \
                        .sum(-1).mean()
            else:
                if num_orders_to_forward == 1:
                    loss = model.nll(xbhat, xb).mean()
                else:
                    # Average across orderings & then across minibatch.
                    #
                    #   p(x) = 1/N sum_i p_i(x)
                    #   log(p(x)) = log(1/N) + log(sum_i p_i(x))
                    #             = log(1/N) + logsumexp ( log p_i(x) )
                    #             = log(1/N) + logsumexp ( - nll_i (x) )
                    #
                    # Used only at test time.
                    logps = []  # [batch size, num orders]
                    assert len(model_logits) == num_orders_to_forward, len(
                        model_logits)
                    for logits in model_logits:
                        # Note the minus.
                        logps.append(-model.nll(logits, xb))
                    logps = torch.stack(logps, dim=1)
                    logps = logps.logsumexp(dim=1) + torch.log(
                        torch.tensor(1.0 / nsamples, device=logps.device))
                    loss = (-logps).mean()

        losses.append(loss.item())


        if split == 'train':
            opt.zero_grad()
            loss.backward()
            opt.step()

        if verbose:
            print('%s epoch average loss: %f' % (split, np.mean(losses)))

    if lr_scheduler:
        lr_scheduler.step()    

    if return_losses:
        return losses
    if return_embed:
        return embeddings
    return np.mean(losses)


def MakeTable(dataset=args.dataset):
    assert dataset in DATASETS
    if dataset == 'imdb':
        table = datasets.LoadIMDB(filename='title_until_part5.csv')
        reference = datasets.LoadIMDB(filename='title_until_part5.csv')
    if dataset == 'imdb_full':
        table = datasets.LoadIMDB(filename='title_until_part5.csv')
        reference = datasets.LoadIMDB(filename='title_until_part5.csv')

    oracle_est = estimators_lib.Oracle(table)
    if args.run_bn:
        return table, common.TableDataset(table), oracle_est
    return table, reference, oracle_est


def ErrorMetric(est_card, card):
    if card == 0 and est_card != 0:
        return est_card
    if card != 0 and est_card == 0:
        return card
    if card == 0 and est_card == 0:
        return 1.0
    return max(est_card / card, card / est_card)


def SampleTupleThenRandom(all_cols,
                          num_filters,
                          rng,
                          table,
                          return_col_idx=False):
    s = table.data.iloc[rng.randint(0, table.cardinality)]
    vals = s.values

    if args.dataset in ['dmv', 'dmv-tiny']:
        # Giant hack for DMV.
        vals[6] = vals[6].to_datetime64()

    idxs = rng.choice(len(all_cols), replace=False, size=num_filters)
    cols = np.take(all_cols, idxs)

    # If dom size >= 10, okay to place a range filter.
    # Otherwise, low domain size columns should be queried with equality.
    ops = rng.choice(['<=', '>=', '='], size=num_filters)
    ops_all_eqs = ['='] * num_filters
    sensible_to_do_range = [c.DistributionSize() >= 10 for c in cols]
    ops = np.where(sensible_to_do_range, ops, ops_all_eqs)

    if num_filters == len(all_cols):
        if return_col_idx:
            return np.arange(len(all_cols)), ops, vals
        return all_cols, ops, vals

    vals = vals[idxs]
    if return_col_idx:
        return idxs, ops, vals

    return cols, ops, vals


def GenerateQuery(all_cols, rng, table, return_col_idx=False):
    """Generate a random query."""
    num_filters = rng.randint(2, 5)
    cols, ops, vals = SampleTupleThenRandom(all_cols,
                                            num_filters,
                                            rng,
                                            table,
                                            return_col_idx=return_col_idx)
    return cols, ops, vals


def Query(estimators,
          do_print=True,
          oracle_card=None,
          query=None,
          table=None,
          ref_table=None,
          ref_cols=None,
          oracle_est=None):
    assert query is not None
    cols, ops, vals = query
    selected_ref_cols = []
    for cl in cols:
        for clm in ref_cols:
            if clm.Name() == cl.Name():
                selected_ref_cols.append(clm)

    selected_ref_cols = np.array(selected_ref_cols)
    ### Actually estimate the query.

    def pprint(*args, **kwargs):
        if do_print:
            print(*args, **kwargs)

    # Actual.
    card = oracle_est.Query(cols, ops,
                            vals) if oracle_card is None else oracle_card
    if card == 0:
        return

    pprint('Q(', end='')
    for c, o, v in zip(cols, ops, vals):
        pprint('{} {} {}, '.format(c.name, o, str(v)), end='')
    pprint('): ', end='')

    pprint('\n  actual {} ({:.3f}%) '.format(card,
                                             card / table.cardinality * 100),
           end='')

    for est in estimators:
        est_card = est.Query(cols, ops, vals, ref_table, selected_ref_cols)
        err = ErrorMetric(est_card, card)
        est.AddError(err, est_card, card)
        pprint('{} {} (err={:.3f}) '.format(str(est), est_card, err), end='')
    pprint()


def ReportEsts(estimators):
    v = -1
    for est in estimators:
        print(est.name, 'max', np.max(est.errs), '99th',
              np.quantile(est.errs, 0.99), '95th', np.quantile(est.errs, 0.95),
              'median', np.quantile(est.errs, 0.5))
        v = max(v, np.max(est.errs))
    return v


def RunN(table,
         ref_table,
         cols,
         ref_cols,
         estimators,
         rng=None,
         num=20,
         log_every=50,
         num_filters=11,
         oracle_cards=None,
         oracle_est=None):
    if rng is None:
        rng = np.random.RandomState(1234)



    pre_query_path = 'previous_queries.pkl'
    queries = []
    if os.path.exists(pre_query_path):
        with open(pre_query_path, 'rb') as d:
            queries = pickle.load(d)

    last_time = None

    if len(queries) == 0:
        for i in range(num):
            do_print = False
            if i % log_every == 0:
                if last_time is not None:
                    print('{:.1f} queries/sec'.format(log_every /
                                                      (time.time() - last_time)))
                do_print = True
                print('Query {}:'.format(i), end=' ')
                last_time = time.time()


            query = GenerateQuery(cols, rng, table)
            queries.append(query)
            Query(estimators,
                  do_print,
                  oracle_card=oracle_cards[i]
                  if oracle_cards is not None and i < len(oracle_cards) else None,
                  query=query,
                  table=table,
                  ref_table=ref_table,
                  ref_cols=ref_cols,
                  oracle_est=oracle_est)

            max_err = ReportEsts(estimators)

        with open(pre_query_path, 'wb') as f:
            pickle.dump(queries, f, protocol=pickle.HIGHEST_PROTOCOL)
        return False
    else:
        for i, query in enumerate(queries):
            do_print = False
            if i % log_every == 0:
                if last_time is not None:
                    print('{:.1f} queries/sec'.format(log_every /
                                                      (time.time() - last_time)))
                do_print = True
                print('Query {}:'.format(i), end=' ')
                last_time = time.time()


            Query(estimators,
                  do_print,
                  oracle_card=oracle_cards[i]
                  if oracle_cards is not None and i < len(oracle_cards) else None,
                  query=query,
                  table=table,
                  ref_table=ref_table,
                  ref_cols=ref_cols,
                  oracle_est=oracle_est)

            max_err = ReportEsts(estimators)

        return False


def RunNParallel(estimator_factory,
                 parallelism=2,
                 rng=None,
                 num=20,
                 num_filters=11,
                 oracle_cards=None):
    """RunN in parallel with Ray.  Useful for slow estimators e.g., BN."""
    import ray
    ray.init(redis_password='xxx')

    @ray.remote
    class Worker(object):

        def __init__(self, i):
            self.estimators, self.table, self.oracle_est = estimator_factory()
            self.columns = np.asarray(self.table.columns)
            self.i = i

        def run_query(self, query, j):
            col_idxs, ops, vals = pickle.loads(query)
            Query(self.estimators,
                  do_print=True,
                  oracle_card=oracle_cards[j]
                  if oracle_cards is not None else None,
                  query=(self.columns[col_idxs], ops, vals),
                  table=self.table,
                  oracle_est=self.oracle_est)

            print('=== Worker {}, Query {} ==='.format(self.i, j))
            for est in self.estimators:
                est.report()

        def get_stats(self):
            return [e.get_stats() for e in self.estimators]

    print('Building estimators on {} workers'.format(parallelism))
    workers = []
    for i in range(parallelism):
        workers.append(Worker.remote(i))

    print('Building estimators on driver')
    estimators, table, _ = estimator_factory()
    cols = table.columns

    if rng is None:
        rng = np.random.RandomState(1234)
    queries = []
    for i in range(num):
        col_idxs, ops, vals = GenerateQuery(cols,
                                            rng,
                                            table=table,
                                            return_col_idx=True)
        queries.append((col_idxs, ops, vals))

    cnts = 0
    for i in range(num):
        query = queries[i]
        print('Queueing execution of query', i)
        workers[i % parallelism].run_query.remote(pickle.dumps(query), i)

    print('Waiting for queries to finish')
    stats = ray.get([w.get_stats.remote() for w in workers])

    print('Merging and printing final results')
    for stat_set in stats:
        for e, s in zip(estimators, stat_set):
            e.merge_stats(s)
    time.sleep(1)

    print('=== Merged stats ===')
    for est in estimators:
        est.report()
    return estimators


def MakeBnEstimators():
    table, train_data, oracle_est = MakeTable()
    estimators = [
        estimators_lib.BayesianNetwork(train_data,
                                       args.bn_samples,
                                       'chow-liu',
                                       topological_sampling_order=True,
                                       root=args.bn_root,
                                       max_parents=2,
                                       use_pgm=False,
                                       discretize=100,
                                       discretize_method='equal_freq')
    ]

    for est in estimators:
        est.name = str(est)
    return estimators, table, oracle_est


def MakeMade(scale, cols_to_train, seed, fixed_ordering=None):
    if args.inv_order:
        print('Inverting order!')
        fixed_ordering = InvertOrder(fixed_ordering)

    model = made.MADE(
        nin=len(cols_to_train),
        hidden_sizes=[scale] *
        args.layers if args.layers > 0 else [512, 256, 512, 128, 1024],
        nout=sum([c.DistributionSize() for c in cols_to_train]),
        input_bins=[c.DistributionSize() for c in cols_to_train],
        input_encoding=args.input_encoding,
        output_encoding=args.output_encoding,
        embed_size=32,
        seed=seed,
        do_direct_io_connections=args.direct_io,
        natural_ordering=False if seed is not None and seed != 0 else True,
        residual_connections=args.residual,
        fixed_ordering=fixed_ordering,
        column_masking=args.column_masking,
    ).to(DEVICE)

    return model


def MakeTransformer(cols_to_train, fixed_ordering, seed=None):
    return transformer.Transformer(
        num_blocks=args.blocks,
        d_model=args.dmodel,
        d_ff=args.dff,
        num_heads=args.heads,
        nin=len(cols_to_train),
        input_bins=[c.DistributionSize() for c in cols_to_train],
        use_positional_embs=True,
        activation=args.transformer_act,
        fixed_ordering=fixed_ordering,
        column_masking=args.column_masking,
        seed=seed,
    ).to(DEVICE)


def ReportModel(model, blacklist=None):
    ps = []
    for name, p in model.named_parameters():
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    print('Number of model parameters: {} (~= {:.1f}MB)'.format(num_params, mb))
    print(model)
    return mb


def SaveEstimators(path, estimators, return_df=False):
    # name, query_dur_ms, errs, est_cards, true_cards
    results = pd.DataFrame()
    print (estimators)
    for est in estimators:
        data = {
            'est': [est.name] * len(est.errs),
            'err': est.errs,
            'est_card': est.est_cards,
            'true_card': est.true_cards,
            'query_dur_ms': est.query_dur_ms,
        }
        results = results.append(pd.DataFrame(data))

    results.to_csv(path, index=False)

    def percentile(n):
        def percentile_(x):
            return np.percentile(x, n)
        percentile_.__name__ = 'percentile_%s' % n
        return percentile_

    grouped = results.groupby('est')
    column = grouped['err']
    stats = column.agg([np.mean, np.median,
        percentile(90), percentile(95), percentile(99)],
        as_index=False).reset_index()
    
    stats.to_csv('summary_'+path,index=False)
    return stats


def LoadOracleCardinalities():
    ORACLE_CARD_FILES = {
        'dmv': 'datasets/dmv-2000queries-oracle-cards-seed1234.csv'
    }
    path = ORACLE_CARD_FILES.get(args.dataset, None)
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        assert len(df) == 2000, len(df)
        return df.values.reshape(-1)
    return None


def Main():
    all_ckpts = glob.glob('./{}'.format(args.glob))
    if args.blacklist:
        all_ckpts = [ckpt for ckpt in all_ckpts if args.blacklist not in ckpt]

    selected_ckpts = all_ckpts
    oracle_cards = None#LoadOracleCardinalities()
    print('ckpts', selected_ckpts)

    if not args.run_bn:
        # OK to load tables now
        table, reference_table, oracle_est = MakeTable()
        cols_to_train = table.columns

    Ckpt = collections.namedtuple(
        'Ckpt', 'epoch model_bits bits_gap path loaded_model seed')
    parsed_ckpts = []

    for s in selected_ckpts:
        if args.order is None:
            z = re.match('.+model([\d\.]+)-data([\d\.]+).+seed([\d\.]+).*.pt',
                         s)
        else:
            z = re.match(
                '.+model([\d\.]+)-data([\d\.]+).+seed([\d\.]+)-order.*.pt', s)
        assert z
        model_bits = float(z.group(1))
        data_bits = float(z.group(2))
        seed = int(z.group(3))
        bits_gap = model_bits - data_bits

        order = None
        if args.order is not None:
            order = list(args.order)

        if args.heads > 0:
            model = MakeTransformer(cols_to_train=reference_table.columns,
                                    fixed_ordering=order,
                                    seed=seed)
        else:
            if args.dataset in DATASETS:
                model = MakeMade(
                    scale=args.fc_hiddens,
                    cols_to_train=reference_table.columns,
                    seed=seed,
                    fixed_ordering=order,
                )
            else:
                assert False, args.dataset

        assert order is None or len(order) == model.nin, order
        ReportModel(model)
        print('Loading ckpt:', s)
        model.load_state_dict(torch.load(s))
        model.eval()

        print(s, bits_gap, seed)

        parsed_ckpts.append(
            Ckpt(path=s,
                 epoch=None,
                 model_bits=model_bits,
                 bits_gap=bits_gap,
                 loaded_model=model,
                 seed=seed))

    # Estimators to run.
    if args.run_bn:
        estimators = RunNParallel(estimator_factory=MakeBnEstimators,
                                  parallelism=50,
                                  rng=np.random.RandomState(1234),
                                  num=args.num_queries,
                                  num_filters=None,
                                  oracle_cards=oracle_cards)
    else:
        estimators = [
            estimators_lib.ProgressiveSampling(c.loaded_model,
                                               table,
                                               args.psample,
                                               device=DEVICE,
                                               shortcircuit=args.column_masking)
            for c in parsed_ckpts
        ]
        for est, ckpt in zip(estimators, parsed_ckpts):
            est.name = str(est) + '_{}_{:.3f}'.format(ckpt.seed, ckpt.bits_gap)

        if args.inference_opts:
            print('Tracing forward_with_encoded_input()...')
            for est in estimators:
                encoded_input = est.model.EncodeInput(
                    torch.zeros(args.psample, est.model.nin, device=DEVICE))

                # NOTE: this line works with torch 1.0.1.post2 (but not 1.2).
                # The 1.2 version changes the API to
                # torch.jit.script(est.model) and requires an annotation --
                # which was found to be slower.
                est.traced_fwd = torch.jit.trace(
                    est.model.forward_with_encoded_input, encoded_input)

        if args.run_sampling:
            SAMPLE_RATIO = {'census': [0.01]}  # ~1.3MB.
            for p in SAMPLE_RATIO.get(args.dataset, [0.01]):
                estimators.append(estimators_lib.Sampling(table, p=p))

        if args.run_maxdiff:
            estimators.append(
                estimators_lib.MaxDiffHistogram(table, args.maxdiff_limit))

        # Other estimators can be appended as well.

        if len(estimators):
            RunN(table,
                 reference_table,
                 cols_to_train,
                 reference_table.columns,
                 estimators,
                 rng=np.random.RandomState(1234),
                 num=args.num_queries,
                 log_every=1,
                 num_filters=None,
                 oracle_cards=oracle_cards,
                 oracle_est=oracle_est)

    results = SaveEstimators(args.err_csv, estimators)
    print('...Done, result:', args.err_csv)


def ConceptDriftTest(seed=0):
    torch.manual_seed(0)
    np.random.seed(0)

    results = {}
    for i in range(0,10):
        assert args.dataset in ['dmv','tpcds','census','forest','imdb']
        if args.dataset == 'dmv':
            table = datasets.LoadPartlyPermutedDmv(num_of_sorted_cols=i)
        elif args.dataset == 'tpcds':
            table = datasets.LoadSingleFile()
        elif args.dataset == 'census':
            table = datasets.LoadPartlyPermutedCensus(num_of_sorted_cols=i)
        elif args.dataset == 'forest':
            table = datasets.LoadPartlyPermutedForest(num_of_sorted_cols=i)

        table_bits = Entropy(
            table,
            table.data.fillna(value=0).groupby([c.name for c in table.columns
                                               ]).size(), [2])

        fixed_ordering = None

        if args.order is not None:
            print('Using passed-in order:', args.order)
            fixed_ordering = args.order

        print(table.data.info())


        table_main = table

        
        if args.dataset in ['dmv-tiny', 'dmv', 'tpcds','census','forest','imdb']:
            model = MakeMade(
                scale=args.fc_hiddens,
                cols_to_train=table.columns,
                seed=seed,
                fixed_ordering=fixed_ordering,
            )

            model.load_state_dict(torch.load(args.glob))
            model.eval()
        else:
            assert False, args.dataset

        mb = ReportModel(model)


        bs = 2048
        log_every = 200



        train_data = common.TableDataset(table_main)
        rndindices1 = torch.randperm(len(train_data.tuples))[:5000]
        train_data.tuples = train_data.tuples[rndindices1]


        print (train_data.tuples.shape)
        train_losses = []
        train_start = time.time()


        all_losses = RunEpoch('test',
                              model,
                              train_data=train_data,
                              val_data=train_data,
                              opt=None,
                              scheduler=None,
                              batch_size=1024,
                              log_every=500,
                              table_bits=table_bits,
                              return_losses=True)

        model_nats = np.mean(all_losses)
        results[i] = model_nats

    for key in results.keys():
        print (key, results[key])

def test_for_drift(seed=0, bootstrap=10000, sample_size=16):

    torch.manual_seed(0)
    np.random.seed(0)

    def offline_phase(table, reference, model, table_bits, simulations, sample_size, fig):
        t1 = time.time()
        train_data = common.TableDataset(table, reference)
        #train_data.tuples = train_data.tuples[:int(len(train_data.tuples)*10/12)]

        avg_ll = []
        for i in range(simulations):
            idx = np.random.randint(len(train_data.tuples), size=sample_size)
            idx_tensor = torch.from_numpy(idx)

            sample = torch.index_select(train_data.tuples,0,idx_tensor)
    

            train_losses = []
            train_start = time.time()


            all_losses = RunEpoch('test',
                                  model,
                                  train_data=sample,
                                  val_data=sample,
                                  opt=None,
                                  batch_size=1024,
                                  log_every=500,
                                  table_bits=table_bits,
                                  return_losses=True)

            avg_ll.append(np.mean(all_losses))            
        t2 = time.time()

        if fig is not None:
            fig.hist(avg_ll, bins=int(len(avg_ll)/10), label='original')

        return np.mean(avg_ll), 2*np.var(avg_ll), t2-t1, fig# 2*np.var(avg_ll), t2-t1

    def online_phase_sample(table, reference, train_data, table_bits, mean, threshold, batch, fig, label):
        t1 = time.time()
        data = common.TableDataset(table, reference)

        losses = []
        for i in range(1000):
            idx = np.random.randint(len(data.tuples), size=batch)
            idx_tensor = torch.from_numpy(idx)

            sample = torch.index_select(data.tuples,0,idx_tensor)


            loss = RunEpoch('test',
                                  model,
                                  train_data=sample,
                                  val_data=sample,
                                  opt=None,
                                  batch_size=batch,
                                  log_every=500,
                                  table_bits=table_bits,
                                  return_losses=False)

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

            

    def online_phase_stream(table, reference, train_data, table_bits, mean, threshold, batch, fig, label):
        t1 = time.time()
        data = common.TableDataset(table, reference)


        #bs = int(len(data)/num_splits)
        bs = batch
        all_losses = RunEpoch('test',
                              model,
                              train_data=None,
                              val_data=data,
                              opt=None,
                              batch_size=bs,
                              log_every=500,
                              table_bits=table_bits,
                              return_losses=True)

        model_nats_new = np.mean(all_losses)

        stat = np.abs(mean - model_nats_new)
        #print (scipy.stats.norm.ppf(1-p, scale=variance))
        t2 = time.time()

        if fig is not None:
            fig.hist(all_losses, bins=int(len(all_losses)/10), label=label)

        ind = []
        ood = []
        for loss in all_losses:
            stat = np.abs(loss - mean)
            if stat > threshold:
                ood.append(loss)
            else:
                ind.append(loss)


        print("mean: {:0.4f}".format(mean), "threshold: {:0.4f}".format(threshold))
        print("number of all samples: {}".format(len(all_losses)))
        print("number of ood detected: {}".format(len(ood)))
        print("number of ind detected: {}".format(len(ind)))
        if label == "ind":
            fpr = len(ood)/len(all_losses)
            print ("FPR = {:0.4f}".format(fpr))
            return fig, fpr
        elif label == "ood":
            fnr = len(ind)/len(all_losses)
            print ("FNR = {:0.4f}".format(fnr))
            return fig, fnr


    assert args.dataset in ['dmv-tiny', 'dmv','tpcds','census','forest','imdb']
    if args.dataset == 'dmv-tiny':
        table = datasets.LoadDmv('dmv-tiny.csv')
    elif args.dataset == 'dmv':
        previous_table = datasets.LoadData(root='./data/', filename='train.csv', cols=None)
        ind_data = datasets.LoadData(root='./data/', filename='test_IND.csv', cols=None)
        ood_data = datasets.LoadData(root='./data/', filename='test_OOD.csv', cols=None)
        reference = datasets.LoadData(root='./data/', filename='main.csv', cols=None)
    elif args.dataset == 'tpcds':
        table = datasets.LoadSingleFile()
    elif args.dataset == 'census':
        previous_table = datasets.LoadData(root='./data/', filename='train.csv', cols=None)
        ind_data = datasets.LoadData(root='./data/', filename='test_IND.csv', cols=None)
        ood_data = datasets.LoadData(root='./data/', filename='test_OOD.csv', cols=None)
        reference = datasets.LoadData(root='./data/', filename='main.csv', cols=None)
    elif args.dataset == 'forest':
        previous_table = datasets.LoadData(root='./data/', filename='train.csv', cols=None)
        ind_data = datasets.LoadData(root='./data/', filename='test_IND.csv', cols=None)
        ood_data = datasets.LoadData(root='./data/', filename='test_OOD.csv', cols=None)
        reference = datasets.LoadData(root='./data/', filename='main.csv', cols=None)

    elif args.dataset == "imdb":
        previous_table = datasets.LoadIMDB(filename='title_join_mc_mii_b2010.csv')
        new_table = datasets.LoadIMDB(filename="title_join_mc_mii_a2010.csv")
        reference = datasets.LoadIMDB(filename="title_join_mc_mii.csv")

    table_bits = Entropy(
        previous_table,
        previous_table.data.fillna(value=0).groupby([c.name for c in previous_table.columns
                                           ]).size(), [2])

    fixed_ordering = None

    if args.order is not None:
        print('Using passed-in order:', args.order)
        fixed_ordering = args.order

    
    if args.dataset in ['dmv-tiny', 'dmv', 'tpcds','census','forest','imdb']:
        model = MakeMade(
            scale=args.fc_hiddens,
            cols_to_train=reference.columns,
            seed=seed,
            fixed_ordering=fixed_ordering,
        )

        model.load_state_dict(torch.load(args.glob))
        model.eval()
    else:
        assert False, args.dataset

    mb = ReportModel(model)

    fig, ax = plt.subplots()
    mean, threshold, time_off, ax = offline_phase(previous_table, reference, model, table_bits, simulations=bootstrap, sample_size=sample_size, fig=ax)
    fprs = []
    fnrs = []
    num_tuples = [1000]#[i for i in range(1, 2000, 100)]
    for n in num_tuples:
        ax, rate = online_phase_sample(ind_data, reference, model, table_bits, mean, threshold, batch=n, fig=ax, label='ind')
        fprs.append(rate)
        ax, rate = online_phase_sample(ood_data, reference, model, table_bits, mean, threshold, batch=n, fig=ax, label='ood')
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
    ax.set_xlabel('number of tuples')
    ax.set_ylabel('FPR/FNR rate')
    #ax.set_xticks(num_tuples)
    #ax.set_xticklabels(num_tuples, rotation=45)
    fig.savefig('errors.png')


if __name__ == '__main__':
    #Main()
    #MainAll(modelspath='models_to_eval/')
    #LLTest(modelspath="models_to_eval/")
    test_for_drift()
