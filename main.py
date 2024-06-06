import argparse

import numpy as np

from config import InsParam, Instance

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ml-1m', help='dataset name')
parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--worker', type=int, default=8, help='number of CPU workers')
parser.add_argument('--verbose', type=int, default=2, help='verbose type')
parser.add_argument('--group', type=int, default=5, help='number of groups')
parser.add_argument('--layer', nargs='+', default=[64, 32], help='setting of layers')
parser.add_argument('--learn', type=str, default='sisa', help='type of learning and unlearning')
parser.add_argument('--delper', type=int, default=5, help='deleted user proportion')
parser.add_argument('--deltype', type=str, default='random', help='unlearn data selection')
parser.add_argument('--model', type=str, default='mf', help='rec model')


# this is an example of main
def main():
    # read parser
    args = parser.parse_args()

    assert args.model in ['mf', 'dmf', 'bpr', 'lightgcn']
    model = args.model

    assert args.dataset in ['ml-100k', 'ml-1m', 'adm', 'gowalla']
    dataset = args.dataset

    assert args.epoch > 0
    epochs = args.epoch

    assert args.worker > 0
    n_worker = args.worker

    '''
    print verbose
    if verbose == 0, print nothing
    if verbose == 1, print every epoch
    if verbose == 2, print every batch
    '''
    assert args.verbose in [0, 1, 2]
    verbose = args.verbose

    assert args.group >= 0
    n_group = args.group

    for i in args.layer:
        assert type(i) == int
    layers = args.layer

    assert args.learn in ['retrain','sisa','receraser','ultraue']
    learn_type = args.learn

    assert args.delper >= 0
    del_per = args.delper

    assert args.deltype in ['random', 'core', 'edge']
    del_type = args.deltype

    # initiate instance
    param = InsParam(dataset, model, epochs, n_worker, layers, n_group, del_per, learn_type, del_type)
    ins = Instance(param)

    # begin instance
    ins.run(verbose=verbose)


if __name__ == '__main__':
    main()
