import sys

import numpy as np
import time
from torch import nn
from torch import optim
import torch

from utils import seed_all, baseTrain, baseTest, LightGCN
from utils import MF, DMF, GMF, NMF, BPR


class Scratch(object):
    def __init__(self, param, model_type):
        # model param
        self.n_user = param.n_user
        self.n_item = param.n_item
        self.k = param.k
        self.model_type = model_type

        # training param
        self.seed = param.seed
        self.lr = param.lr
        self.epochs = param.epochs
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.pos_dir = param.pos_data

        # log
        self.log = {'train_loss': [],
                    'test_rmse': [],
                    'test_ndcg': [],
                    'test_hr': [],
                    'total_rmse': [],
                    'total_ndcg': [],
                    'total_hr': [],
                    'time': []}

        if self.model_type in ['mf']:
            self.loss_fn = 'point-wise'
            # self.loss_fn = nn.MSELoss(reduction='sum')
            # self.is_rmse = True
            # self.loss_fn = nn.BCELoss()
            # self.is_rmse = False
        elif self.model_type == 'bpr':
            self.loss_fn = 'pair-wise'

        elif self.model_type in ['dmf', 'gmf', 'nmf']:
            self.layers = param.layers
            # self.loss_fn = nn.BCELoss()
            self.loss_fn = 'point-wise'
            self.is_rmse = False

        elif self.model_type == 'lightgcn':
            self.loss_fn = 'gcn_bpr_loss'

    def train(self, train_data, test_data, active_test_data, inactive_test_data, verbose=2, save_dir='',
              id=0, given_model=''):
        print('Using device:', self.device)
        # seed for reproducibility
        seed_all(self.seed)

        # build model
        if given_model == '':
            if self.model_type == 'mf':
                model = MF(self.n_user, self.n_item, self.k).to(self.device)
            elif self.model_type == 'bpr':
                model = BPR(self.n_user, self.n_item, self.k).to(self.device)
            elif self.model_type == 'gmf':
                model = GMF(self.n_user, self.n_item, self.k).to(self.device)
            elif self.model_type == 'nmf':
                model = NMF(self.n_user, self.n_item, self.k, self.layers).to(self.device)
            elif self.model_type == 'dmf':
                model = DMF(self.n_user, self.n_item, self.k, self.layers).to(self.device)
            elif self.model_type == 'lightgcn':
                model = LightGCN(self.n_user, self.n_item).to(self.device)
        else:
            model = given_model.to(self.device)

        # set optimizer

        opt = optim.Adam(model.parameters(), lr=self.lr)

        # main loop
        best_ndcg = 0
        best_hr = 0
        count_dec = 0
        total_time = 0

        pos_dict = np.load(self.pos_dir, allow_pickle=True).item()

        for t in range(self.epochs):
            # if verbose == 2:
            print(f'Epoch: [{t + 1:>3d}/{self.epochs:>3d}] --------------------')
            epoch_start = time.time()

            # train
            train_loss = baseTrain(train_data, model, self.loss_fn, opt, self.device, verbose)

            train_time = time.time() - epoch_start

            print(train_time)
            total_time += train_time
            user_mapping = None
            pos_mapping = None
            if self.model_type == 'lightgcn':
                user_mapping = train_data.dataset.user_mapping
                pos_mapping = train_data.dataset.pos_mapping

            test_ndcg, test_hr = baseTest(test_data, model, self.loss_fn, self.device, verbose, pos_dict, self.n_item,
                                          20,user_mapping,pos_mapping)


            # print info
            epoch_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - epoch_start))

            # if verbose == 2:
            print('Time:', epoch_time)
            print('train_loss:', train_loss)
            print('test_ndcg:', test_ndcg)
            print('test_hr:', test_hr)

            # save log
            self.log['train_loss'].append(train_loss)
            self.log['test_ndcg'].append(test_ndcg)
            self.log['test_hr'].append(test_hr)
            self.log['time'].append(epoch_time)

            if test_ndcg > best_ndcg:
                count_dec = 0
                best_ndcg = test_ndcg
                best_hr = test_hr
                # if len(save_dir) > 0:
                #     torch.save(model.state_dict(), save_dir + '/model' + '.pth')
                #     torch.save(model.user_mat.weight.detach().cpu().numpy(), save_dir + '/user_mat' + '.npy')
            else:
                count_dec += 1

            if count_dec > 5:
                break

        if active_test_data is not None:
            active_ndcg, active_hr = baseTest(active_test_data, model, self.loss_fn, self.device, verbose, pos_dict,
                                              self.n_item, 20,user_mapping,pos_mapping)
            print('active_test_ndcg:', active_ndcg)
        else:
            active_ndcg = 0
            active_hr = 0
        inactive_ndcg, inactive_hr = baseTest(inactive_test_data, model, self.loss_fn, self.device, verbose,
                                              pos_dict,
                                              self.n_item,
                                              20,user_mapping,pos_mapping)
        print('inactive_test_ndcg:', inactive_ndcg)
        print('best:!!!!!!!!!!!!!!!')
        result = {'time': total_time, 'ndcg': best_ndcg, 'hr': best_hr, 'active_ndcg': active_ndcg,
                  'active_hr': active_hr, 'inactive_ndcg': inactive_ndcg, 'inactive_hr': inactive_hr}

        # save model
        # torch.save(model.state_dict(), f'results/gowalla/{self.model_type}.pth')

        # load mat
        # user_mat = np.load('user_mat.npy')
        # item_mat = np.load('item_mat.npy')

        # np.save(save_dir + '/log.npy', self.log)
        # load log
        # log = np.load('log.npy', allow_pickle=True).item()

        return model, result
