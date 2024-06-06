import os
import warnings
from os.path import abspath, join, dirname, exists

import numpy as np
from scratch import Scratch
from utils import saveObject

from read import RatingData, PairData
from read import loadData, readRating_full, readRating_group

DATA_DIR = abspath(join(dirname(__file__), 'data'))
SAVE_DIR = abspath(join(dirname(__file__), 'result'))


class InsParam(object):
    def __init__(self, dataset='toy', model='mf', epochs=50, n_worker=24, layers=[32], n_group=5, del_per=5,
                 learn_type='retrain',
                 del_type='random'):
        # model param
        self.k = 32  # dimension of embedding
        self.lam = 0.1  # regularization coefficient
        self.layers = layers  # structure of FC layers in DMF

        # training param
        self.seed = 42
        self.n_worker = n_worker
        self.batch = 1024
        self.lr = 0.001
        self.epochs = epochs
        self.n_group = n_group
        self.learn_type = learn_type

        # dataset-varied param
        self.del_rating = []  # 2d array/list [[uid, iid], ...]
        self.dataset = dataset
        self.max_rating = 5
        self.del_per = del_per
        self.del_type = del_type
        self.model = model

        if dataset == 'ml-100k':
            self.train_dir = DATA_DIR + '/ml-100k/train.csv'
            self.test_dir = DATA_DIR + '/ml-100k/test.csv'
            self.pos_data = DATA_DIR + '/ml-100k/pos_dict.npy'
            self.n_user = 943
            self.n_item = 1349

        elif dataset == 'ml-1m':
            self.train_dir = DATA_DIR + '/ml-1m/train.csv'
            self.test_dir = DATA_DIR + '/ml-1m/test.csv'
            self.pos_data = DATA_DIR + '/ml-1m/pos_dict.npy'

            self.n_user = 6040
            self.n_item = 3416

        elif dataset == 'adm':
            self.train_dir = DATA_DIR + '/adm/train.csv'
            self.test_dir = DATA_DIR + '/adm/test.csv'
            self.pos_data = DATA_DIR + '/adm/pos_dict.npy'

            self.n_user = 22878
            self.n_item = 115082

        elif dataset == 'gowalla':
            self.train_dir = DATA_DIR + '/gowalla/train.csv'
            self.test_dir = DATA_DIR + '/gowalla/test.csv'
            self.pos_data = DATA_DIR + '/gowalla/pos_dict.npy'

            self.n_user = 64115
            self.n_item = 164532


class Instance(object):
    def __init__(self, param):
        self.param = param
        prefix = '/test/' if self.param.del_type == 'test' else '/' + str(
            self.param.del_per) + '/' + self.param.del_type + '/'
        self.name = prefix + self.param.dataset + '_g_' + str(
            self.param.n_group)  # time.strftime("%Y%m%d_%H%M%S", time.localtime())
        param_dir = SAVE_DIR + self.name
        if exists(param_dir) == False:
            os.makedirs(param_dir)

        # save param
        saveObject(param_dir + '/param', self.param)  # loadObject(dir + '/param')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # np.save(param_dir + '/deletion', deletion)  # np.load('deletion.npy')

    def read(self):
        learn_type = self.param.learn_type
        del_type = self.param.del_type
        del_per = self.param.del_per
        group = self.param.n_group
        if learn_type == 'retrain':
            train_rating, test_rating, active_rating, inactive_rating = readRating_full(self.param.train_dir,
                                                                                        self.param.test_dir, del_type,
                                                                                        del_per)
        else:
            train_rating, test_rating, active_rating, inactive_rating = readRating_group(self.param.train_dir,
                                                                                         self.param.test_dir, del_type,
                                                                                         del_per, learn_type, group,self.param.dataset)

        return train_rating, test_rating, active_rating, inactive_rating

    def runModel(self, model_type='dmf', verbose=2):
        print(self.name, 'begin:')
        # read raw data
        train_rating, test_rating, active_rating, inactive_rating = self.read()

        if self.param.learn_type == 'retrain':
            # load data
            if model_type == 'mf' or model_type == 'dmf':
                train_data = loadData(RatingData(train_rating), self.param.batch,
                                      self.param.n_worker,
                                      True)

            elif model_type in ['bpr', 'lightgcn']:
                train_data = loadData(PairData(train_rating, self.param.pos_data), self.param.batch,
                                      self.param.n_worker,
                                      True)

            test_data = loadData(RatingData(test_rating), len(test_rating[0]), self.param.n_worker, False)
            active_test_data = loadData(RatingData(active_rating), len(active_rating[0]), self.param.n_worker,
                                        False)
            inactive_test_data = loadData(RatingData(inactive_rating), len(inactive_rating[0]), self.param.n_worker,
                                          False)

            model = Scratch(self.param, model_type)
            model, result = model.train(train_data, test_data, active_test_data, inactive_test_data, verbose,
                                        given_model='')
            result.update({'model': model_type, 'dataset': self.param.dataset, 'deltype': self.param.del_type,
                           'method': self.param.learn_type})
            np.save(f'results/{self.param.learn_type}/{model_type}_{self.param.dataset}_{self.param.del_type}_{self.param.del_per}.npy',
                    result)
            print('End of training', self.name)
        else:
            group = self.param.n_group
            for i in range(group):
                # load data
                if model_type == 'mf' or model_type == 'dmf':
                    train_data = loadData(RatingData(train_rating[i]), self.param.batch,
                                          self.param.n_worker,
                                          True)
                elif model_type in ['bpr', 'lightgcn']:
                    train_data = loadData(PairData(train_rating[i], self.param.pos_data), self.param.batch,
                                          self.param.n_worker,
                                          True)

                test_data = loadData(RatingData(test_rating[i]), len(test_rating[i][0]), self.param.n_worker, False)
                if len(active_rating[i][0]) > 0:
                    active_test_data = loadData(RatingData(active_rating[i]), len(active_rating[i][0]),
                                                self.param.n_worker,
                                                False)
                else:
                    active_test_data = None
                inactive_test_data = loadData(RatingData(inactive_rating[i]), len(inactive_rating[i][0]),
                                              self.param.n_worker,
                                              False)

                model = Scratch(self.param, model_type)
                model, result = model.train(train_data, test_data, active_test_data, inactive_test_data, verbose,
                                            given_model='')
                result.update({'model': model_type, 'dataset': self.param.dataset, 'deltype': self.param.del_type,
                               'method': self.param.learn_type, 'group': i + 1})
                np.save(
                    f'results/{self.param.learn_type}/group{i + 1}_{self.param.n_group}_{model_type}_{self.param.dataset}_{self.param.del_type}_{self.param.del_per}.npy',
                    result)
                print(f'End of Group {str(i + 1)} / {group} training', self.name)

    def run(self, verbose=2):
        '''model MF'''
        # full train without deletion
        self.runModel(self.param.model, verbose)

        # retrain from scratch after deletion
        # self.__full(is_save, 'MF_retrain', 'mf', True, verbose)

        '''model DMF'''
        # full train without deletion
        # self.__full(is_save, 'DMF_full_train', 'dmf', False, verbose)

        # retrain from scratch after deletion
        # self.__full(is_save, 'DMF_retrain', 'dmf', True, verbose)

        '''model NMF'''
        # full train without deletion
        # s_time = time.time()
        # self.__full(is_save, 'NMF_full_train', 'nmf', False, verbose)
        # e_time = time.time()
        # print(e_time - s_time)

        # retrain from scratch after deletion
        # s_time = time.time()
        # self.__full(is_save, 'NMF_retrain', 'nmf', True, verbose)
        # e_time = time.time()
        # print(e_time - s_time)
        '''test model'''
        # self.__full(is_save, 'GMF_full_train', 'gmf', False, verbose)
