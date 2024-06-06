import random
import time

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from utils import kmeans, ot_cluster


def delete(ratings, del_type, del_per):
    if del_type == 'random':
        np.random.seed(42)
        user_counts = ratings['uid'].value_counts()
        num_delete_users = int(len(user_counts) * del_per / 100)
        delete_users = np.random.choice(user_counts.index, num_delete_users, replace=False).tolist()
    elif del_type == 'core':
        user_counts = ratings['uid'].value_counts()
        num_delete_users = int(len(user_counts) * del_per / 100)
        delete_users = user_counts.index[:num_delete_users].tolist()
    elif del_type == 'edge':
        user_counts = ratings['uid'].value_counts(ascending=True)
        num_delete_users = int(len(user_counts) * del_per / 100)
        delete_users = user_counts.index[:num_delete_users].tolist()
    return delete_users


def readRating_full(train_dir, test_dir, del_type='random', del_per=5):
    train_ratings = pd.read_csv(train_dir, sep=',')
    test_ratings = pd.read_csv(test_dir, sep=',')

    if del_per > 0:
        del_user = delete(train_ratings, del_type, del_per)

        train_ratings = train_ratings[~train_ratings['uid'].isin(del_user)].reset_index(drop=True)
        test_ratings = test_ratings[~test_ratings['uid'].isin(del_user)].reset_index(drop=True)
    # active and inactive
    user_counts = train_ratings['uid'].value_counts()
    num_active_users = int(len(user_counts) * 5 / 100)

    active_users = user_counts.index[:num_active_users].tolist()
    inactive_users = user_counts.index[num_active_users:].tolist()
    # test set
    active_ratings = test_ratings[test_ratings['uid'].isin(active_users)].reset_index(drop=True)
    inactive_ratings = test_ratings[test_ratings['uid'].isin(inactive_users)].reset_index(drop=True)

    train_rating_lists = [train_ratings['uid'], train_ratings['iid'], train_ratings['val']]
    test_rating_lists = [test_ratings['uid'], test_ratings['iid'], test_ratings['val']]

    active_ratings = [active_ratings['uid'], active_ratings['iid'], active_ratings['val']]
    inactive_ratings = [inactive_ratings['uid'], inactive_ratings['iid'], inactive_ratings['val']]

    return train_rating_lists, test_rating_lists, active_ratings, inactive_ratings


def readRating_group(train_dir, test_dir, del_type='random', del_per=5, learn_type='sisa', num_groups=5,
                     dataset='ml-100k'):
    train_ratings = pd.read_csv(train_dir, sep=',')
    test_ratings = pd.read_csv(test_dir, sep=',')

    if del_per > 0:
        del_user = delete(train_ratings, del_type, del_per)

        train_ratings = train_ratings[~train_ratings['uid'].isin(del_user)].reset_index(drop=True)
        test_ratings = test_ratings[~test_ratings['uid'].isin(del_user)].reset_index(drop=True)
    # active and inactive
    user_counts = train_ratings['uid'].value_counts()
    num_active_users = int(len(user_counts) * 5 / 100)

    active_users = user_counts.index[:num_active_users].tolist()
    inactive_users = user_counts.index[num_active_users:].tolist()
    # test set

    if learn_type == 'sisa':
        # random choice
        # ratings = ratings.sample(frac=1).reset_index(drop=True)
        # random.seed(42)

        unique_users = train_ratings['uid'].unique()
        random.shuffle(unique_users)

        group_size = len(unique_users) // num_groups
        user_groups = [unique_users[i * group_size: (i + 1) * group_size] for i in range(num_groups)]

        if len(unique_users) % num_groups != 0:
            for i in range(len(unique_users) % num_groups):
                user_groups[i] = np.append(user_groups[i], unique_users[num_groups * group_size + i])

        train_rating_groups = [train_ratings[train_ratings['uid'].isin(group)].reset_index(drop=True) for group in
                               user_groups]
        test_rating_groups = [test_ratings[test_ratings['uid'].isin(group)].reset_index(drop=True) for group in
                              user_groups]
    elif learn_type == 'receraser':
        start_time = time.time()
        # 加载用户嵌入
        user_embeddings = np.load(f'results/user_emb/{dataset}_mf_emb.npy', allow_pickle=True).item()
        # 获取所有用户的唯一ID
        unique_users = train_ratings['uid'].unique()
        # 提取用户嵌入矩阵
        user_mat = np.array([user_embeddings[user_id][0] for user_id in unique_users])

        _, labels = kmeans(user_mat, num_groups, True, 30)

        # 创建用户分组
        user_groups = [[] for _ in range(num_groups)]
        for user_id, label in zip(unique_users, labels):
            user_groups[int(label)].append(user_id)

        train_rating_groups = [train_ratings[train_ratings['uid'].isin(group)].reset_index(drop=True) for group in
                               user_groups]
        test_rating_groups = [test_ratings[test_ratings['uid'].isin(group)].reset_index(drop=True) for group in
                              user_groups]

        print(f'Grouping time: {time.time() - start_time}')

    elif learn_type == 'ultraue':
        start_time = time.time()
        # 加载用户嵌入
        user_embeddings = np.load(f'results/user_emb/{dataset}_mf_emb.npy', allow_pickle=True).item()
        # 获取所有用户的唯一ID
        unique_users = train_ratings['uid'].unique()
        # 提取用户嵌入矩阵
        user_mat = np.array([user_embeddings[user_id][0] for user_id in unique_users])

        _, labels = ot_cluster(user_mat, num_groups)

        # 创建用户分组
        user_groups = [[] for _ in range(num_groups)]
        for user_id, label in zip(unique_users, labels):
            user_groups[int(label)].append(user_id)

        train_rating_groups = [train_ratings[train_ratings['uid'].isin(group)].reset_index(drop=True) for group in
                               user_groups]
        test_rating_groups = [test_ratings[test_ratings['uid'].isin(group)].reset_index(drop=True) for group in
                              user_groups]

        print(f'Grouping time: {time.time() - start_time}')

    # 初始化保存active和inactive ratings的列表
    active_groups = []
    inactive_groups = []

    # 生成每组中的active_rating和inactive_rating
    for i, ratings in enumerate(test_rating_groups):
        active_ratings = ratings[ratings['uid'].isin(active_users)].reset_index(drop=True)
        inactive_ratings = ratings[ratings['uid'].isin(inactive_users)].reset_index(drop=True)
        active_groups.append(active_ratings)
        inactive_groups.append(inactive_ratings)
        # 输出每组的active user和inactive user的个数
        print(f"Group {i + 1} active users: {len(active_ratings['uid'].unique())}")
        print(f"Group {i + 1} inactive users: {len(inactive_ratings['uid'].unique())}")

    train_rating_groups = [[ratings['uid'], ratings['iid'], ratings['val']] for ratings in train_rating_groups]
    test_rating_groups = [[ratings['uid'], ratings['iid'], ratings['val']] for ratings in test_rating_groups]
    active_groups = [[ratings['uid'], ratings['iid'], ratings['val']] for ratings in active_groups]
    inactive_groups = [[ratings['uid'], ratings['iid'], ratings['val']] for ratings in inactive_groups]

    return train_rating_groups, test_rating_groups, active_groups, inactive_groups


class RatingData(Dataset):
    def __init__(self, rating_array):
        super(RatingData, self).__init__()
        self.rating_array = rating_array

        self.users = self.rating_array[0].astype(int)
        self.items = self.rating_array[1].astype(int)
        self.ratings = self.rating_array[2].astype(float)

        self.total_users = self.users
        self.total_items = self.items
        self.total_ratings = self.ratings

        self.pos_dict = {user: set() for user in self.users}
        for user, item in zip(self.users, self.items):
            self.pos_dict[user].add(item)

    def __len__(self):
        return len(self.total_users)

    def __getitem__(self, idx):
        user = self.total_users[idx]
        item = self.total_items[idx]
        rating = self.total_ratings[idx]
        return (torch.tensor(user, dtype=torch.long),
                torch.tensor(item, dtype=torch.long),
                torch.tensor(rating, dtype=torch.float32))

    def average_ng_sample(self):
        # total pos:neg = 1:1

        user_list = []
        item_list = []
        val_list = []

        new_rating_array = []

        num_item = len(self.items.unique())
        unique_users = self.users.unique()
        ng_sample = int(len(self.users) / len(unique_users))

        if ng_sample > 0:
            for userid in unique_users:
                for i in range(ng_sample):
                    j = np.random.randint(num_item)
                    while j in self.pos_dict[userid]:
                        j = np.random.randint(num_item)
                    user_list.append(userid)
                    item_list.append(j)
                    val_list.append(0)

            new_rating_array.append(pd.concat([self.rating_array[0], pd.Series(user_list)], ignore_index=True))
            new_rating_array.append(pd.concat([self.rating_array[1], pd.Series(item_list)], ignore_index=True))
            new_rating_array.append(pd.concat([self.rating_array[2], pd.Series(val_list)], ignore_index=True))

        self.total_users = new_rating_array[0].astype(int)
        self.total_items = new_rating_array[1].astype(int)
        self.total_ratings = new_rating_array[2].astype(float)

    def ng_sample(self, ng_sample):

        user_list = []
        item_list = []
        val_list = []

        new_rating_array = []

        num_item = max(self.rating_array[1].unique())

        if ng_sample > 0:
            for userid in self.users:
                for i in range(ng_sample):
                    j = np.random.randint(num_item)
                    while j in self.pos_dict[userid]:
                        j = np.random.randint(num_item)
                    user_list.append(userid)
                    item_list.append(j)
                    val_list.append(0)

            new_rating_array.append(pd.concat([self.rating_array[0], pd.Series(user_list)], ignore_index=True))
            new_rating_array.append(pd.concat([self.rating_array[1], pd.Series(item_list)], ignore_index=True))
            new_rating_array.append(pd.concat([self.rating_array[2], pd.Series(val_list)], ignore_index=True))

        self.total_users = new_rating_array[0].astype(int)
        self.total_items = new_rating_array[1].astype(int)
        self.total_ratings = new_rating_array[2].astype(float)


class PairData(Dataset):
    def __init__(self, rating_array, pos_dir):
        super(PairData, self).__init__()
        self.pos_dir = pos_dir
        self.rating_array = rating_array

        self.users = self.rating_array[0].astype(int)
        self.pos_items = self.rating_array[1].astype(int)
        self.neg_items = self.rating_array[1].astype(int)

        self.user_mapping = {index: i for i, index in enumerate(self.rating_array[0].unique())}
        self.pos_mapping = {index: i for i, index in enumerate(self.rating_array[1].unique())}
        src = [self.user_mapping[index] for index in rating_array[0]]
        dst = [self.pos_mapping[index] for index in rating_array[1]]

        self.edge_index = [[], []]
        for i in range(len(self.users)):
            self.edge_index[0].append(src[i])
            self.edge_index[1].append(dst[i])

        # self.ratings = self.rating_array[2].astype(float)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        pos = self.pos_items[idx]
        neg = self.neg_items[idx]
        return (torch.tensor(user, dtype=torch.long),
                torch.tensor(pos, dtype=torch.long),
                torch.tensor(neg, dtype=torch.long))

    def ng_sample(self, ng_sample):

        user_list = []

        neg_item_list = []

        new_rating_array = []
        num_item = max(self.rating_array[1].unique())

        pos_dict = np.load(self.pos_dir, allow_pickle=True).item()
        for userid in self.users:
            for i in range(ng_sample):
                j = np.random.randint(num_item)
                while j in pos_dict[userid]:
                    j = np.random.randint(num_item)
                user_list.append(userid)
                neg_item_list.append(j)

        # new_rating_array.append(pd.Series(neg_item_list))

        # self.neg_items = new_rating_array[1].astype(int)

        self.neg_items = neg_item_list


def loadData(data, batch=30000, n_worker=24, shuffle=True):
    '''
    Parameters
    ----------
    data:   RatingData object 
    '''

    return DataLoader(data, batch_size=batch, shuffle=shuffle, num_workers=n_worker)


def readSparseMat(dir, n_user, n_item, max_rating=5):
    ratings = pd.read_csv(dir, header=None, sep=',')

    row = ratings[0].astype(int).values
    col = ratings[1].astype(int).values
    val = ratings[2].astype(float).values / max_rating
    # ind = np.ones_like(val, dtype=int)

    val_mat = coo_matrix((val, (row, col)), shape=(n_user, n_item), dtype=np.float16)
    # ind_mat = coo_matrix((ind, (row, col)), shape=(n_user, n_item), dtype=np.float16)  # set to float! int will cause error in kmeans

    return val_mat.tocsr()  # , ind_mat.tocsr()
