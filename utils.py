from itertools import combinations
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import laplacian
from sklearn.metrics.pairwise import pairwise_distances

from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj

import warnings
import ot
import torch
from torch import nn, Tensor
import random

import numpy as np
import torch
import pandas as pd
import scipy.sparse as sp

STD = 0.01


def hit(gt_items, pred_items):
    hr = 0
    for gt_item in gt_items:
        if gt_item in pred_items:
            hr = hr + 1

    return hr / len(gt_items)


def ndcg(gt_items, pred_items):
    dcg = 0
    idcg = 0

    for gt_item in gt_items:
        if gt_item in pred_items:
            index = pred_items.index(gt_item)
            dcg = dcg + np.reciprocal(np.log2(index + 2))

    for index in range(len(gt_items)):
        idcg = idcg + np.reciprocal(np.log2(index + 2))

    return dcg / idcg


##################### 
# model training
##################### 

# seed everything
def seed_all(seed):
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class MF(nn.Module):
    def __init__(self, n_user, n_item, k=16):
        super(MF, self).__init__()
        self.k = k
        self.user_mat = nn.Embedding(n_user, k)
        self.item_mat = nn.Embedding(n_item, k)
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.user_mat.weight, std=0.01)
        nn.init.normal_(self.item_mat.weight, std=0.01)

    def forward(self, uid, iid):
        return (self.user_mat(uid) * self.item_mat(iid)).sum(dim=1)


class BPR(nn.Module):
    def __init__(self, n_user, n_item, k=16):
        super(BPR, self).__init__()
        self.k = k
        self.user_mat = nn.Embedding(n_user, k)
        self.item_mat = nn.Embedding(n_item, k)
        self.func = nn.Sigmoid()
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.user_mat.weight, std=0.01)
        nn.init.normal_(self.item_mat.weight, std=0.01)

    def forward(self, uid, iid):
        return (self.user_mat(uid) * self.item_mat(iid)).sum(dim=1)

        # r_pos = (self.user_mat(uid) * self.item_mat(pos_id)).sum(dim=1)
        # r_neg = (self.user_mat(uid) * self.item_mat(neg_id)).sum(dim=1)

        # return self.func(r_pos - r_neg)


# build model Generalized MF
class GMF(nn.Module):
    def __init__(self, n_user, n_item, k=16):
        super(GMF, self).__init__()
        self.k = k
        self.user_mat = nn.Embedding(n_user, k)
        self.item_mat = nn.Embedding(n_item, k)

        self.affine = nn.Linear(self.k, 1)
        self.logistic = nn.Sigmoid()

        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.user_mat.weight, std=STD)
        nn.init.normal_(self.item_mat.weight, std=STD)

        nn.init.xavier_uniform_(self.affine.weight)

    def forward(self, uid, iid):
        user_embedding = self.user_mat(uid)
        item_embedding = self.item_mat(iid)
        logits = self.affine(torch.mul(user_embedding, item_embedding))
        rating = self.logistic(logits)
        return rating.squeeze()


# build model DMF
class DMF(nn.Module):
    def __init__(self, n_user, n_item, k=16, layers=[64, 32]):
        super(DMF, self).__init__()
        self.k = k
        self.user_mat = nn.Embedding(n_user, k)
        self.item_mat = nn.Embedding(n_item, k)
        self.layers = [k]
        self.layers += layers
        self.user_fc = nn.ModuleList()
        self.item_fc = nn.ModuleList()
        self.cos = nn.CosineSimilarity()

        for (in_size, out_size) in zip(self.layers[:-1], self.layers[1:]):
            self.user_fc.append(nn.Linear(in_size, out_size))
            self.item_fc.append(nn.Linear(in_size, out_size))
            self.user_fc.append(nn.ReLU())
            self.item_fc.append(nn.ReLU())

        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.user_mat.weight, std=STD)
        nn.init.normal_(self.item_mat.weight, std=STD)

        for i in self.modules():
            if isinstance(i, nn.Linear):
                nn.init.xavier_uniform_(i.weight)
                if i.bias is not None:
                    i.bias.data.zero_()

    def forward(self, uid, iid):
        user_embedding = self.user_mat(uid)
        item_embedding = self.item_mat(iid)
        for i in range(len(self.user_fc)):
            user_embedding = self.user_fc[i](user_embedding)
            item_embedding = self.item_fc[i](item_embedding)
        rating = self.cos(user_embedding, item_embedding)
        return rating.squeeze()


# build model Neural MF
class NMF(nn.Module):
    def __init__(self, n_user, n_item, k=16, layser=[64, 32]):
        super(NMF, self).__init__()
        self.k = k
        self.k_mlp = int(layser[0] / 2)

        self.user_mat_mf = nn.Embedding(n_user, k)
        self.item_mat_mf = nn.Embedding(n_item, k)
        self.user_mat_mlp = nn.Embedding(n_user, self.k_mlp)
        self.item_mat_mlp = nn.Embedding(n_item, self.k_mlp)

        self.layers = layser
        self.fc = nn.ModuleList()
        for (in_size, out_size) in zip(self.layers[:-1], self.layers[1:]):
            self.fc.append(nn.Linear(in_size, out_size))
            self.fc.append(nn.ReLU())

        self.affine = nn.Linear(self.layers[-1] + self.k, 1)
        self.logistic = nn.Sigmoid()
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.user_mat_mf.weight, std=STD)
        nn.init.normal_(self.item_mat_mf.weight, std=STD)
        nn.init.normal_(self.user_mat_mlp.weight, std=STD)
        nn.init.normal_(self.item_mat_mlp.weight, std=STD)

        for i in self.modules():
            if isinstance(i, nn.Linear):
                nn.init.xavier_uniform_(i.weight)
                if i.bias is not None:
                    i.bias.data.zero_()
        # for i in self.fc:
        #     if isinstance(i, nn.Linear):
        #         nn.init.xavier_uniform_(i.weight)
        #         if i.bias is not None:
        #             i.bias.data.zero_()

        # nn.init.xavier_uniform_(self.affine.weight)
        # if self.affine.bias is not None:
        #     self.affine.bias.data.zero_()

    def forward(self, uid, iid):
        user_embedding_mlp = self.user_mat_mlp(uid)
        item_embedding_mlp = self.item_mat_mlp(iid)

        user_embedding_mf = self.user_mat_mf(uid)
        item_embedding_mf = self.item_mat_mf(iid)

        mlp_vec = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        mf_vec = torch.mul(user_embedding_mf, item_embedding_mf)

        for i in range(len(self.fc)):
            mlp_vec = self.fc[i](mlp_vec)

        vec = torch.cat([mlp_vec, mf_vec], dim=-1)
        logits = self.affine(vec)
        rating = self.logistic(logits)
        return rating.squeeze()


class LightGCN(MessagePassing):
    def __init__(self, num_users, num_items, embedding_dim=64, K=3, add_self_loops=False, **kwargs):
        """
        Args:
            num_users (int): 用户数量
            num_items (int): 电影数量
            embedding_dim (int, optional): 嵌入维度，设置为64，后续可以调整观察效果
            K (int, optional): 传递层数，设置为3，后续可以调整观察效果
            add_self_loops (bool, optional): 传递时加不加自身节点，设置为不加
        """
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.num_users, self.num_items = num_users, num_items
        self.embedding_dim, self.K = embedding_dim, K
        self.add_self_loops = add_self_loops

        self.users_emb = nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.embedding_dim)  # e_u^0
        self.items_emb = nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.embedding_dim)  # e_i^0

        nn.init.normal_(self.users_emb.weight, std=0.1)  # 从给定均值和标准差的正态分布N(mean, std)中生成值，填充输入的张量或变量
        nn.init.normal_(self.items_emb.weight, std=0.1)

    def forward(self, edge_index: SparseTensor):
        """
        Args:
            edge_index (SparseTensor): 邻接矩阵
        Returns:
            tuple (Tensor): e_u%^k, e_u^0, e_i^k, e_i^0
        """
        # compute \tilde{A}: symmetrically normalized adjacency matrix
        # edge_index_norm = gcn_norm(
        #     edge_index, add_self_loops=self.add_self_loops)

        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight])  # E^0
        embs = [emb_0]
        emb_k = emb_0

        # 多尺度扩散
        for i in range(self.K):
            emb_k = self.propagate(edge_index, x=emb_k)
            embs.append(emb_k)

        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1)  # E^K

        users_emb_final, items_emb_final = torch.split(
            emb_final, [self.num_users, self.num_items])  # splits into e_u^K and e_i^K

        # returns e_u^K, e_u^0, e_i^K, e_i^0
        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        # computes \tilde{A} @ x
        return matmul(adj_t, x)


def gcn_bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final,
                 neg_items_emb_0,
                 lambda_val):
    """
    Args:
        users_emb_final (torch.Tensor): e_u^k
        users_emb_0 (torch.Tensor): e_u^0
        pos_items_emb_final (torch.Tensor): positive e_i^k
        pos_items_emb_0 (torch.Tensor): positive e_i^0
        neg_items_emb_final (torch.Tensor): negative e_i^k
        neg_items_emb_0 (torch.Tensor): negative e_i^0
        lambda_val (float): λ的值
    Returns:
        torch.Tensor: loss值
    """
    reg_loss = lambda_val * (users_emb_0.norm(2).pow(2) +
                             pos_items_emb_0.norm(2).pow(2) +
                             neg_items_emb_0.norm(2).pow(2))  # L2 loss L2范数是指向量各元素的平方和然后求平方根

    pos_scores = torch.mul(users_emb_final, pos_items_emb_final)
    pos_scores = torch.sum(pos_scores, dim=-1)  # 正采样预测分数
    neg_scores = torch.mul(users_emb_final, neg_items_emb_final)
    neg_scores = torch.sum(neg_scores, dim=-1)  # 负采样预测分数

    loss = -torch.mean(torch.nn.functional.softplus(pos_scores - neg_scores)) + reg_loss

    return loss


# torch train
def baseTrain(dataloader, model, loss_fn, opt, device, verbose):
    size = len(dataloader.dataset)
    train_loss = 0

    dataloader.dataset.ng_sample(1)
    # dataloader.dataset.average_ng_sample()

    model.train(True)
    if loss_fn == 'point-wise':
        loss_func = nn.BCEWithLogitsLoss()
        for batch, (user, item, rating) in enumerate(dataloader):
            user = user.to(device)
            item = item.to(device)
            rating = rating.to(device)

            pred = model(user, item)
            loss = loss_func(pred, rating)

            train_loss += loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

    elif loss_fn == 'pair-wise':
        sig_func = nn.Sigmoid()

        for batch, (user, pos, neg) in enumerate(dataloader):
            user = user.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            pred_pos = model(user, pos)
            pred_neg = model(user, neg)

            loss = (-torch.log(sig_func(pred_pos - pred_neg))).sum()

            # r_pos = (self.user_mat(uid) * self.item_mat(pos_id)).sum(dim=1)
            # r_neg = (self.user_mat(uid) * self.item_mat(neg_id)).sum(dim=1)

            # return self.func(r_pos - r_neg)

            train_loss += loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

    elif loss_fn == 'gcn_bpr_loss':
        # forward propagation
        train_sparse_edge_index = torch.tensor(dataloader.dataset.edge_index,
                                               dtype=torch.long)
        # print(train_sparse_edge_index.shape)  # 确认形状
        # print(train_sparse_edge_index[0].shape)
        # print(train_sparse_edge_index[1].shape)
        #
        # # 检查行数
        # assert train_sparse_edge_index.shape[0] == 2, "edge_index 应该有两行（src 和 dst）"
        # train_sparse_edge_index = SparseTensor(row=train_sparse_edge_index[0],col=train_sparse_edge_index[1])

        LAMBDA = 1e-6
        train_sparse_edge_index = train_sparse_edge_index.to(device)
        user_mapping = dataloader.dataset.user_mapping
        pos_mapping = dataloader.dataset.pos_mapping
        for user, pos, neg in dataloader:
            # train_sparse_edge_index = torch.cat(user,pos)
            user = torch.tensor(list(map(user_mapping.get, user.tolist())), dtype=torch.long)
            pos = torch.tensor(list(map(pos_mapping.get, pos.tolist())), dtype=torch.long)
            neg = torch.tensor(list(map(pos_mapping.get, neg.tolist())), dtype=torch.long)

            users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(
                train_sparse_edge_index)

            user, pos, neg = user.to(device), pos.to(device), neg.to(device)
            users_emb_final, users_emb_0 = users_emb_final[user], users_emb_0[user]
            pos_items_emb_final, pos_items_emb_0 = items_emb_final[
                                                       pos], items_emb_0[pos]
            neg_items_emb_final, neg_items_emb_0 = items_emb_final[
                                                       neg], items_emb_0[neg]

            # loss computation
            loss = gcn_bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final,
                                pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, LAMBDA)
            train_loss += loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

    return train_loss / size


# torch test
def baseTest(dataloader, model, loss_fn, device, verbose, pos_dict, n_items, top_k=20, user_mapping=None,
             pos_mapping=None):
    '''
    Parameters
    ----------
    models: list [n_group]
    '''

    # dataloader.dataset.ng_sample(1)

    model.eval()

    full_items = [i for i in range(n_items)]

    HR = []
    NDCG = []

    for user, item, rating in dataloader:
        all_users = user.unique()
        all_users = all_users.to(device)
        user = user.to(device)
        item = item.to(device)

        # get ratings between every user and item - shape is num users x num movies
        if loss_fn == 'gcn_bpr_loss':
            user_embedding = model.users_emb.weight
            item_embedding = model.items_emb.weight
            gcn_rating = torch.matmul(user_embedding, item_embedding.T)

        for uid in all_users:
            user_id = uid.item()
            user_indices = torch.where(user == uid)
            gt_items = item[user_indices].cpu().numpy().tolist()

            neg_items = list(set(full_items) - set(pos_dict[user_id]))

            if loss_fn == 'gcn_bpr_loss':
                # user_mapping = torch.tensor(user_mapping)
                # pos_mapping = torch.tensor(pos_mapping)
                # print(user_mapping)
                # print(all_users)
                new_user = torch.tensor([user_mapping[user_id]] * len(neg_items), dtype=torch.long).to(device)
                nnnnn = []
                for i in neg_items:
                    nnnnn.append(pos_mapping[i])
                new_item = torch.tensor(nnnnn, dtype=torch.long).to(device)
                predictions = gcn_rating[new_user, new_item]
            else:
                new_user = torch.tensor([user_id] * len(neg_items), dtype=torch.long).to(device)
                new_item = torch.tensor(neg_items, dtype=torch.long).to(device)
                predictions = model(new_user, new_item)
            _, indices = torch.topk(predictions, top_k)
            recommends = torch.take(new_item, indices).cpu().numpy().tolist()

            HR.append(hit(gt_items, recommends))
            NDCG.append(ndcg(gt_items, recommends))

    return np.mean(NDCG), np.mean(HR)


# shrink and perturb
def spTrick(model, shrink=0.5, sigma=0.01):
    for (name, param) in model.named_parameters():
        if 'weight' in name:
            for i in range(param.shape[0]):
                for j in range(param.shape[1]):
                    param.data[i][j] = shrink * param.data[i][j] + torch.normal(0.0, sigma, size=(1, 1))
    return model


#####################
# mat visualization
##################### 

def embeddingUI(data, userset, featureK):
    # data = [user_mat, item_mat]
    pca = PCA(n_components=2)
    for i in range(2):
        if i == 0:
            user_mat = data[0].copy()
            for j in range(data[0].shape[0]):
                if j not in userset:
                    user_mat[j] = np.zeros(featureK)
        title = 'USER' if i == 0 else 'ITEM'
        pca.fit(data[i])
        feature2d = pca.transform(data[i])
        ax = plt.subplot(int('12' + str(i + 1)))
        plt.scatter(feature2d[:, 0], feature2d[:, 1])
        ax.set_title(title)


def embeddingItem(data, id=[53, 54, 58, 59]):
    # data = item_mat
    pca = PCA(n_components=2)
    res = np.empty((len(id), data.shape[1]))
    for i, idx in enumerate(id):
        res[i] = data[idx - 1]
    pca.fit(res)
    feature2d = pca.transform(res)
    plt.figure()
    plt.scatter(feature2d[:, 0], feature2d[:, 1])
    for i in range(len(id)):
        plt.annotate(id[i], xy=(feature2d[i, 0], feature2d[i, 1]))


##################### 
# object saving
##################### 

def saveObject(filename, obj):
    with open(filename + '.pkl', 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def loadObject(filename):
    with open(filename + '.pkl', 'rb') as input:
        obj = pickle.load(input)
    return obj


def kmeans(X, k, balance=False, max_iters=10):
    # Initialize centroids randomly
    n, _ = X.shape
    group_len = int(np.ceil(n / k))
    centroid = X[np.random.choice(n, size=k, replace=False)]

    dist = ((X - centroid[:, np.newaxis]) ** 2).sum(axis=2)  # [k, n]

    inertia = np.min(dist, axis=0).sum()

    # Iterate until convergence or maximum iterations
    for _ in range(max_iters):
        # print(_)
        # Assign each sample to the nearest centroid
        dist = ((X - centroid[:, np.newaxis]) ** 2).sum(axis=2)  # [k, n]
        if balance:
            label_count = [group_len] * k
            assinged_sample = []
            assigned_dict = set()

            label = np.zeros(n)
            inertia = 0

            flat_idx_sorted = np.argsort(dist.ravel())[::-1]

            # print(np.argsort(dist.ravel())[::-1].shape)

            row_idx, col_idx = np.unravel_index(flat_idx_sorted, dist.shape)

            for val, cen_idx, sample_idx in zip(dist[row_idx, col_idx], row_idx, col_idx):

                if len(assigned_dict) == n:
                    break
                # if sample_idx in assinged_sample:
                if sample_idx in assigned_dict:
                    continue
                if label_count[cen_idx] > 0:
                    label[sample_idx] = cen_idx
                    # assinged_sample.append(sample_idx)
                    assigned_dict.add(sample_idx)
                    label_count[cen_idx] -= 1
                    inertia += val
        else:
            label = np.argmin(dist, axis=0)
            inertia = np.min(dist, axis=0).sum()

        # Update centroids to the mean of assigned samples
        new_centroid = np.array([X[label == i].mean(axis=0) for i in range(k)])

        # Check if centroids have converged
        if np.allclose(centroid, new_centroid):
            break

        centroid = new_centroid

    print(f'{inertia:.3f}', end=' ')
    return inertia, label  # , centroid


def ot_cluster(X, k, max_iters=10):
    # Initialize centroids randomly
    n, _ = X.shape
    centroid = X[np.random.choice(n, size=k, replace=False)]

    # Iterate until convergence or maximum iterations

    for _ in range(max_iters):
        # compute distance
        dist = ((X - centroid[:, np.newaxis]) ** 2).sum(axis=2)  # [k, n]

        # print(dist.shape)
        inertia = np.min(dist, axis=0).sum()

        # compute sinkhorn distance
        lam = 1e-3
        a = np.ones(n) / n
        b = np.ones(k) / k
        # b = np.array([0.1, 0.2, 0.3, 0.4])

        trans = ot.emd(a, b, dist.T, lam)

        # Update centroids to the mean of assigned samples
        label = np.argmax(trans, axis=1)
        new_centroid = np.array([X[label == i].mean(axis=0) for i in range(k)])

        # Check if centroids have converged
        if np.allclose(centroid, new_centroid):
            break

        centroid = new_centroid
    return inertia, label  # , centroids
