import pickle

import numpy as np
import ot
import torch
from torch import nn

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


class WMF(nn.Module):
    def __init__(self, n_user, n_item, k=16):
        super(WMF, self).__init__()
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

    return train_loss / size


# torch test
def baseTest(dataloader, model, loss_fn, device, verbose, pos_dict, n_items, top_k=20, user_mapping=None,
             pos_mapping=None):
    model.eval()

    full_items = [i for i in range(n_items)]

    HR = []
    NDCG = []

    for user, item, rating in dataloader:
        all_users = user.unique()
        all_users = all_users.to(device)
        user = user.to(device)
        item = item.to(device)

        for uid in all_users:
            user_id = uid.item()
            user_indices = torch.where(user == uid)
            gt_items = item[user_indices].cpu().numpy().tolist()

            neg_items = list(set(full_items) - set(pos_dict[user_id]))

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


def aggregation(model_list, train_dlist, test_dlist, test_data, verbose, save_dir):
    '''
            train_dlist:   list of dataloader[n_group]
            '''
    self.model_list = model_list

    assert len(train_dlist) == self.n_group
    assert len(test_dlist) == self.n_group

    # find deletion
    retrain_gid = set()
    for user in del_user:
        for i in range(self.n_group):
            if user in self.group_index[i]:
                retrain_gid.add(i)
                break

    # sisa retraining
    model_before_unlearn = model_list[0]

    for i in retrain_gid:
        given_model = ''
        model = super(Sisa, self).train(train_dlist[i], test_dlist[i], test_data, verbose, save_dir, i + 1,
                                        given_model)
        self.model_list[i] = model

    # merge user mat
    if self.model_type == "nmf":
        weight_list1 = [m.user_mat_mf.weight.to(self.device) for m in self.model_list]
        weight_list2 = [m.user_mat_mlp.weight.to(self.device) for m in self.model_list]
        merged_weight1 = model_before_unlearn.user_mat_mf.weight.clone()
        merged_weight2 = model_before_unlearn.user_mat_mlp.weight.clone()

        for i in retrain_gid:
            merged_weight1[self.group_index[i]] = weight_list1[i][self.group_index[i]]
            merged_weight2[self.group_index[i]] = weight_list2[i][self.group_index[i]]

        for m in self.model_list:
            m.user_mat_mf.weight = nn.Parameter(merged_weight1)
            m.user_mat_mlp.weight = nn.Parameter(merged_weight2)

    else:
        weight_list = [m.user_mat.weight.to(self.device) for m in self.model_list]
        merged_weight = model_before_unlearn.user_mat.weight.clone()

        for i in retrain_gid:
            merged_weight[self.group_index[i]] = weight_list[i][self.group_index[i]]
        for m in self.model_list:
            m.user_mat.weight = nn.Parameter(merged_weight)

    # total test
    self.test(test_data, verbose, save_dir)

    return self.model_list
