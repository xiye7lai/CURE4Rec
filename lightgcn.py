import argparse
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn, optim, Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import structured_negative_sampling
from torch_sparse import SparseTensor, matmul

from utils import kmeans, ot_cluster


def load_node_csv(path, index_col):
    """
    Args:
        path (str): 数据集路径
        index_col (str): 数据集文件里的列索引
    Returns:
        dict: 列号和用户ID的索引、列好和电影ID的索引
    """
    df = pd.read_csv(path, index_col=index_col)
    mapping = {index: i for i, index in enumerate(df.index.unique())}  # enumerate()索引函数,默认索引从0开始
    return mapping


def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping, link_index_col, rating_threshold=4):
    """
    Args:
        path (str): 数据集路径
        src_index_col (str): 用户列名
        src_mapping (dict): 行号和用户ID的映射
        dst_index_col (str): 电影列名
        dst_mapping (dict): 行号和电影ID的映射
        link_index_col (str): 交互的列名
        rating_threshold (int, optional): 决定选取多少评分交互的阈值，设置为4分
    Returns:
        torch.Tensor: 2*N的用户电影交互节点图
    """
    df = pd.read_csv(path)
    edge_index = None
    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_attr = torch.from_numpy(df[link_index_col].values).view(-1, 1).to(
        torch.long) >= rating_threshold  # 将数组转化为tensor张量
    edge_index = [[], []]
    for i in range(edge_attr.shape[0]):
        if edge_attr[i]:
            edge_index[0].append(src[i])
            edge_index[1].append(dst[i])

    return torch.tensor(edge_index)


def sample_mini_batch(batch_size, edge_index):
    """
    Args:
        batch_size (int): 批大小
        edge_index (torch.Tensor): 2*N的边列表
    Returns:
        tuple: user indices, positive item indices, negative item indices
    """
    edges = structured_negative_sampling(edge_index)
    edges = torch.stack(edges, dim=0)
    indices = random.choices(
        [i for i in range(edges[0].shape[0])], k=batch_size)
    batch = edges[:, indices]
    user_indices, pos_item_indices, neg_item_indices = batch[0], batch[1], batch[2]
    return user_indices, pos_item_indices, neg_item_indices


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


def bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0,
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


def get_user_positive_items(edge_index):
    """为每个用户生成正采样字典
    Args:
        edge_index (torch.Tensor): 2*N的边列表
    Returns:
        dict: 每个用户的正采样字典
    """
    user_pos_items = {}
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        if user not in user_pos_items:
            user_pos_items[user] = []
        user_pos_items[user].append(item)
    return user_pos_items


def NDCGatK_r(groundTruth, r, k):
    """Computes Normalized Discounted Cumulative Gain (NDCG) @ k
    Args:
        groundTruth (list): 同上一个函数
        r (list): 同上一个函数
        k (int): 同上一个函数
    Returns:
        float: ndcg @ k
    """
    assert len(r) == len(groundTruth)

    test_matrix = torch.zeros((len(r), k))

    for i, items in enumerate(groundTruth):
        length = min(len(items), k)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = torch.sum(max_r * 1. / torch.log2(torch.arange(2, k + 2)), axis=1)
    dcg = r * (1. / torch.log2(torch.arange(2, k + 2)))
    dcg = torch.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0.
    return torch.mean(ndcg).item()


def hit(groundTruth, r, k):
    """Computes Hit Rate @ k"""
    assert len(r) == len(groundTruth)

    hits = []
    for i, gt_items in enumerate(groundTruth):
        hits.append(any(item in gt_items for item in r[i][:k]))
    return torch.mean(torch.tensor(hits).float()).item()


def RecallPrecision_ATk(groundTruth, r, k):
    """
    Args:
        groundTruth (list): 每个用户对应电影列表的高评分项
        r (list): 是否向每个用户推荐了前k个电影的列表
        k (intg): 确定要计算精度和召回率的前k个电影
    Returns:
        tuple: recall @ k, precision @ k
    """
    num_correct_pred = torch.sum(r, dim=-1)  # number of correctly predicted items per user
    # number of items liked by each user in the test set
    user_num_liked = torch.Tensor([len(groundTruth[i])
                                   for i in range(len(groundTruth))])
    recall = torch.mean(num_correct_pred / user_num_liked)
    precision = torch.mean(num_correct_pred) / k
    return recall.item(), precision.item()


def get_metrics(model, edge_index, exclude_edge_indices, k):
    """
    Args:
        model (LighGCN): lightgcn model
        edge_index (torch.Tensor): 2*N列表
        exclude_edge_indices ([type]): 2*N列表
        k (int): 前多少个电影
    Returns:
        tuple: recall @ k, precision @ k, ndcg @ k
    """
    user_embedding = model.users_emb.weight
    item_embedding = model.items_emb.weight

    # get ratings between every user and item - shape is num users x num movies
    rating = torch.matmul(user_embedding, item_embedding.T)

    for exclude_edge_index in exclude_edge_indices:
        # gets all the positive items for each user from the edge index
        user_pos_items = get_user_positive_items(exclude_edge_index)
        # get coordinates of all edges to exclude
        exclude_users = []
        exclude_items = []
        for user, items in user_pos_items.items():
            exclude_users.extend([user] * len(items))
            exclude_items.extend(items)

        # set ratings of excluded edges to large negative value
        rating[exclude_users, exclude_items] = -(1 << 10)

    # get the top k recommended items for each user
    _, top_K_items = torch.topk(rating, k=k)

    # get all unique users in evaluated split
    users = edge_index[0].unique()

    test_user_pos_items = get_user_positive_items(edge_index)

    # convert test user pos items dictionary into a list
    test_user_pos_items_list = [
        test_user_pos_items[user.item()] for user in users]

    # determine the correctness of topk predictions
    r = []
    for user in users:
        ground_truth_items = test_user_pos_items[user.item()]
        label = list(map(lambda x: x in ground_truth_items, top_K_items[user]))
        r.append(label)
    r = torch.Tensor(np.array(r).astype('float'))

    # recall, precision = RecallPrecision_ATk(test_user_pos_items_list, r, k)
    ndcg = NDCGatK_r(test_user_pos_items_list, r, k)
    hr = hit(test_user_pos_items_list, r, k)

    return ndcg, hr


def evaluation(model, edge_index, sparse_edge_index, exclude_edge_indices, k, lambda_val):
    # get embeddings
    users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(
        sparse_edge_index)
    edges = structured_negative_sampling(
        edge_index, contains_neg_self_loops=False)
    user_indices, pos_item_indices, neg_item_indices = edges[0], edges[1], edges[2]
    users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
    pos_items_emb_final, pos_items_emb_0 = items_emb_final[
                                               pos_item_indices], items_emb_0[pos_item_indices]
    neg_items_emb_final, neg_items_emb_0 = items_emb_final[
                                               neg_item_indices], items_emb_0[neg_item_indices]

    loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0,
                    neg_items_emb_final, neg_items_emb_0, lambda_val).item()

    ndcg, hr = get_metrics(
        model, edge_index, exclude_edge_indices, k)

    return loss, ndcg, hr


def remove_users(edge_index_train, edge_index_test, src_mapping, percentage, type):
    user_interactions = {user: 0 for user in src_mapping.values()}
    for user in edge_index_train[0].tolist() + edge_index_test[0].tolist():
        user_interactions[user] += 1

    if type == 'core':
        sorted_users = sorted(user_interactions, key=user_interactions.get, reverse=True)
    elif type == 'edge':
        sorted_users = sorted(user_interactions, key=user_interactions.get)
    elif type == 'random':
        sorted_users = list(user_interactions.keys())
        random.shuffle(sorted_users)

    num_remove = int(len(sorted_users) * percentage)
    remove_users = set(sorted_users[:num_remove])

    def filter_edges(edge_index):
        new_edge_index = [[], []]
        for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            if src not in remove_users:
                new_edge_index[0].append(src)
                new_edge_index[1].append(dst)
        return torch.tensor(new_edge_index)

    return filter_edges(edge_index_train), filter_edges(edge_index_test)


def split_users(edge_index_train, edge_index_test, src_mapping, num_groups, type, dataset):
    users = list(src_mapping.values())
    if type == 'sisa':
        random.shuffle(users)
        group_size = len(users) // num_groups
        user_groups = [users[i * group_size:(i + 1) * group_size] for i in range(num_groups)]
    elif type == 'receraser':
        # 加载用户嵌入
        user_embeddings = np.load(f'results/user_emb/{dataset}_mf_emb.npy', allow_pickle=True).item()
        # 提取用户嵌入矩阵
        user_mat = np.array([user_embeddings[user_id][0] for user_id in users])

        _, labels = kmeans(user_mat, num_groups, True, 30)

        # 创建用户分组
        user_groups = [[] for _ in range(num_groups)]
        for user_id, label in zip(users, labels):
            user_groups[int(label)].append(user_id)
    elif type == 'ultrare':
        # 加载用户嵌入
        user_embeddings = np.load(f'results/user_emb/{dataset}_mf_emb.npy', allow_pickle=True).item()
        # 提取用户嵌入矩阵
        user_mat = np.array([user_embeddings[user_id][0] for user_id in users])

        _, labels = ot_cluster(user_mat, num_groups)

        # 创建用户分组
        user_groups = [[] for _ in range(num_groups)]
        for user_id, label in zip(users, labels):
            user_groups[int(label)].append(user_id)

    def filter_edges_by_group(edge_index, group_set):
        new_edge_index = [[], []]
        for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            if src in group_set:
                new_edge_index[0].append(src)
                new_edge_index[1].append(dst)
        return torch.tensor(new_edge_index)

    edge_groups_train = [filter_edges_by_group(edge_index_train, set(group)) for group in user_groups]
    edge_groups_test = [filter_edges_by_group(edge_index_test, set(group)) for group in user_groups]

    return edge_groups_train, edge_groups_test


def classify_users(edge_index, src_mapping):
    user_interactions = {user: 0 for user in src_mapping.values()}
    for user in edge_index[0].tolist():
        user_interactions[user] += 1

    sorted_users = sorted(user_interactions, key=user_interactions.get, reverse=True)
    num_active = int(len(sorted_users) * 0.05)
    active_users = set(sorted_users[:num_active])
    inactive_users = set(sorted_users[num_active:])

    return active_users, inactive_users


def generate_active_inactive_datasets(edge_index_groups, active_users, inactive_users):
    def filter_edges_by_activity(edge_index, user_set):
        new_edge_index = [[], []]
        for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            if src in user_set:
                new_edge_index[0].append(src)
                new_edge_index[1].append(dst)
        return torch.tensor(new_edge_index)

    active_datasets = [filter_edges_by_activity(edge_index, active_users) for edge_index in edge_index_groups]
    inactive_datasets = [filter_edges_by_activity(edge_index, inactive_users) for edge_index in edge_index_groups]

    active_counts = [len(set(edge_index[0].tolist()) & active_users) for edge_index in edge_index_groups]
    inactive_counts = [len(set(edge_index[0].tolist()) & inactive_users) for edge_index in edge_index_groups]

    return active_datasets, inactive_datasets, active_counts, inactive_counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m', help='dataset name')
    parser.add_argument('--learn', type=str, default='sisa', help='type of learning and unlearning')
    parser.add_argument('--deltype', type=str, default='random', help='unlearn data selection')

    args = parser.parse_args()

    assert args.dataset in ['ml-100k', 'ml-1m', 'adm', 'gowalla']
    dataset = args.dataset

    del_type = args.deltype
    method = args.learn
    groups = 1
    del_per = 0.05

    # dataset = 'ml-100k'
    #
    # del_type = 'core'
    # method = 'sisa'

    rating_path = f'data/{dataset}/ratings.csv'
    train_path = f'data/{dataset}/train.csv'
    test_path = f'data/{dataset}/test.csv'

    user_mapping = load_node_csv(rating_path, index_col='uid')
    movie_mapping = load_node_csv(rating_path, index_col='iid')

    #
    # edge_index = load_edge_csv(
    #     rating_path,
    #     src_index_col='uid',
    #     src_mapping=user_mapping,
    #     dst_index_col='iid',
    #     dst_mapping=movie_mapping,
    #     link_index_col='val',
    #     rating_threshold=1,
    # )
    train_edge_index = load_edge_csv(
        train_path,
        src_index_col='uid',
        src_mapping=user_mapping,
        dst_index_col='iid',
        dst_mapping=movie_mapping,
        link_index_col='val',
        rating_threshold=1,
    )
    test_edge_index = load_edge_csv(
        test_path,
        src_index_col='uid',
        src_mapping=user_mapping,
        dst_index_col='iid',
        dst_mapping=movie_mapping,
        link_index_col='val',
        rating_threshold=1,
    )

    num_users, num_movies = len(user_mapping), len(movie_mapping)
    # num_interactions = edge_index.shape[1]
    # all_indices = [i for i in range(num_interactions)]  # 所有索引
    #
    # train_indices, test_indices = train_test_split(
    #     all_indices, test_size=0.2, random_state=1)  # 将数据集划分成80:10的训练集:测试集
    #
    # train_edge_index = edge_index[:, train_indices]
    # test_edge_index = edge_index[:, test_indices]

    # Remove 5% of users
    train_edge_index, test_edge_index = remove_users(train_edge_index, test_edge_index, user_mapping, del_per,
                                                     del_type)

    # Split remaining users into 10 groups
    train_edge_groups, test_edge_groups = split_users(train_edge_index, test_edge_index, user_mapping, groups,
                                                      type=method, dataset=dataset)

    # Classify users into active and inactive
    active_users, inactive_users = classify_users(train_edge_index, user_mapping)

    # Generate active and inactive datasets
    # active_train_datasets, inactive_train_datasets = generate_active_inactive_datasets(train_edge_groups, active_users,
    #                                                                                    inactive_users)
    active_test_datasets, inactive_test_datasets, active_nums, inactive_nums = generate_active_inactive_datasets(
        test_edge_groups, active_users,
        inactive_users)

    ndcgs = [0] * groups
    hrs = [0] * groups
    active_ndcgs = [0] * groups
    inactive_ndcgs = [0] * groups
    times = [0] * groups

    for i in range(groups):
        ITERATIONS = 10000
        BATCH_SIZE = 1024
        LR = 1e-3
        ITERS_PER_EVAL = 200
        ITERS_PER_LR_DECAY = 200
        K = 20
        LAMBDA = 1e-6
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

        best_ndcg = 0
        best_hr = 0
        count_dec = 0

        model = LightGCN(num_users, num_movies)
        model = model.to(device)
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        # edge_index = edge_index.to(device)
        train_edge_index = train_edge_groups[i].to(device)
        train_sparse_edge_index = train_edge_groups[i].to(device)

        test_edge_index = test_edge_groups[i].to(device)
        test_sparse_edge_index = test_edge_groups[i].to(device)

        active_edge_index = active_test_datasets[i].to(device)
        active_sparse_edge_index = active_test_datasets[i].to(device)

        inactive_edge_index = inactive_test_datasets[i].to(device)
        inactive_sparse_edge_index = inactive_test_datasets[i].to(device)

        train_losses = []
        val_losses = []

        start_time = time.time()
        total_time = 0

        for iter in range(ITERATIONS):
            # forward propagation
            users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(
                train_sparse_edge_index)

            # mini batching
            user_indices, pos_item_indices, neg_item_indices = sample_mini_batch(
                BATCH_SIZE, train_edge_index)
            user_indices, pos_item_indices, neg_item_indices = user_indices.to(
                device), pos_item_indices.to(device), neg_item_indices.to(device)
            users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
            pos_items_emb_final, pos_items_emb_0 = items_emb_final[
                                                       pos_item_indices], items_emb_0[pos_item_indices]
            neg_items_emb_final, neg_items_emb_0 = items_emb_final[
                                                       neg_item_indices], items_emb_0[neg_item_indices]

            # loss computation
            train_loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final,
                                  pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, LAMBDA)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()


            if iter % ITERS_PER_EVAL == 0:
                model.eval()
                val_loss, ndcg, hr = evaluation(
                    model, test_edge_index, test_sparse_edge_index, [train_edge_index], K, LAMBDA)
                print(
                    f"[Iteration {iter}/{ITERATIONS}] train_loss: {round(train_loss.item(), 5)}, val_loss: {round(val_loss, 5)}, ndcg@{K}: {round(ndcg, 5)}, hr@{K}: {round(hr, 5)}")
                train_losses.append(train_loss.item())
                val_losses.append(val_loss)
                model.train()
                if ndcg > best_ndcg:
                    count_dec = 0
                    best_ndcg = ndcg
                    best_hr = hr
                    # if len(save_dir) > 0:
                    #     torch.save(model.state_dict(), save_dir + '/model' + '.pth')
                    #     torch.save(model.user_mat.weight.detach().cpu().numpy(), save_dir + '/user_mat' + '.npy')
                else:
                    count_dec += 1

                if count_dec > 5:
                    break

            if iter % ITERS_PER_LR_DECAY == 0 and iter != 0:
                scheduler.step()

        t_time = time.time() - start_time
        ndcgs[i] = best_ndcg
        hrs[i] = best_hr
        times[i] = t_time
        model.eval()
        if active_nums[i] > 0:
            val_loss, a_ndcg, a_hr = evaluation(
                model, active_edge_index, active_sparse_edge_index, [train_edge_index], K, LAMBDA)
        val_loss, i_ndcg, i_hr = evaluation(
            model, inactive_edge_index, inactive_sparse_edge_index, [train_edge_index], K, LAMBDA)
        print(f'Group {i}/{groups} Finish!')
        print("best!!!!!!!!!!!!")
        print(
            f"[best_ndcg@{K}: {round(best_ndcg, 5)}")
        if active_nums[i] > 0:
            print(
                f"[active_ndcg@{K}: {round(a_ndcg, 5)}")
        print(
            f"[inactive_ndcg@{K}: {round(i_ndcg, 5)}")
        if active_nums[i] > 0:
            active_ndcgs[i] = a_ndcg
        inactive_ndcgs[i] = i_ndcg
    print(np.mean(ndcgs))
    print(np.mean(hrs))
    active_ndcg = 0
    inactive_ndcg = 0
    for i in range(groups):
        active_ndcg += (active_nums[i] * active_ndcgs[i])
        inactive_ndcg += (inactive_nums[i] * inactive_ndcgs[i])

    active_num = np.sum(active_nums)
    active_ndcg = active_ndcg / active_num
    inactive_num = np.sum(inactive_nums)
    inactive_ndcg = inactive_ndcg / inactive_num
    fairness = active_ndcg - inactive_ndcg
    print(fairness)
    print(np.var(ndcgs))
    print(np.mean(times))
