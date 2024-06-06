import numpy as np
import pandas as pd


def map_user_id(user_id):
    global new_user_id
    if user_id not in user_id_mapping:
        user_id_mapping[user_id] = new_user_id
        new_user_id += 1
    return user_id_mapping[user_id]


def map_item_id(item_id):
    global new_item_id
    if item_id not in item_id_mapping:
        item_id_mapping[item_id] = new_item_id
        new_item_id += 1
    return item_id_mapping[item_id]


if __name__ == '__main__':
    # # 读取 CSV 文件
    # file_path = 'origin_ratings.csv'  # 请更改为您上传的 CSV 文件路径
    # df = pd.read_csv(file_path, header=None, names=['user_id', 'item_id', 'rating','time'])
    # df = df.drop(columns=['time'])
    #
    # # 统计每个用户交互的物品数量
    # user_item_counts = df['user_id'].value_counts()
    #
    # # 过滤掉交互物品数量小于5的用户
    # users_to_keep = user_item_counts[user_item_counts >= 5].index
    # df = df[df['user_id'].isin(users_to_keep)].reset_index(drop=True)
    #
    # # 创建用户和物品的映射字典
    # user_mapping = {user_id: idx for idx, user_id in enumerate(df['user_id'].unique())}
    # item_mapping = {item_id: idx for idx, item_id in enumerate(df['item_id'].unique())}
    #
    # # 应用映射
    # df['user_id'] = df['user_id'].map(user_mapping)
    # df['item_id'] = df['item_id'].map(item_mapping)
    # df['rating'] = 1  # 设置评分为 1
    #
    # # 保存新的 CSV 文件
    # output_file_path = 'x_ratings.csv'
    # df.to_csv(output_file_path, index=False, header=False)
    # print(f"save to: {output_file_path}")

    # # 读取 CSV 文件
    # file_path = 'x_ratings.csv'  # 请更改为您上传的 CSV 文件路径
    # df = pd.read_csv(file_path, header=None, names=['uid', 'iid', 'val'])
    #
    # # 分割训练集和测试集，按照每个用户5分之一的交互物品作为测试集
    # def split_train_test(group):
    #     group = group.sample(frac=1).reset_index(drop=True)
    #     test_size = max(1, int(len(group) * 0.2))
    #     test_set = group.iloc[:test_size]
    #     train_set = group.iloc[test_size:]
    #     return train_set, test_set
    #
    #
    # train_list = []
    # test_list = []
    #
    # for _, group in df.groupby('uid'):
    #     train, test = split_train_test(group)
    #     train_list.append(train)
    #     test_list.append(test)
    #
    # train_df = pd.concat(train_list).reset_index(drop=True)
    # test_df = pd.concat(test_list).reset_index(drop=True)
    #
    # # 保存为新的 CSV 文件
    # train_file_path = 'train.csv'
    # test_file_path = 'test.csv'
    # x_file_path = 'ratings.csv'
    #
    # df.to_csv(x_file_path, index=False)
    # train_df.to_csv(train_file_path, index=False)
    # test_df.to_csv(test_file_path, index=False)
    #
    # print(f"ratings.csv 文件已保存到: {x_file_path}")
    # print(f"train.csv 文件已保存到: {train_file_path}")
    # print(f"test.csv 文件已保存到: {test_file_path}")

    # 读取训练集 CSV 文件
    train_file_path = 'train.csv'  # 请更改为您保存的 train.csv 文件路径
    train_df = pd.read_csv(train_file_path)

    # 创建用户与交互物品的字典
    pos_dict = train_df.groupby('uid')['iid'].apply(list).to_dict()

    # 保存字典为 .npy 文件
    pos_dict_file_path = 'pos_dict.npy'
    np.save(pos_dict_file_path, pos_dict)

    print(f"文件已保存到: {pos_dict_file_path}")
    x = np.load('pos_dict.npy', allow_pickle=True).item()
    print(len(x))
    y = []
    for i in x.values():
        y += i
    print(max(y))
