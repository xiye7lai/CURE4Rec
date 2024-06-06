import re
from datetime import datetime, timedelta

# 从文件中读取文本
import numpy as np
import pandas as pd
import torch

from utils import MF


def read_text_from_file(filename):
    with open(filename, 'r') as file:
        return file.read()


if __name__ == '__main__':
    # pos = np.load('data/ml-100k/pos_dict.npy',allow_pickle=True)
    # print(pos)
    # 加载训练好的模型
    model_path = 'results/gowalla/mf.pth'
    model = MF(64115, 164532, 32)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置模型为评估模式

    # 加载数据集
    data_path = 'data/gowalla/train.csv'
    df = pd.read_csv(data_path)

    # 获取所有用户的唯一ID
    unique_user_ids = df['uid'].unique()

    # 获取用户嵌入
    user_embeddings = {}
    for user_id in unique_user_ids:
        user_tensor = torch.tensor([user_id])  # 用户ID从1开始，所以减1
        with torch.no_grad():
            user_embedding = model.user_mat(user_tensor).numpy()
        user_embeddings[user_id] = user_embedding


    embeddings_path = 'results/user_emb/gowalla_mf_emb.npy'
    np.save(embeddings_path, user_embeddings)
