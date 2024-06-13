# 从文件中读取文本


def read_text_from_file(filename):
    with open(filename, 'r') as file:
        return file.read()


if __name__ == '__main__':
    # # pos = np.load('data/ml-100k/pos_dict.npy',allow_pickle=True)
    # # print(pos)
    # # 加载训练好的模型
    # model_path = 'results/gowalla/mf.pth'
    # model = MF(64115, 164532, 32)
    # model.load_state_dict(torch.load(model_path))
    # model.eval()  # 设置模型为评估模式
    #
    # # 加载数据集
    # data_path = 'data/gowalla/train.csv'
    # df = pd.read_csv(data_path)
    #
    # # 获取所有用户的唯一ID
    # unique_user_ids = df['uid'].unique()
    #
    # # 获取用户嵌入
    # user_embeddings = {}
    # for user_id in unique_user_ids:
    #     user_tensor = torch.tensor([user_id])  # 用户ID从1开始，所以减1
    #     with torch.no_grad():
    #         user_embedding = model.user_mat(user_tensor).numpy()
    #     user_embeddings[user_id] = user_embedding
    #
    #
    # embeddings_path = 'results/user_emb/gowalla_wmf_emb.npy'
    # np.save(embeddings_path, user_embeddings)
    import numpy as np
    import pandas as pd


    def delete(ratings, del_type, del_per):
        # Determine the users to be deleted based on del_type and del_per
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

        # Filter the ratings to get the data of the users to be deleted
        delete_data = ratings[ratings['uid'].isin(delete_users)]

        # Save the deleted users' data to CSV files
        if del_type == 'core':
            delete_data.to_csv(f'data/gowalla/core_data_{del_per}%.csv', index=False)
        elif del_type == 'edge':
            delete_data.to_csv(f'data/gowalla/edge_data_{del_per}%.csv', index=False)

        return delete_users


    # Read the ratings data from ratings.csv
    ratings = pd.read_csv('data/gowalla/ratings.csv')
    for i in [5,10,15,20]:
        # Delete 40% of core users and save to core.csv
        deleted_users_core = delete(ratings, 'core', i)

        # Delete 40% of edge users and save to edge.csv
        deleted_users_edge = delete(ratings, 'edge', i)




