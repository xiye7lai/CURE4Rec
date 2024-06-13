import re

import numpy as np


def extract_numbers(file_path):
    active_nums = []
    inactive_nums = []

    with open(file_path, 'r') as file:
        for line in file:
            active_match = re.search(r' active users: (\d+)', line)
            inactive_match = re.search(r' inactive users: (\d+)', line)

            if active_match:
                active_nums.append(int(active_match.group(1)))

            if inactive_match:
                inactive_nums.append(int(inactive_match.group(1)))

    return active_nums, inactive_nums


def print_result(learn_type='ultraue'
                 , model_type='mf'
                 , dataset='ml-100k'
                 , del_type='edge'
                 , del_per=5
                 , groups=10):
    if learn_type == 'retrain':
        result = np.load(
            f'results/{learn_type}/{model_type}_{dataset}_{del_type}_{del_per}.npy', allow_pickle=True).item()
        ndcg = result['ndcg']
        hr = result['hr']
        active_ndcg = result['active_ndcg']
        inactive_ndcgs = result['inactive_ndcg']
        time = result['time']
        var = 0
        fairness = active_ndcg - inactive_ndcgs
    else:
        ndcgs = []
        hrs = []
        active_ndcgs = []
        inactive_ndcgs = []
        times = []
        active_num, inactive_num = extract_numbers(
            f'./log/{del_per}_{dataset}_{model_type}_{learn_type}_{del_type}_{groups}.txt')

        # # active_num, inactive_num = extract_numbers(
        # #     f'./log/{dataset}_{model_type}_{learn_type}_{del_type}.txt')
        #
        # # print(active_num)
        # # print(inactive_num)
        # for i in range(groups):
        #     result = np.load(
        #         f'results/{learn_type}/group{i + 1}_{groups}_{model_type}_{dataset}_{del_type}_{del_per}.npy',
        #         allow_pickle=True).item()
        #     # result = np.load(
        #     #     f'results/{learn_type}/group{i + 1}_{model_type}_{dataset}_{del_type}.npy',
        #     #     allow_pickle=True).item()
        #     ndcgs.append(result['ndcg'])
        #     hrs.append(result['hr'])
        #     active_ndcgs.append(result['active_ndcg'])
        #     inactive_ndcgs.append(result['inactive_ndcg'])
        #     times.append(result['time'])
        #
        # ndcg = np.mean(ndcgs)
        # hr = np.mean(hrs)
        # var = np.var(ndcgs)
        # time = np.mean(times)
        # active_ndcg = 0
        # inactive_ndcg = 0
        # for i in range(groups):
        #     active_ndcg += (active_num[i] * active_ndcgs[i])
        #     inactive_ndcg += (inactive_num[i] * inactive_ndcgs[i])
        #
        # active_total_num = np.sum(active_num)
        # active_ndcg = active_ndcg / active_total_num
        # inactive_total_num = np.sum(inactive_num)
        # inactive_ndcg = inactive_ndcg / inactive_total_num
        # fairness = active_ndcg - inactive_ndcg

    # print('NDCG:', ndcg)
    # print('HR:', hr)
    # print('var:', var)
    # print('fairness:', fairness)
    # print('time:', time)

    # print(ndcgs)

    # print(ndcg)
    # print(hr)
    # print(fairness)
    # print(var)
    # print(time)
    for i in range(5):
        print(active_num[i])
        print(inactive_num[i])
    print(" --------------")
    for i in range(5,10):
        print(active_num[i])
        print(inactive_num[i])


if __name__ == '__main__':
    learn_type = 'ultraue'
    model_type = 'bpr'
    dataset = 'adm'
    del_type = 'core'
    del_per = 5
    groups = 10

    for i in ['core','random','edge']:
        print_result(learn_type=learn_type,
                     model_type=model_type,
                     dataset=dataset,
                     del_type=i,
                     del_per=del_per,
                     groups=groups)
        print('dfsdfgdfsfsdfds')
