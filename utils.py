import random
import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def visualize_data(visual_data,save_path):
    plt.figure(figsize=(6,6))
    x_c=[i[0] for i in visual_data]
    y_c=[i[1] for i in visual_data]
    plt.scatter(x_c,y_c,s=5)
    plt.savefig(save_path)
    plt.close()
    
def normalize_data(data):
    min_value = torch.min(data)
    max_value = torch.max(data)

    # 将数据归一化到[0,1]
    normalized_data = (data - min_value) / (max_value - min_value)
    return normalized_data, min_value, max_value