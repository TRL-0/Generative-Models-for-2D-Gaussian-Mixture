import numpy as np
import random
import os
from utils import set_seed,visualize_data


def mean_generate():
    x = np.arange(-25, 26, 10)
    # y = np.arange(20, 81, 10)
    y= np.arange(-25, 26, 10)
    
    center = [np.array((i, j)) for i in x for j in y]

    assert len(center) >= 36  # there are at least 36 components

    return center

def data_generate(num=15000):
    means = mean_generate()
    for type in ['train','val','test']:
        data = []
        for i in range(num):
            mean = random.choice(means)
            data.append(np.random.normal(mean, 1))
        
        # visualization
        visualize_data(data[:1800],f'./data/{type}.png')
        
        # save data
        data = np.array(data)
        assert data.shape == (num, 2)
        os.makedirs('./data', exist_ok=True)
        np.save(f'./data/{type}.npy', data)
    


set_seed(0)
data_generate()




