import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.linalg import sqrtm


def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def visualize_data(visual_data, save_path):
    plt.figure(figsize=(6, 6))
    x_c = [i[0] for i in visual_data]
    y_c = [i[1] for i in visual_data]
    plt.scatter(x_c, y_c, s=5)
    plt.savefig(save_path)
    plt.close()


def compute_fid(p_samples, q_samples):
    """
    Calculate the Fréchet Inception Distance (FID) between two 2D distributions.
    """
    mu1, sigma1 = p_samples.mean(axis=0), np.cov(p_samples, rowvar=False)
    mu2, sigma2 = q_samples.mean(axis=0), np.cov(q_samples, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def compute_kl_divergence_2d(p_samples, q_samples,
                             xrange=[-30, 30], yrange=[-30, 30],
                             num_points=200):
    """
    Calculate the 2D KL divergence estimated by KDE.
    """
    p_samples, q_samples=p_samples.T, q_samples.T
    p_kde = gaussian_kde(p_samples)
    q_kde = gaussian_kde(q_samples)

    x = np.linspace(xrange[0], xrange[1], num_points)
    y = np.linspace(yrange[0], yrange[1], num_points)
    X, Y = np.meshgrid(x, y)
    xy = np.vstack([X.ravel(), Y.ravel()])

    # 计算这些点上的概率密度
    p_pdf = p_kde(xy).reshape(X.shape)
    q_pdf = q_kde(xy).reshape(X.shape)

    # 计算KL散度
    kl_divergence = np.sum(p_pdf * np.log(p_pdf / q_pdf)
                           ) * (x[1] - x[0]) * (y[1] - y[0])
    return kl_divergence
