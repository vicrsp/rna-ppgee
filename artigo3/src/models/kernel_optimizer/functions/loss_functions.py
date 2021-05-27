import numpy as np
import torch
import torch.nn.functional as F

from scipy.spatial import distance_matrix


def supervised_rbf_loss(X, y, sigma):

    if sigma == 0.:
        return 0.

    dist_matrix = distance_matrix(X, X)
    gram_matrix = np.exp(-.5 * ((dist_matrix**2)/(sigma**2)))

    classes = sorted(np.unique(y))
    N = gram_matrix.shape[0]

    gram_matrix = gram_matrix * (1 - np.eye(N))

    similarities = [
        np.sum(gram_matrix[
            :, np.where(y == c)[0]
        ], axis=1)/(N-1) for c in classes
    ]

    similarities = np.array(similarities).transpose()
    mean_sims = np.array([
        similarities[
            np.where(y == c)[0]
        ].mean(axis=0) for c in classes
    ])

    return -1. * np.triu(
        distance_matrix(mean_sims, mean_sims)
    ).sum()


def supervisedDotProductLoss_torch(X, y, normalize=False, negative=True):

    if normalize:
        X = F.normalize(X, dim=1)

    gram_mat = torch.mm(X, X.t())
    N = gram_mat.shape[0]
    gram_mat = gram_mat * (1 - torch.eye(N))

    classes = torch.unique(y)
    nclasses = len(classes)
    loss = 0
    for i in range(nclasses):
        for j in range(i+1, nclasses):
            c0 = classes[i]
            c1 = classes[j]
            which_c0 = np.where(y == c0)[0]
            which_c1 = np.where(y == c1)[0]
            if len(which_c0) == 0 or len(which_c1) == 0:
                continue

            sim_to_c0 = torch.sum(gram_mat[:, which_c0], dim=1)/(N-1)
            sim_to_c1 = torch.sum(gram_mat[:, which_c1], dim=1)/(N-1)

            sim_c0_c0 = torch.mean(sim_to_c0[which_c0])
            sim_c0_c1 = torch.mean(sim_to_c1[which_c0])
            sim_c1_c0 = torch.mean(sim_to_c0[which_c1])
            sim_c1_c1 = torch.mean(sim_to_c1[which_c1])

            loss += torch.log(torch.sqrt(
                (sim_c0_c0 - sim_c1_c0) ** 2 + (sim_c1_c1 - sim_c0_c1) ** 2
            ))

    if negative:
        loss = -1 * loss
    return loss
    #print(sim_c0_c0, sim_c1_c1, sim_c1_c0, sim_c0_c1)
    #loss = (sim_c0_c0 - 1.)**2 + (sim_c1_c1 - 1.)**2 + (sim_c1_c0)**2 + (sim_c0_c1)**2
    #return loss
