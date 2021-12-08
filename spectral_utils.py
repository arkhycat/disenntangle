import torch
import numpy as np
from sklearn.cluster import KMeans


device = "cuda" if torch.cuda.is_available() else "cpu"

def ws_to_adj(ws):
    # input: list of weight matrices
    # output: adjacency matrix of the corresponding graph
    total_nodes = ws[0].shape[1] + sum([w.shape[0] for w in ws])
    adj = torch.zeros((total_nodes, total_nodes), device = device)
    x = ws[0].shape[1]
    y = 0
    for w in ws:
        #print("{}:{}, {}:{}".format(x, x+w.shape[1], y, y+w.shape[0]))
        adj[x:(x+w.shape[0]), y:(y+w.shape[1])] = torch.abs(w)
        adj[y:(y+w.shape[1]), x:(x+w.shape[0])] = torch.abs(w).transpose(-2, -1)
        x += w.shape[0]
        y += w.shape[1]
    return adj

def normalize_w(w):
    d1_inv_root = torch.diag(1/torch.sqrt(torch.abs(w).sum(1)))
    d2_inv_root = torch.diag(1/torch.sqrt(torch.abs(w).sum(0)))
    return torch.matmul(torch.matmul(d1_inv_root, torch.abs(w)), d2_inv_root)

def laplacian(adj, norm='sym'):
    w = torch.abs(adj)
    d = torch.diag(w.sum(-1))
    if norm is None:
        return d-w
    elif norm == 'sym':
        d_inv_root = torch.diag(1/torch.sqrt(w.sum(-1)))
        return torch.eye(w.shape[0], device=device) - torch.matmul(torch.matmul(d_inv_root, w), d_inv_root)
    else:
        raise Exception("Don't know the {} norm".format(norm))

def blocks_from_svd(u, ncc):
    u = u[:, :ncc].detach().cpu().numpy()

    kmeans = KMeans(n_clusters=ncc)
    kmeans.fit(u)
    y_kmeans = kmeans.predict(u)
    return y_kmeans

    # uset = []
    # blocks = []
    # for x in (u>0):
    #     if str(x) not in uset:
    #         uset.append(str(x))
    #     blocks.append(uset.index(str(x)))
    # return np.array(blocks)
