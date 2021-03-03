import torch
import numpy as np

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
        adj[x:(x+w.shape[0]), y:(y+w.shape[1])] = w
        adj[y:(y+w.shape[1]), x:(x+w.shape[0])] = w.transpose(-2, -1)
        x += w.shape[0]
        y += w.shape[1]
    return adj

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

def block_diag_perm(adj, evals, evecs):
    # input: adjacency matrix, list of eigen values (ascending order), list of the corresponding eigen eigenvectors
    # output: matrix close to a block diagonal. Number of blocks equals to the number of provided eigen values
    evals = evals.numpy()
    evecs = evecs.numpy()
