import torch
import numpy as np
import dill as pickle


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = torch.exp(x - x.max())
    return e_x / e_x.sum(dim=0)


## Data loader
def load_data(fPath):
    with open(fPath, 'rb') as fid:
        data = pickle.load(fid)
    return data


## Compute actual \Delta vector at each time step
def computeDelta(Seq, Param):
    n = len(Seq[0])
    T = len(Seq)
    K = Param['K']
    PtoG = Param['player-group']
    GtoP = Param['group-player']
    
    ret = []
    for t in range(T-1):
        X_t = Seq[t]
        DeltaVec = np.zeros(n)
        for i in range(n):
            selfGroupIdx = PtoG[i]
            delta_self = np.sum(X_t[GtoP[selfGroupIdx]])
            otherGroups = [g for g in range(K) if g != selfGroupIdx]

            Delta = delta_self
            for otherGroupIdx in otherGroups:
                delta_other = np.sum(X_t[GtoP[otherGroupIdx]])
                Delta -= delta_other / len(otherGroups)
            DeltaVec[i] = Delta
        ret.append(DeltaVec)
    return ret


## gMat: encode group info. of each player
def compute_gMat(Param):
    n = len(Param['G'])
    gMat = np.zeros((n, n))
    pTog = Param['player-group']
    gTop = Param['group-player']
    for i in range(n):
        gMat[i, gTop[pTog[i]]] = 1
    return gMat


## output results to disk
class logging(object):
    def __init__(self, P):
        self.path = P
    def output(self, S):
        with open(self.path, 'a') as fid:
            fid.write(S) 


