import torch
import argparse
import numpy as np
import networkx as nx
import scipy.stats as scis
from Models import BNPGModel, DeltaDiffAggre
from utils import load_data, compute_gMat


parser = argparse.ArgumentParser()
parser.add_argument('--timeStep', type=int, default=10000, help='time steps')
parser.add_argument('--gamma', type=float, default=5, help='gamma')
parser.add_argument('--numAgent', type=int, default=100, help='#(agent)')
parser.add_argument('--graph_type', type=str, default='BA', help='graph type')
parser.add_argument('--numTest', type=int, default=0, help='whether to save the model')
parser.add_argument('--numTrain', type=int, default=500, help='whether to save the model')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--verbose', type=int, default=1)
parser.add_argument('--fPath', type=str, default="")
args = parser.parse_args()

SEED = args.seed
np.random.seed(SEED)
torch.manual_seed(SEED)
DTYPE = torch.float32
#device = torch.device("cuda:0" if args.cuda else "cpu")
device='cpu'

n = args.numAgent
numTrain = args.numTrain
numTest  = args.numTest


###
################################################################################
Seq, Seq_cnt, AgentParam, Param, _ = load_data(args.fPath)
SeqTensor = torch.vstack([torch.tensor(item, dtype=DTYPE, device=device)  for item in Seq])
Seq_train = SeqTensor[:numTrain, :]
A_true    = nx.adjacency_matrix(Param['G']).toarray()
gMat      = compute_gMat(Param)


Aggregator      = DeltaDiffAggre(Seq_train, Param)
BNPG_game       = BNPGModel(A_true, Aggregator)
lik_est, b_c_est, beta_est, eta_est = BNPG_game.learn(Seq_train, optimizer='cvx', noGroup=False)
lik_noG, _, _, _                    = BNPG_game.learn(Seq_train, optimizer='cvx', noGroup=True)
testStat        = 2 * (lik_est - lik_noG)
DF              = args.numAgent
pValue          = scis.chi2.sf(testStat, DF)
print(f"{numTrain} {args.graph_type} {pValue:.4f}")

################################################################################



