import torch
import argparse
import numpy as np
import networkx as nx
from Models import BNPGModel, DeltaDiffAggre
from utils import load_data, computeDelta


parser = argparse.ArgumentParser()
parser.add_argument('--Iter',           type=int,    default=50,        help='train iterations')
parser.add_argument('--Lambda',         type=float,  default=0,         help='regularization to promote sparsity')
parser.add_argument('--timeStep',       type=int,    default=500,       help='time steps')
parser.add_argument('--gamma',          type=float,  default=5,         help='gamma')
parser.add_argument('--numAgent',       type=int,    default=100,       help='#(agent)')
parser.add_argument('--graph_type',     type=str,    default='BA',      help='graph type')
parser.add_argument('--seed',           type=int,    default='123',     help='random seed')
parser.add_argument('--numTrain',       type=int,    default=500,       help='number of training data')
parser.add_argument('--numTest',        type=int,    default=0,         help='number of testing data')
parser.add_argument('--num_restart',    type=int,    default=3,         help='number of restarts')
parser.add_argument('--group',          type=str,    default='noGroup', help='whether to have group structure')
parser.add_argument('--lr',             type=float,  default=0.001,     help='lr')
parser.add_argument('--fPath',          type=str,    default='')
parser.add_argument('--verbose',        type=int,    default=1)
args = parser.parse_args()

SEED = args.seed
np.random.seed(SEED)
torch.manual_seed(SEED)
DTYPE = torch.float64

n = args.numAgent
Seq, Seq_cnt, AgentParam, Param, DeltaTrue = load_data(
    '../result/synthetic/Data_timeStep_{}_gamma_{:.2f}_numAgent_{}_graphType_{}_seed_{}.p'.format(args.timeStep, args.gamma,
            args.numAgent, args.graph_type, args.seed)
)
numTrain = args.numTrain
numTest  = args.numTest
trainSeq = Seq[:numTrain]
testSeq  = Seq[numTrain:numTrain+numTest]
A_true = nx.adjacency_matrix(Param['G']).toarray()

b_c_true  = AgentParam['b'] - AgentParam['cost']
beta_true = AgentParam['beta'] 
eta_true  = AgentParam['eta'] 

## aggregator
Aggregator = DeltaDiffAggre(trainSeq, Param)

## game model 
BNPG = BNPGModel(A_true, Aggregator, Param, verbose=args.verbose, num_random=args.num_restart)
lik_est, b_c_est, beta_est, eta_est = BNPG.learn(trainSeq, optimizer='cvx', noGroup=False)

Dist_b_c  = np.linalg.norm(b_c_est  -  b_c_true)
Dist_beta = np.linalg.norm(beta_est -  beta_true)
Dist_eta  = np.linalg.norm(eta_est  -  eta_true) 

print(f"{args.timeStep} {lik_est} {Dist_b_c} {Dist_beta} {Dist_eta}")
