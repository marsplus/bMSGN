import json
import argparse
import numpy as np
import dill as pickle
import networkx as nx
from modelAgentBased import NetworkGame
from networkx.algorithms.community import greedy_modularity_communities


parser = argparse.ArgumentParser()
parser.add_argument('--timeStep',       type=int,   default=100,  help='time steps')
parser.add_argument('--gamma',          type=float, default=5,    help='temperate')
parser.add_argument('--graph_type',     type=str,   default='BA', help='graph type')
parser.add_argument('--numAgent',       type=int,   default=100,  help='number of agents')
parser.add_argument('--seed',           type=int,   default=123,  help='random seed')
parser.add_argument('--burnin',         type=int,   default=200,  help='#(burnin)')
parser.add_argument('--fPath',          type=str,   default="")
parser.add_argument('--oPath',          type=str,   default="")
parser.add_argument('--noGroup',        type=int,   default=0)
parser.add_argument('--liktest',        type=int,   default=0)
args = parser.parse_args()
SEED = args.seed
np.random.seed(SEED)


n = args.numAgent
if args.graph_type == 'SBM':
    ## simulate a stochastic-block model
    sizes = [20, 30, 50]
    N = np.sum(sizes)
    probs = [[0.3, 0.05, 0.03], [0.05, 0.3, 0.05], [0.03, 0.05, 0.3]]
    G = nx.stochastic_block_model(sizes, probs, seed=SEED)
elif args.graph_type == 'BA':
    G = nx.barabasi_albert_graph(n, 3, seed=SEED)
elif args.graph_type == 'SW':
    G = nx.watts_strogatz_graph(n, 10, 0.2, seed=SEED)
elif args.graph_type == 'RG':
    G = nx.random_geometric_graph(n, 0.2, seed=SEED)
elif args.graph_type == 'BTER':
    Adj = np.load('../result/BTER/BTER_{}.npy'.format(args.seed))
    G = nx.from_numpy_array(Adj)
else:
    raise ValueError("Unknow graph type")



assert(nx.is_connected(G))
Adj_true = nx.adjacency_matrix(G).todense()

## partition the graph by community detection
Comm = list(greedy_modularity_communities(G))
numComm = len(Comm)
print("Number of groups: {}".format(numComm))
G.graph["partition"] = {idx: list(Comm[idx]) for idx in range(numComm)}


### read into config.
configPath = f'sim-config/{args.graph_type}_synthetic_config.txt' if not args.fPath else args.fPath
with open(configPath, 'r') as fid:
    config = json.load(fid)


### generate group information and some other parameters
Param = {
    'K'    :         numComm,
    'b'    :         config['b'],
    'beta' :         config['beta'],
    'cost' :         config['cost'],
    'gamma':         args.gamma,
    'G'    :         G,
    'numAgent'    :  len(G),
    'all-group'   :  list(range(numComm)),
    'group-player':  {idx: list(G.graph["partition"][idx]) for idx in range(numComm)},
    'player-group':  {idx: group for group, players in G.graph["partition"].items() for idx in players},
    'eta'           : config['eta']
}


Game = NetworkGame(Param, noGroup=args.noGroup)
output, output_cnt = Game.gen_seq(args.timeStep, burnin=args.burnin)
AgentParam = Game.get_agentParam()
allDelta = Game.allDelta



if not args.liktest:
    outPath = f'../result/synthetic/Data_timeStep_{args.timeStep}_gamma_{args.gamma:.2f}_numAgent_{args.numAgent}_graphType_{args.graph_type}_seed_{args.seed}.p' \
              if not args.oPath else args.oPath
    with open(outPath, 'wb') as fid:
        pickle.dump((output, output_cnt, AgentParam, Param, allDelta), fid)
### simulate data for the likelihood test
else:
    mode = "withGroup" if not args.noGroup else "noGroup"
    outPath = f'../result/LikRatio/Data_timeStep_{args.timeStep}_gamma_{args.gamma:.2f}_numAgent_{args.numAgent}_graphType_{args.graph_type}_seed_{args.seed}_{mode}.p' \
             if not args.oPath else args.oPath
    with open(outPath, 'wb') as fid:
        pickle.dump((output, output_cnt, AgentParam, Param, allDelta), fid)
