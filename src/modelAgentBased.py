import numpy as np
from scipy.special import logsumexp


### This script is used for simulating synthetic data ### 


### the game agent used in our experiments
### notice that the implementation 
### specifies on linear-quadratic utilities.
class GameAgent(object):
    def __init__(self, Param, groupID, globalID, noGroup=False):
        '''
            Param:    simulation parameters
            groupID:  which group the agent belongs to
            globalID: a unique identifier of the agent in the game
            noGroup:  whether the group structure exists
        '''
        super(GameAgent, self).__init__()
        self.gamma = Param['gamma']

        self.b     = Param['b']    + np.random.normal(scale=0.01)
        self.beta  = Param['beta'] + np.random.normal(scale=0.01)
        self.cost  = Param['cost'] + np.random.normal(scale=0.1)
        self.eta   = 0 if noGroup else Param['eta'] + np.random.normal(scale=0.01)

        self.b     = Param['b']    if self.b    < 0 else self.b
        self.beta  = Param['beta'] if self.beta > 0 else self.beta
        self.cost  = Param['c']    if self.cost < 0 else self.cost
        
        self.groupID  = groupID
        self.globalID = globalID
        
        ### the probability to commit crimes at current step
        self.Prob = 0

        ### action at the previous time step, i.e., x^t_i in the paper
        ### x^t_i: the action of agent i at time step t
        self.actionCurrent = 0


    def utility(self, decision, numNbrInvest, Delta_hat):
        '''
            Delta_hat: total differences in investment between player's group and all other groups
        '''
        U = (self.b - self.cost) * decision + self.beta * decision * numNbrInvest \
                    + self.eta * decision * Delta_hat
        return U


    def response_dynamics(self, numNbrInvest, Delta):  
        ### if x^t_i = 1 -> compute u^{t+1}_i
        if self.actionCurrent == 1:
            util_action_1 = self.utility(1.0, numNbrInvest, Delta)
            util_action_0 = self.utility(0.0, numNbrInvest, Delta-1)
        ### if x^t_i = 0 -> compute u^{t+1}_i
        elif self.actionCurrent == 0:
            util_action_1 = self.utility(1.0, numNbrInvest, Delta+1)
            util_action_0 = self.utility(0.0, numNbrInvest, Delta)
        else:
            raise ValueError("x^t_i is not correct\n")


        ### logit best response
        S = logsumexp([self.gamma * util_action_1, self.gamma * util_action_0])
        prob_1 = self.gamma * util_action_1 - S
        self.Prob = np.exp(prob_1)
        ### sample the player's action
        self.actionCurrent = 1.0 if np.random.rand() <= self.Prob else 0.0
        return self.actionCurrent

    


### Define the network game for simulation ###
class NetworkGame(object):
    def __init__(self, Param, noGroup=False):
        super(NetworkGame, self).__init__()
        self.numAgent    = Param['numAgent']
        self.agentList   = [GameAgent(Param, Param['player-group'][i], i, noGroup) for i in range(self.numAgent)]
        self.globalState = np.zeros(self.numAgent)
        self.G           = Param['G']

        self.allGroups   = Param['all-group']

        ### given a group idx, returns all the players in it
        self.groupToPlayers = Param['group-player']

        ### given a player idx, returns the group the agent belongs to
        self.playerToGroup = Param['player-group']

        ### record Delta information at each step for debug purpose
        self.allDelta = []


    def get_neighbors(self, ID):
        return self.G.neighbors(ID)


    def get_agentParam(self):
        ret = {
            'b':    np.array([agent.b    for agent in self.agentList]),
            'cost': np.array([agent.cost for agent in self.agentList]),
            'beta': np.array([agent.beta for agent in self.agentList]),
            'eta':  np.array([agent.eta  for agent in self.agentList])
            }
        return ret


    ### set agents' parameters with external values 
    ################################################
    def set_b(self, b_hat):
        b_avg = np.mean(b_hat)
        for idx, agent in enumerate(self.agentList):
            agent.b = b_avg if b_hat[idx] < 0 else b_hat[idx]
            
    def set_beta(self, beta_hat):
        beta_avg = np.mean(beta_hat)
        for idx, agent in enumerate(self.agentList):
            agent.beta = beta_avg if beta_hat[idx] > 0 else beta_hat[idx]

    def set_cost(self, cost_hat):
        cost_avg = np.mean(cost_hat)
        for idx, agent in enumerate(self.agentList):
            agent.beta = cost_avg if cost_hat[idx] < 0 else cost_hat[idx]  

    def set_eta(self, eta_hat):
        for idx, agent in enumerate(self.agentList):
            agent.eta = eta_hat[idx]
    ################################################

    ### set agents' actions to paticular values
    def set_agentAction(self, x):
        for idx, a in enumerate(self.agentList):
            a.actionCurrent = x[idx]


    def step(self):
        N = self.numAgent
        next_state = self.globalState.copy()
        DeltaCurrent = []
        for i in range(N):
            agent = self.agentList[i]
            nbrs = list(self.get_neighbors(i))

            ### the number of investing neighbors 
            numNbrInvest = np.sum(self.globalState[nbrs])

            ### the total difference in investment between player i's group and the other groups
            selfGroupIdx  = agent.groupID
            selfGroupPlayers = self.groupToPlayers[selfGroupIdx]
            ### player i should be in her own group
            assert(agent.globalID in selfGroupPlayers)
            delta_self = np.sum(self.globalState[selfGroupPlayers]) 
            
            otherGroups = [g for g in self.allGroups if g != selfGroupIdx]
            Delta = delta_self
            for otherGroupIdx in otherGroups:
                otherGroupPlayers = self.groupToPlayers[otherGroupIdx]
                ### player i should be in only one group
                assert(agent.globalID not in otherGroupPlayers)
                delta_other = np.sum(self.globalState[otherGroupPlayers])
                Delta -= delta_other / len(otherGroups)
            
            action = agent.response_dynamics(numNbrInvest, Delta)
            next_state[i] = action

            ## record Delta for debug purpose
            DeltaCurrent.append(Delta)
        self.allDelta.append(np.array(DeltaCurrent))
        self.globalState = next_state


    """
    generate a sequence of states
    burnin: to make sure the underlying Markov chain 
            is in stationarity
    """
    def gen_seq(self, Iter, burnin=200):
        output = []
        output_cnt = []
        for i in range(burnin + Iter):
            if i < burnin:
                self.step()
                if i % 50 == 0:
                    print("burnin: {}".format(i))
                continue
            self.step()
            output.append(self.globalState)
            output_cnt.append(np.sum(self.globalState))
        ### discard the Delta corresponding to burnin
        self.allDelta = self.allDelta[burnin+1:]
        return output, output_cnt



    ### simulate non-stationary data by
    ### increasing the marginal benefit b
    def gen_seq_burst(self, Iter, burnin=200, strength=0.2, window=10):
        """
            Increase the marginal benefit by percent
            Simulate a burst with length=window
        """
 
        assert(strength >= 0 and strength <= 1) 
        b_backup = self.get_agentParam()['b']
        b_new    = b_backup * (1 + strength) 

        ## pick a starting point (make sure the starting point is not in the testing data)
        start = np.random.choice(range(burnin, burnin + int(Iter*0.5)))
        
        output = []
        output_cnt = []
        start_burst = False 
        for i in range(burnin + Iter):
            if i < burnin:
                self.step()
                if i % 50 == 0:
                    print("burnin: {}".format(i))
                continue
          
            if i >= start and i < start + window: 
                if not start_burst:
                    start_burst = True 
                    self.set_b(b_new)
                self.step()
                output.append(self.globalState)
                output_cnt.append(np.sum(self.globalState))
            else:
                if start_burst:
                    start_burst = False
                    self.set_b(b_backup)  
                self.step()
                output.append(self.globalState)
                output_cnt.append(np.sum(self.globalState))
        
        ## discard the Delta corresponding to burnin
        self.allDelta = self.allDelta[burnin+1:]
        return output, output_cnt



### the following are some experimental code which are not used in the paper
############################################################################################################
### Simulate Information Propagation with DeGroot's model ###

class InfoAgent(object):
    def __init__(self, alpha, w):
        self.alpha = alpha
        self.w = w
    
    def action(self, NbrInvRatio):
        Prob = (1 - self.alpha) * self.w + self.alpha * NbrInvRatio

        ## make sure 0 <= Prob <= 1
        Prob = min(1, Prob)
        return 1.0 if np.random.rand() <= Prob else 0


class BeliefGame(object):
    def __init__(self, Param):
        super(BeliefGame, self).__init__()
        self.numAgent = Param['numAgent']
        self.agentList = [InfoAgent(Param['alpha'][i], Param['w'][i]) for i in range(self.numAgent)]
        self.globalState = np.zeros(self.numAgent)
        self.G = Param['G']

    def get_neighbors(self, ID):
        return self.G.neighbors(ID)

    def step(self):
        N = self.numAgent
        next_state = self.globalState.copy()
        for i in range(N):
            agent = self.agentList[i]
            nbrs = list(self.get_neighbors(i))

            ## the ratio of neighbors who invest
            NbrInvRatio = np.sum(self.globalState[nbrs]) * 1.0 / len(nbrs)
            
            action = agent.action(NbrInvRatio)
            next_state[i] = action
        self.globalState = next_state


    def gen_seq(self, Iter, burnin=200):
        output = []
        output_cnt = []
        for i in range(burnin + Iter):
            if i < burnin:
                self.step()
                if i % 50 == 0:
                    print("burnin: {}".format(i))
                continue
            self.step()
            output.append(self.globalState)
            output_cnt.append(np.sum(self.globalState))
        return output, output_cnt
