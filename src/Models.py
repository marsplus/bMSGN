import torch
import torch.nn as nn
from torch import optim
from collections import Counter

import numpy as np
import cvxpy as cvx
from time import time
from utils import computeDelta
DTYPE = torch.float32

    
### This is the bMSGN framework with capability of 
### learning the agents' utility functions.

class BNPGModel(nn.Module):
    def __init__(self, A, aggregator, regParam=0.0, gamma=5, num_random=15, verbose=False):
        """
            A:           the adjacency matrix of the underlying network
            aggregator:  used to count the difference between y_i and y_j (needed to evaluate Eq.(5))
            regParam:    whether to have regularization term during training
            gamma:       the parameter gamma in the logit-response dynamics
        """
        super(BNPGModel, self).__init__()
        self.A     = torch.tensor(A, dtype=DTYPE)
        self.n     = len(self.A)
        self.b_c   = nn.Parameter(torch.normal(mean=torch.zeros(self.n), std=torch.ones(self.n)))
        self.beta  = nn.Parameter(torch.normal(mean=torch.zeros(self.n), std=torch.ones(self.n)))
        self.eta   = nn.Parameter(torch.normal(mean=torch.zeros(self.n), std=torch.ones(self.n)))
        self.aggregator = aggregator
        self.groupLevel = self.aggregator.aggregate()
        self.gamma      = torch.tensor(gamma, dtype=DTYPE)

        ## regularization
        self.regParam = regParam

        self.verbose    = verbose
        self.num_random = num_random

    def _set_parameter(self, b_c, beta, eta):
        if not isinstance(b_c, torch.Tensor):
            b_c  = torch.tensor(b_c, dtype=DTYPE)
        if not isinstance(beta, torch.Tensor):
            beta = torch.tensor(beta, dtype=DTYPE)
        if not isinstance(eta, torch.Tensor):
            eta  = torch.tensor(eta, dtype=DTYPE)     
        with torch.no_grad():
            self.b_c.copy_(b_c)
            self.beta.copy_(beta)
            self.eta.copy_(eta)
    

    def _update_aggregator(self, aggregator):
        self.aggregator = aggregator
        self.groupLevel = self.aggregator.aggregate()
        

    def _random_restart(self):
        torch.manual_seed(int(time()))
        with torch.no_grad():
            for _, param in self.named_parameters():
                param.copy_(torch.normal(mean=torch.zeros(self.n), std=torch.ones(self.n)))


    def _get_adj(self):
        return self.A.numpy()


    def _enum_param(self):
        return (self.b_c, self.beta, self.eta)


    ### Eq.(2) in the paper
    ### P(xt | x_t_1, y_t_1)
    ### logP = logP(x1=1, x2=1, ..., )
    ### log1_P = logP(x1=0, x2=0, ..., )
    def _trans_prob(self, x, y):
        r = -self.b_c - self.beta * (self.A @ x) - self.eta * y
        logP   = torch.log( 1 / (1 + torch.exp( self.gamma * r)))
        log1_P = torch.log( 1 / (1 + torch.exp(-self.gamma * r)))
        return (logP, log1_P)


    ### Compute log-likelihood of the input sequence
    def _loglikelihood(self, Seq):
        if not isinstance(Seq, torch.Tensor):
            Seq = torch.vstack([torch.tensor(item, dtype=DTYPE) for item in Seq])
        T = len(Seq)
        loglik = 0
        for t in range(1, T):
            x_t     = Seq[t, :]
            x_t_1   = Seq[t-1, :]
            ## group-level statistics
            y_t_1   = self.groupLevel[t-1, :]  
            ## compute the transition probability
            logP, log1_P  = self._trans_prob(x_t_1, y_t_1)
            loglik += x_t.T @ logP
            loglik += (1 - x_t).T @ log1_P
        return loglik    


    ## Compute log-likelihood at aggregate level
    ## for the purpose of comparing with the baselines, e.g., Markov chain
    def aggLogLik(self, Seq, numSim=20000, eps=1e-8):
        if not isinstance(Seq, torch.Tensor):
            Seq = torch.vstack([torch.tensor(item, dtype=DTYPE) for item in Seq])
        L = 0
        T = len(Seq)
        missCnt = 0
        for t in range(1, T):
            xt_1 = Seq[t-1, :]
            xt_aggre = int(Seq[t, :].sum().item())

            ## group-level statistics
            yt_1 = self.groupLevel[t-1, :]
            
            ## compute the transition probability
            logP, _  = self._trans_prob(xt_1, yt_1)
            
            pt_est = torch.exp(logP)
            Cnt = Counter([int(torch.bernoulli(pt_est).sum().item()) for _ in range(numSim)])
            if xt_aggre in Cnt:
                L += torch.log(torch.tensor(Cnt[xt_aggre] * 1.0 / numSim))
            else:
                missCnt += 1
                L += torch.log(torch.tensor(eps))
        return L



    ## Generate confidence interval around predictions
    ## NOTICE: we are just doing one-step prediction,
    ## i.e., predicting the total counts at step t+1 based on the data on step t.
    def computeConfidence(self, Seq, numSim=10000):
        if not isinstance(Seq, torch.Tensor):
            Seq = torch.vstack([torch.tensor(item, dtype=DTYPE) for item in Seq])

        confidence = []
        T = len(Seq)
        for t in range(1, T):
            xt_1     = Seq[t-1, :]
            xt_aggre = Seq[t, :].sum().item() * 1.0
            
            ## group-level statistics
            yt_1     = self.groupLevel[t-1, :]

            ## compute the transition probability
            logP, _  = self._trans_prob(xt_1, yt_1)
            pt_est   = torch.exp(logP)

            ## simulate next-step aggregate counts
            nextTotals = [int(torch.bernoulli(pt_est).sum().item()) for _ in range(numSim)]
            avgTotals  = np.mean(nextTotals)
            radius     = np.std(nextTotals)
            confidence.append((avgTotals, radius, xt_aggre))
        return confidence

    
    ### This is mainly used for gradient-based learning
    def forward(self, Seq, with_reg=False):
        loglik = self._loglikelihood(Seq)
        if with_reg and self.regParam:
            ## add regularizations
            for param in self.regParam.keys():
                regFunc  = self.regParam[param]['regFunc']
                assert(callable(regFunc))
                regCoeff = self.regParam[param]['regCoeff'] 
                loglik  -= regFunc(getattr(self, param)) * regCoeff
        return -loglik


    ### Two ways to learn the model: 1) convex programming and 2) gradient-based optimization
    def learn(self, Seq, optimizer='lbfgs', **opt_args):
        if not isinstance(Seq, torch.Tensor):
            Seq = torch.vstack([torch.tensor(item, dtype=DTYPE) for item in Seq])
        
        ### formulate the learning problem as a convex programming
        ### and use off-the-shelf solver (we use CVXPY) to learn
        if optimizer == 'cvx':
            return self._cvx_learn(Seq, **opt_args)

        ### another way to learn is to use gradient-based method
        ### we tried the second-order method LBFGS
        rs_ret = []
        for _ in range(self.num_random):
            self._random_restart()
            if optimizer == 'lbfgs':
                b_c_, beta_, eta_ = self._lbfgs_learn(Seq, **opt_args)
                curr_lik = -1.0 * self.forward(Seq)
                if self.verbose:
                    print(f"(random start) log-likelihood: {curr_lik.item()}")
                rs_ret.append((curr_lik, b_c_, beta_, eta_))
        learned_ret = max(rs_ret, key=lambda x: x[0])
        return (item.detach().numpy() for item in learned_ret)


    def _lbfgs_learn(self, Seq, **opt_args):
        opt = optim.LBFGS([param for _, param in self.named_parameters()], **opt_args)
        def closure():
            opt.zero_grad()
            loss = self.forward(Seq)
            loss.backward(retain_graph=True)
            if self.verbose:
                print(f"(lbfgs) Loss: {loss.item()}")
            return loss
        opt.step(closure)
        return ( self.b_c.detach(), self.beta.detach(), self.eta.detach() )


    ## Apply interior-point method to accurately estimate the parameters
    ## this is VERY MEMORY INTENSIVE!!!
    def _cvx_learn(self, Seq, **opt_args):
        noGroup = opt_args['noGroup']
        Reg     = self.regParam

        ## convert everying to numpy format
        T        = len(Seq)
        Seq      = Seq.numpy()
        groupLevel   = self.groupLevel.numpy()
        A        = self.A.numpy()
        gamma    = self.gamma.numpy()

        b_c  = cvx.Variable(self.n)
        eta  = cvx.Variable(self.n)
        beta = cvx.Variable(self.n)
        
        obj = 0
        for t in range(1, T):
            X_t       = Seq[t, :]
            X_t_1     = Seq[t-1, :]
            Delta_hat = groupLevel[t-1, :]
            r         = -b_c - cvx.multiply(beta, A @ X_t_1) - cvx.multiply(eta, Delta_hat)
            obj       += -1 * X_t.T @ cvx.logistic(r * gamma)
            obj       += -1 * (1 - X_t).T @ cvx.logistic(-r * gamma)
        
        const = []
        const.extend([eta >= -1, eta <= 1])
        const.extend([beta >= -1, beta <= 1])

        ## for the statistical test
        if noGroup:
            const.extend([eta == 0])

        prob = cvx.Problem(cvx.Maximize(obj), const)
        prob.solve(solver="MOSEK", verbose=False)
        return (prob.value, b_c.value, beta.value, eta.value)
        


### aggregators ###
####################################################################################
class Aggregator:
    def __init__(self, Seq, Param):
        self.n = len(Seq[0])
        if not isinstance(Seq, torch.Tensor):
            Seq = torch.vstack([torch.tensor(item, dtype=DTYPE) for item in Seq])
        self.Seq = Seq
        self.Param = Param
    
    def aggregate(self):
        raise NotImplementedError('Must be implemented in child class')


## relative difference between y_i and y_j
class DeltaDiffAggre(Aggregator):
    def aggregate(self):
        K    = self.Param['K']
        PtoG = self.Param['player-group']
        GtoP = self.Param['group-player']

        T = len(self.Seq)
        ret = []
        for t in range(T):
            X_t = self.Seq[t, :]
            DeltaVec = torch.zeros(self.n)
            for i in range(self.n):
                selfGroupIdx = PtoG[i]
                delta_self = torch.sum(X_t[GtoP[selfGroupIdx]])
                otherGroups = [g for g in range(K) if g != selfGroupIdx]
                Delta = delta_self
                for otherGroupIdx in otherGroups:
                    delta_other = torch.sum(X_t[GtoP[otherGroupIdx]])
                    Delta -= delta_other / len(otherGroups)
                DeltaVec[i] = Delta
            ret.append(DeltaVec)
        return torch.vstack(ret)


## NN-based aggregator
class NNAggre(Aggregator):
    def __init__(self, Seq, Param, numHidden=10):
        super(NNAggre, self).__init__(Seq, Param)
        self.K = self.Param['K']
        self.n = self.Param['numAgent']
        self.GtoP = self.Param['group-player']
        self.hiddenLayer = nn.Linear(self.n, numHidden)
        self.outputLayer = nn.Linear(numHidden, self.n)

    def aggregate(self):
        # gMat = torch.zeros((self.K, self.n))
        # for g in range(self.K):
        #     gMat[g, self.GtoP[g]] = 1
        
        ret = []
        T = len(self.Seq)
        for t in range(T):
            X_t = self.Seq[t, :]
            h   = torch.tanh(self.hiddenLayer(X_t))
            out = self.outputLayer(h)
            ret.append(out)
        return torch.vstack(ret)


## zero aggregator (this is specifically for cases without group structure)
class ZeroAggre(Aggregator):
    def aggregate(self):
        T = len(self.Seq)
        ret = []
        for _ in range(T):
            DeltaVec = torch.zeros(self.n)
            ret.append(DeltaVec)
        return torch.vstack(ret)
          

