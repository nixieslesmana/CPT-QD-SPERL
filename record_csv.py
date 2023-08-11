# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 20:21:27 2022

@author: Nixie S Lesmana
"""

import csv
import numpy as np
#from datetime import datetime
import pandas as pd
import itertools
import matplotlib.pyplot as plt

import os

# env = 'shiBM', 'barberis', ...
# algo = 'SPERL', 'SPE', 'SPSA', 'Precomm', ...
# runID = datetime.now().strftime("%d%m%y%H%M%S")

'''
### Check HOLDER PARAMETERS ####
cptParams = [(.88, .65, 2.25), (.95, .5, 1.5), (1., 1., 1.)] 

for cptID in cptParams:
    alpha, rho1, lmbd = cptID
    rho2 = rho1
    
    print('alpha=', alpha)
    
    ################# CPTw ##################
    
    
    def w(F, pos, rho1 = rho1, rho2 = rho2): 
        # @Prash: rho1 = .61, rho2 = .69 \in[0.3, 1]; assert w(F) monotonic in F;
        # @Barberis: rho1 = rho2 = .5 
    
        if pos == True:
            return F**rho1 / ((F**rho1 + (1-F)**rho1)**(1/rho1))
        else:
            return F**rho2 / ((F**rho2 + (1-F)**rho2)**(1/rho2))
    
    #beta = 1 #.15 #rho1 - .3 # must be >= 1 - alpha
    if .9 < 1-alpha+.05:
        print('---')
        continue
    
    for beta in np.linspace(1-alpha+.05, .9, 5):
        beta = np.round(beta, 2)
        print('w, beta=', beta)
        
        K = 20
        X = np.linspace(0, 1, 5000)
        Y = np.linspace(0, 1, 5000)
        xy = [(x, y) for x in X for y in Y if x != y and abs(x - y) <= 1/K]
        
        denom = np.array([(abs(x - y))**beta for x, y in xy])
        numer = np.array([abs(w(x, True) - w(y, True)) for x, y in xy]) 
        
        # numer is problematic!!
        # different pair of x, y can have similar values |x - y|^beta
        # for diff pairs of x, y, |w(x) - w(y)| v close 
        
        # Assump on w Holder is to ensure integrability [Prash 16]
        
        # why denom cannot go smaller than .15?
        
        plt.figure()
        plt.scatter(denom, numer, label = 'beta_w=' + str(beta))
        plt.legend()
        
        plt.title('|w(x) - w(y)| vs |x - y|^beta_w \n CPT: ' + str(cptID))
        plt.savefig("./cpt{}_K{}_wHolder{}.png".format(alpha, K, beta), dpi =100)
        plt.close()

        ################# CPTu ##################
        X = np.linspace(0, 100, 200) # SUPPORT = {-50, 50} -- if K =50, each z_k, z_k-1 gap ~= 2; beta_u at denom > 2. is INFINITY
        Y = np.linspace(0, 100, 200)
        xy = [(x, y) for x in X for y in Y if abs(x - y) <= 100.]#x != y]
        
        #beta_u = alpha
        if .9 < 1-beta+.05:
            print('---')
            continue
        
        for beta_u in np.linspace(1-beta+.05, .9, 5):
            beta_u = np.round(beta_u, 2)
            print('u, beta=', beta_u)
            
            denom_u = np.array([(abs(x - y))**beta_u for x, y in xy])
            numer_u = np.array([abs(x**alpha - y**alpha) for x, y in xy])
        
            plt.figure()
            plt.scatter(denom_u, numer_u, label = 'beta_u=' + str(beta_u))
            plt.legend()
            
            plt.title('|u(x) - u(y)| vs |x - y|^beta_u \n CPT: ' + str(cptID))
            plt.savefig("./cpt{}uHolder{}.png".format(alpha, beta_u), dpi =100)
            plt.close()
            
        print('---')

#plt.figure()
#plt.plot(numer/denom, label = 'beta=' + str(beta))
#plt.legend()
'''

'''
######### QDRL - CPTPE Proof Illustrate ######
import math

suppMin = 50
suppMax = -50

for seed in range(10):
    #seed = 7
    np.random.seed(seed)
    
    tauVals = np.random.choice(np.linspace(suppMin, suppMax, 101), size=5-1, replace=False)
    tauVals = np.round(np.sort(tauVals), 4)
    tauVals = [-50] + list(tauVals) + [50]
    
    var = np.random.choice(range(15, 20), size=len(tauVals)-1)
    
    X = []
    for i in range(len(tauVals)-1):
        mu = (tauVals[i] + tauVals[i+1]) / 2
        sig = math.sqrt(var[i])
        
        if i == len(tauVals)-2:
            vals = np.random.normal(mu, sig, size=201)
        else:
            vals = np.random.normal(mu, sig, size=200)
            
        X += list(vals)
        
    X = np.sort(X)
    Y = np.linspace(0, 1, 1001)
    
    plt.figure()
    plt.plot(X, Y, color = 'black', linewidth=4, label = r'$\vartheta$')
    plt.title('seed='+str(seed))

    K = 5
    tauLocs = np.arange(0, 6)/K
    tauMids = (tauLocs[:-1] + tauLocs[1:]) / 2
    
    XMids = []
    for q in tauMids:
        XMids += [np.quantile(X, q, interpolation = 'higher')]
    XMids += [max(X)]
    
    plt.step(XMids, tauLocs, color='tab:blue', linewidth=4, label = r'$\Pi_{W_1}\vartheta$')
    plt.scatter(XMids[:-1], tauLocs[1:], s=50, color='tab:blue')
    
    # fill-between
    x = X
    y1 = Y
    y2 = []
    for z in X:
        Zvals = [min(X)] + XMids
        for i in range(len(Zvals)-1):
            i_z = i
            if z >= Zvals[i] and z < Zvals[i+1]:
                break
            
        if z == XMids[0]:
            print('z:', z)
            print('i_z:', i_z, '=1?')
            
        y2 += [tauLocs[i_z]]
    plt.fill_between(x, y1, y2, color = 'r', alpha = .5)
    
    plt.ylim([0, 1.05])
    plt.xlim([min(X), max(X)])
    
    # remove tick labels/marks
    plt.tick_params(bottom=False)
    plt.xticks(color = 'w')
    plt.yticks(color = 'w')
    
    plt.legend(loc='lower right')
    plt.xlabel('Space of Returns')
    plt.ylabel('Probability Space')
    
    plt.savefig("./qdrl_illustrate_fill_{}.png".format(seed), dpi = 200)
'''

'''
# QR Loss illustrate
U = np.linspace(-2.5, 2.5, 100)

for tau in np.linspace(.1, .9, 3):
    Loss = (tau - np.ones(len(U))*(U<0)) * U
    plt.plot(U, Loss, label = r'$\tau=$'+str(tau))
plt.ylabel(r'$\rho(u; \tau)$')
plt.xlabel('u')
plt.legend()
'''


'''
### Check GAME PAYOFF (Var)
# Player: X, Y
# Actions: a, b
rX = 1
rY = 2
gamma = .9 # 0, 1:gamma**0; 2, 3:gamma**2; ..
p_Xa = .5
p_Xb = .3
p_Ya = .8
p_Yb = .7

count = 0

for rY in [2]: #[1.4, 1.6, 1.8, 2]:
    for gamma in [.99, .9, .8, .7, .6, .5]:#np.linspace(.5, 1, 6):
        for p_Xa in np.linspace(0.1, .9, 9):
            for p_Xb in np.linspace(0.1, .9, 9):
                for p_Ya in np.linspace(0.1, .9, 9):
                    for p_Yb in np.linspace(0.1, .9, 9):
        
                        def Payoff(actX, actY, p_Xa = p_Xa, p_Xb = p_Xb, p_Ya = p_Ya, p_Yb = p_Yb, rX = rX, rY = rY, gamma = gamma):
                        
                            if actX == 1 and actY == 0:
                                p_X = p_Xb
                                p_Y = p_Ya
                            elif actX == 0 and actY == 1:
                                p_X = p_Xa
                                p_Y = p_Yb
                                
                            elif actX == 0 and actY == 0:
                                p_X = p_Xa
                                p_Y = p_Ya
                            else:
                                p_X = p_Xb
                                p_Y = p_Yb
                            
                            pVec_X = np.array([p_X, (1-p_X)*p_Y])
                            pVec_Y = np.array([p_Y, (1-p_Y)*p_X])
                            uVec_X = np.array([rX, rY*gamma]) #, rX*(gamma**2), rY*(gamma**3)] 
                            uVec_Y = np.array([rY, rX*gamma]) #, rY*(gamma**2), rX*(gamma**3)]
                            
                            M = (1-p_X)*(1-p_Y)
                            
                            J_X = 1/(1-gamma**4 * M) * np.dot(pVec_X, uVec_X**2) - (1/(1-gamma**2*M)  * np.dot(pVec_X, uVec_X))**2
                            J_Y = 1/(1-gamma**4 * M) * np.dot(pVec_Y, uVec_Y**2) - (1/(1-gamma**2*M)  * np.dot(pVec_Y, uVec_Y))**2
                            
                            return (J_X, J_Y)
                        
                        PayoffTable = {}
                        for actX in [0, 1]:
                            for actY in [0, 1]:
                                PayoffTable[(actX, actY)] = Payoff(actX, actY)
                                
                        # for each entry, check if NE
                        # aa ab
                        # ba bb
                        # r0, c0 = 0, 0:
                        # want minVar
                        # fix r, if c=c0 <= c=1-c0: then ok
                        # fix c, if r=r0 <= r=1-r0: then ok
                        # bool = ok * ok
                        # WTS: for all r0, c0: bool = 0
                        
                        bool_ = []
                        str_ = ''
                        for r0 ,c0 in PayoffTable.keys():
                            bool_ += [(list(PayoffTable[(r0, c0)])[1] <= list(PayoffTable[(r0, int(1-c0))])[1]) * (list(PayoffTable[(r0, c0)])[0] <= list(PayoffTable[(int(1-r0), c0)])[0])]
                            str_ += str(PayoffTable[(r0, c0)])
                            str_ += '\n'
                            
                        
                        # NE does not exist
                        if sum(bool_) <= 0:    
                            print('rY:', rY)
                            print('gamma:', gamma)
                            print('p:', (p_Xa, p_Xb, p_Ya, p_Yb))
                            print(bool_) # want: ALL FALSE!
                            print(PayoffTable)
                            
                            raise ValueError
                        
                        
                        # NE exists unique, n problem is TIC
                        utilX = []
                        utilY = []
                        for r0, c0 in PayoffTable.keys():
                            utilX += [PayoffTable[(r0, c0)][0]]
                            utilY += [PayoffTable[(r0, c0)][1]]
                            
                        if np.argmin(utilX) != np.argmin(utilY) and sum(bool_) == 1 and np.argmax(bool_) != np.argmin(utilX) and np.argmax(bool_) != np.argmin(utilY):
                            print('rY:', rY)
                            print('gamma:', gamma)
                            print('p:', (p_Xa, p_Xb, p_Ya, p_Yb))
                            print('---')
                            print(str_)
                            
                            print('pure NE:', np.argmax(bool_), bool_) # want: ALL FALSE!
                            
                            print('x0=x:', np.argmin(utilX))
                            print('x0=y:', np.argmin(utilY))
                            
                            
                            count += 1
                            
                            if count > 10:
                                raise ValueError

'''

def record_params(hyperparams, env, algo, runID, dyn = False):
    
    print('Check params now! .csv does not contain params - so,')
    print('if we have more than one param set to try, add new index to .csv')
    print('or make paramID, or read .txt to extract params, or infer from')
    print('.csv columns(lvl=1) when joining..')
    
    if not dyn:
        filename = './{}/results/static/{}_{}.txt'.format(env, algo, runID)
    else:
        filename = './{}/results/dynamic/{}_{}.txt'.format(env, algo, runID)
        
    f = open(filename, 'a', newline = '')    
    
    for h in hyperparams:
        f.write('{}\n'.format(h))
    f.close()


def record_quantiles(to_append, env, algo, runID, saID, qtileTransform = None, dyn = False):
    
    state, a = saID
    t, x = state
    
    if qtileTransform is not None: # str
        if not dyn:
            filename = './{}/results/static/{}_{}_QFDyn_{}_{}_{}_{}.csv'.format(env, algo, runID, qtileTransform, t, x, a)
        else:
            filename = './{}/results/dynamic/{}_{}_QFDyn_{}_{}_{}_{}.csv'.format(env, algo, runID, qtileTransform, t, x, a)
    else:
        if not dyn:
            filename = './{}/results/static/{}_{}_QFDyn_{}_{}_{}.csv'.format(env, algo, runID, t, x, a)
        else:
            filename = './{}/results/dynamic/{}_{}_QFDyn_{}_{}_{}.csv'.format(env, algo, runID, t, x, a)
            
    f = open(filename, 'a', newline = '')
    writer = csv.writer(f)
    writer.writerows(to_append)
    f.close()

def record_results(model, env, algo, runID, seed = None, cptMdpID=None, ablationID=None, feat = [None], dyn = False):
    
    if cptMdpID is None:
        print('Set cptMdpID in main.py!')
        
    alpha, rho, lmbd, pwin = cptMdpID
    
    zidx_to_z, = feat
    if zidx_to_z is None:
        zidx_to_z = lambda z: z
    
    if not dyn:
        filename = './{}/results/static/{}_{}.csv'.format(env, algo, runID)
    else:
        filename = './{}/results/dynamic/{}_{}.csv'.format(env, algo, runID)
    
    runIDstr = runID
    runID = int(runIDstr)
    
    if algo == 'SPERL':
        cpt_est = dict()
        for x in model.CPT_val.keys():
            s, a = x
            s = zidx_to_z(s)
            t, z = s
            cpt_est[t, z, a] = np.array(model.CPT_val[x]).flatten() # 1-dim array
        CPT_val = pd.DataFrame(cpt_est)
        iter_nums = range(len(CPT_val.index))
        tuples = itertools.product([runID], ['q_i'], iter_nums)
        CPT_val.index = pd.MultiIndex.from_tuples(tuples)
        
        cpt_true = dict()
        for x in model.CPT_true.keys():
            s, a = x
            s = zidx_to_z(s)
            t, z = s
            cpt_true[t, z, a] = np.array(model.CPT_true[x]).flatten()
        CPT_true = pd.DataFrame(cpt_true)
        tuples = itertools.product([runID], ['q_true'], iter_nums)
        CPT_true.index = pd.MultiIndex.from_tuples(tuples)
        
        policy_val = CPT_val.copy() # iter_num rows
        for r in CPT_val.index:
            for c1, c2, _ in CPT_val.columns:
                c = (c1, c2)
                
                policy_val.loc[r, c] = np.zeros(len(CPT_val.loc[r, c]))
                max_id = np.argmax(CPT_val.loc[r, c])
                policy_val.loc[r, c][max_id] = 1
        tuples = itertools.product([runID], ['policy_i'], iter_nums)
        policy_val.index = pd.MultiIndex.from_tuples(tuples)
        policy_val = policy_val.sort_index(axis=1)
        
        visit_freq = dict()
        for x in model.visitFreq.keys():
            s, a = x
            #s = zidx_to_z(s)
            t, z = s
            visit_freq[t, z, a] = np.array(model.visitFreq[x]).flatten()
        visit_freq = pd.DataFrame(visit_freq)
        tuples = itertools.product([runID], ['visit_freq_total'], iter_nums) # no need iter_nums? q_true need cz depends on policy_i
        visit_freq.index = pd.MultiIndex.from_tuples(tuples)
        
        df = pd.concat([CPT_val, policy_val, CPT_true, visit_freq], axis=0, join = 'inner')
        df.to_csv(filename)
        
        # AGGR seeds
        if ablationID is not None:
            K, lbub, smoothen, ss_inverted, eps = ablationID
            if not dyn:
                filename = './{}/results/static/{}_aggr_{}_{}_{}_{}_{}_{}_{}_{}_{}_'.format(env, algo, alpha, rho, lmbd, pwin, K, lbub, smoothen, ss_inverted, eps)
            else:
                filename = './{}/results/dynamic/{}_aggr_{}_{}_{}_{}_{}_{}_{}_{}_{}_'.format(env, algo, alpha, rho, lmbd, pwin, K, lbub, smoothen, ss_inverted, eps)
            
        else:
            # set default: K, lbub, smoothen, ss_inverted = 50, 0, 0, 0
            if not dyn:
                filename = './{}/results/static/{}_aggr_{}_{}_{}_{}_'.format(env, algo, alpha, rho, lmbd, pwin)
            else:
                filename = './{}/results/dynamic/{}_aggr_{}_{}_{}_{}_'.format(env, algo, alpha, rho, lmbd, pwin)
            
        idx = pd.IndexSlice 
        dfa = CPT_val.loc[idx[:, :, CPT_val.shape[0]-1], :] # last row, trained Q-table
        tuples = itertools.product([runID], [seed], [CPT_val.shape[0]-1])
        dfa.index = pd.MultiIndex.from_tuples(tuples)
        filename_ = filename + 'trainQ.csv'
        dfa.to_csv(filename_, mode='a', header=not os.path.exists(filename_))
        
        dfa = CPT_true.loc[idx[:, :, CPT_true.shape[0]-1], :] # last row, true Q-value given trained policy
        tuples = itertools.product([runID], [seed], [CPT_true.shape[0]-1])
        dfa.index = pd.MultiIndex.from_tuples(tuples)
        filename_ = filename + 'trueQPol.csv' #_of_trainedPolicy.csv'
        dfa.to_csv(filename_, mode='a', header=not os.path.exists(filename_))
        
        #policy_val = policy_val[-1] 
        dfa = policy_val.loc[idx[:, :, policy_val.shape[0]-1], :] # last row, trained policy
        tuples = itertools.product([runID], [seed], [policy_val.shape[0]-1])
        dfa.index = pd.MultiIndex.from_tuples(tuples)
        filename_ = filename + 'trainPol.csv'
        dfa.to_csv(filename_, mode='a', header=not os.path.exists(filename_))
        
        dfa = visit_freq.loc[idx[:, :, visit_freq.shape[0]-1], :] # last row, num visits during training
        tuples = itertools.product([runID], [seed], [visit_freq.shape[0]-1])
        dfa.index = pd.MultiIndex.from_tuples(tuples)
        filename_ = filename + 'visitFreq.csv' #uency_inTrain.csv'
        dfa.to_csv(filename_, mode='a', header=not os.path.exists(filename_))
        
    elif algo == 'SPSA':
        print('zidx_to_z placeholder as identity')
        zidx_to_z = lambda s : s
        
        cpt_true = dict()
        for x in model.CPT_true.keys():
            s, a = x
            s = zidx_to_z(s)
            t, z = s
            cpt_true[t, z, a] = np.array(model.CPT_true[x]).flatten()
        CPT_true = pd.DataFrame(cpt_true)
        iter_nums = range(len(CPT_true.index))
        tuples = itertools.product([runID], ['q_true'], iter_nums)
        CPT_true.index = pd.MultiIndex.from_tuples(tuples)
        
        policy_val = dict()
        for x in model.policy_val.keys():
            s, a = x
            s = zidx_to_z(s)
            t, z = s
            policy_val[t, z, a] = np.array(model.policy_val[x]).flatten()
        policy_val = pd.DataFrame(policy_val)
        
        tuples = itertools.product([runID], ['policy_i'], iter_nums)
        policy_val.index = pd.MultiIndex.from_tuples(tuples)
        policy_val = policy_val.sort_index(axis=1)
        
        visit_freq = dict()
        for x in model.visitFreq.keys():
            s, a = x
            #s = zidx_to_z(s)
            t, z = s
            visit_freq[t, z, a] = np.array(model.visitFreq[x]).flatten()
        visit_freq = pd.DataFrame(visit_freq)
        tuples = itertools.product([runID], ['visit_freq_total'], iter_nums) # no need iter_nums? q_true need cz depends on policy_i
        visit_freq.index = pd.MultiIndex.from_tuples(tuples)
        
        df = pd.concat([policy_val, CPT_true, visit_freq], axis=0, join = 'inner')
        df.to_csv(filename)
        
        # AGGR seeds
        if ablationID is not None:
            K, step_size, = ablationID
            if not dyn:
                filename = './{}/results/static/{}_aggr_{}_{}_{}_{}_{}_{}_'.format(env, algo, alpha, rho, lmbd, pwin, K, step_size)
            else:
                filename = './{}/results/dynamic/{}_aggr_{}_{}_{}_{}_{}_{}_'.format(env, algo, alpha, rho, lmbd, pwin, K, step_size)
        else:
            # set default: K = 50, step_size = 5.
            if not dyn:
                filename = './{}/results/static/{}_aggr_{}_{}_{}_{}_'.format(env, algo, alpha, rho, lmbd, pwin)
            else:
                filename = './{}/results/dynamic/{}_aggr_{}_{}_{}_{}_'.format(env, algo, alpha, rho, lmbd, pwin)
        
        idx = pd.IndexSlice 
        dfa = CPT_true.loc[idx[:, :, CPT_true.shape[0]-1], :] # last row, true Q-value given trained policy
        tuples = itertools.product([runID], [seed], [CPT_true.shape[0]-1])
        dfa.index = pd.MultiIndex.from_tuples(tuples)
        filename_ = filename + 'trueQ_of_trainedPolicy.csv'
        dfa.to_csv(filename_, mode='a', header=not os.path.exists(filename_))
        
        #policy_val = policy_val[-1] 
        dfa = policy_val.loc[idx[:, :, policy_val.shape[0]-1], :] # last row, trained policy
        tuples = itertools.product([runID], [seed], [policy_val.shape[0]-1])
        dfa.index = pd.MultiIndex.from_tuples(tuples)
        filename_ = filename + 'trainedPolicy.csv'
        dfa.to_csv(filename_, mode='a', header=not os.path.exists(filename_))
        
        dfa = visit_freq.loc[idx[:, :, visit_freq.shape[0]-1], :] # last row, num visits during training
        tuples = itertools.product([runID], [seed], [visit_freq.shape[0]-1])
        dfa.index = pd.MultiIndex.from_tuples(tuples)
        filename_ = filename + 'visitFrequency_inTrain.csv'
        dfa.to_csv(filename_, mode='a', header=not os.path.exists(filename_))
        
    elif algo == 'SPE':
        cpt_true = dict()
        for x in model.CPT_true_for_SPE.keys():
            s, a = x
            s = zidx_to_z(s)
            t, z = s
            cpt_true[t, z, a] = [model.CPT_true_for_SPE[x]]
        CPT_true = pd.DataFrame(cpt_true)
        iter_nums = range(len(CPT_true.index))
        tuples = itertools.product([runID], ['q_true'], iter_nums)
        CPT_true.index = pd.MultiIndex.from_tuples(tuples)
        CPT_true = CPT_true.sort_index(axis=1)
        
        policy_val = CPT_true.copy()
        for r in CPT_true.index:
            for c1, c2, _ in CPT_true.columns:
                c = (c1, c2)
                
                policy_val.loc[r, c] = np.zeros(len(CPT_true.loc[r, c]))
                max_id_ = np.argmax(CPT_true.loc[r, c])
                
                max_id, _ = model.trained_policy[c]
                max_id = np.argmax(max_id)
                
                print('state:', c)
                print('argmaxQ, pol:', max_id_, max_id)
                
                policy_val.loc[r, c][max_id] = 1
        tuples = itertools.product([runID], ['policy_i'], iter_nums)
        policy_val.index = pd.MultiIndex.from_tuples(tuples)
        policy_val = policy_val.sort_index(axis=1)
        
        # for discretized cont env (Shi)
        tuples = []
        for t, z, a in policy_val.columns:
            t, z = zidx_to_z((t, z))
            tuples += [(t, z, a)]
            
        policy_val.columns = pd.MultiIndex.from_tuples(tuples)
        
        df = pd.concat([CPT_true, policy_val], axis=0, join = 'inner')
        df.to_csv(filename)
        
        # AGGR
        algo = model.policy.name
        if not dyn:
            filename = './{}/results/static/{}_aggr_{}_{}_{}_{}_'.format(env, algo, alpha, rho, lmbd, pwin)
        else:
            filename = './{}/results/dynamic/{}_aggr_{}_{}_{}_{}_'.format(env, algo, alpha, rho, lmbd, pwin)
            
        # qTrue
        idx = pd.IndexSlice 
        dfa = CPT_true.loc[idx[:, :, CPT_true.shape[0]-1], :] # last row, true Q of true policy
        tuples = itertools.product([runID], [seed], [CPT_true.shape[0]-1])
        dfa.index = pd.MultiIndex.from_tuples(tuples)
        filename_ = filename + 'trueQ.csv'
        dfa.to_csv(filename_, mode='a', header=not os.path.exists(filename_))
        
        # policyTrue
        idx = pd.IndexSlice 
        dfa = policy_val.loc[idx[:, :, policy_val.shape[0]-1], :] # last row, true Q of true policy
        tuples = itertools.product([runID], [seed], [policy_val.shape[0]-1])
        dfa.index = pd.MultiIndex.from_tuples(tuples)
        filename_ = filename + 'truePolicy.csv'
        dfa.to_csv(filename_, mode='a', header=not os.path.exists(filename_))
        
    else: # Precomm (only for Barberis); SPERLeval, SPSAeval
        CPT_true = dict()
        for t in model.player_set:
            gain_supp = np.linspace(-model.eval_env.bet * t, model.eval_env.bet * t, t + 1)
            for z in gain_supp:
                state = (t, z)
                #state = zidx_to_z(state) # no need to implement on [Shi], remove zidx_to_z
                
                for action in range(model.eval_env.action_space.n):
                    # compute_CPT given sample returns (n = n_eval_eps)
                    # default sort=True
                    _, _, true_CPT_sa = model.evaluate_policy(init_sa = (state, action), 
                                                              deterministic = True, 
                                                              return_episode_rewards = False, 
                                                              for_critic = True)
                    
                    if (t, z, action) not in CPT_true.keys():
                        CPT_true[t, z, action] = [true_CPT_sa]
                    else:
                        print('cannot visit t,z,a twice!!')
        
        CPT_true = pd.DataFrame(CPT_true)
        iter_nums = range(len(CPT_true.index))
        tuples = itertools.product([runID], ['q_true'], iter_nums)
        CPT_true.index = pd.MultiIndex.from_tuples(tuples)
        CPT_true = CPT_true.sort_index(axis=1)
        
        policy_val = CPT_true.copy()
        for r in CPT_true.index:
            for c1, c2, _ in CPT_true.columns:
                c = (c1, c2) # t, z
                
                #policy_val.loc[r, c] = np.zeros(len(CPT_true.loc[r, c]))
                #max_id = np.argmax(CPT_true.loc[r, c])
                actProbs, _ = model.trained_policy[c]
                policy_val.loc[r, c] = actProbs #[max_id] = 1
        tuples = itertools.product([runID], ['policy_i'], iter_nums)
        policy_val.index = pd.MultiIndex.from_tuples(tuples)
        policy_val = policy_val.sort_index(axis=1)
        
        #tuples = []
        #for t, z, a in policy_val.columns:
        #    t, z = zidx_to_z((t, z))
        #    tuples += [(t, z, a)]
        #policy_val.columns = pd.MultiIndex.from_tuples(tuples)
        
        df = pd.concat([CPT_true, policy_val], axis=0, join = 'inner')
        df.to_csv(filename)
        
        # AGGR
        if not dyn:
            filename = './{}/results/static/{}_aggr_{}_{}_{}_{}_'.format(env, algo, alpha, rho, lmbd, pwin)
        else:
            filename = './{}/results/dynamic/{}_aggr_{}_{}_{}_{}_'.format(env, algo, alpha, rho, lmbd, pwin)
        
        # qTrue
        idx = pd.IndexSlice 
        dfa = CPT_true.loc[idx[:, :, CPT_true.shape[0]-1], :] # last row, true Q of true policy
        tuples = itertools.product([runID], [seed], [CPT_true.shape[0]-1])
        dfa.index = pd.MultiIndex.from_tuples(tuples)
        filename_ = filename + 'trueQ.csv'
        dfa.to_csv(filename_, mode='a', header=not os.path.exists(filename_))
        
        # policyTrue
        idx = pd.IndexSlice 
        dfa = policy_val.loc[idx[:, :, policy_val.shape[0]-1], :] # last row, true Q of true policy
        tuples = itertools.product([runID], [seed], [policy_val.shape[0]-1])
        dfa.index = pd.MultiIndex.from_tuples(tuples)
        filename_ = filename + 'truePolicy.csv'
        dfa.to_csv(filename_, mode='a', header=not os.path.exists(filename_))
        
    '''TO-DO:
    v 1. Add curVisitCount; edit visitFreq[key] to list
    v 2. Add model.visitFreq[key] to record_results(algo=All) as scalar, record_csv(algo=ALL) as TUPLE
    v 3. Edit policy_val in record_results(algo=ALL) as TUPLE
    
    v can use SPE_runID code to aggregate over seeds
    v copy for SPERL_aggr(CPT_true replace w CPT_val[-1]; add visit_freq[-1]), SPE_aggr; add input seed, rowIndex: seed
    
    v 4. Edit record_results(algo=Precomm); check record_results(Precomm) no bug
    
    v 5. Try rerunning SPERL, SPE, Precomm; compare w BFS20.
    v 6. Try read 'algo_aggr', plot SPE/a, etc.
    
    v 7. Check SPSA record_results, record_csv
    8. @Shi, edit compute_SPE policy.name = 'trueSPE', 
    
    '''

# EXPLORING SPE CONTROL ADVANTAGE
# Aim: Sweep CPTMDP parameters
'''
v 1. Set np.linspace CPT-MDP params in main.py, instead of checking Holder, run from param closer to Expectation
v 2. Add cptmdpID={}_{}_{}_{}.format(alpha, rho1, rho2, lmbd, p_win) to FILENAME (SPERL, SPSA, SPE, Precomm)
v 3. For each cptmdpID, run in series seed 0-9; run SPSA, SPERL, cptMdpID in parallel

v 4. Outside main.py, run record_csv.plotvsSPSA()

'''

from datetime import datetime

# SWEEP for errorTresh in [0, .2, .., 1., .., 1.54]
# compute_SPE(n_eval_eps = 2000, errorTresh=.., tieBreak=None)
# rename algo= SPE{n_eval}{err}{rule} # if rule==None, it is 'randomize'!!!

# find err* that match SPERL behavior!

'''
TO-DO:
1. ANALYZE .88, .65, .2.25, .72 FOR BETTER SPE-MATCH; NOW: BLACK >> RED!, NEED: BLUE = BLACK!
- explore rate tweaks (.3 effective is .14, too small!), do softmax too (tweak beta s.t. if distance Q0-Q1 close, choose w .5 prob)
- step size schedule balance across x, a given fixed t
- step size schedule balance across a given fixed t, x [since a seems matter more]

- QF recording n analysis
- lbub trials, visualize QF n CPTval
- if fails, filtering..?

2. SWEEP CPTMDP, FIND BLUE >> RED, DESPITE LOWER SPE-MATCH;
+ p<=.6 ok

3. SWEEP CPTMDP NEAR MEAN, FIND EVIDENCE OF SPE-MATCH GETS EASIER TO DONE IN THESE SETUPS (SUBSUMING DISTORTED MEAN)
+ cpt = (1, 1, 1) error is the lowest by far
- others..?
'''

'''# SPSA: pwin = 6 does not match BFS20?
for pwin in [.72]: #np.linspace(.72, .51, 8): #[.72, .6]: #[.6, .66, .54, .48, .72]:
    pwin = np.round(pwin, 4)
    env = 'barberis'
    cptMdpID = (.88, .65, 2.25, pwin)
    ablationID = (50, 5.)
    plotLearningCurves(env, cptMdpID, 'SPSA', ablationID = ablationID)
    # trainNum = 200, K=50, ss=5. --> no ablationID

# SPERL-IQN
for pwin in [.72]: #np.linspace(.72, .51, 8): #[.72, .6]: #[.6, .66, .54, .48, .72]:
    pwin = np.round(pwin, 4)
    env = 'barberisIQN'
    cptMdpID = (.88, .65, 2.25, pwin)
    ablationID = None
    plotLearningCurves(env, cptMdpID, 'SPERL', swingIdx = [True]*(2*15), ablationID = ablationID)
    # trainNum = 200, K=50, ss=5. --> no ablationID


'''

def plotLearningCurves(env, cptMdpID, algo, swingIdx = None, ablationID = None):

    alpha, rho, lmbd, pwin = cptMdpID
    lbub = 0
    smoothen, ss_inverted = 0, 0
    
    filename = './{}/results/static/{}_aggr_{}_{}_{}_{}_'.format(env, algo, alpha, rho, lmbd, pwin)
    
    if algo == 'SPERL':
        if ablationID is not None:
            K, lbub, smoothen, ss_inverted, eps = ablationID # K=100
            filename += '{}_{}_{}_{}_{}_'.format(K, lbub, smoothen, ss_inverted, eps)
        # else if in ['spsa']: += ''.format(nbatch)
        
        if ablationID is not None:
            filename_ = filename + 'trainPol.csv' #'trainedPolicy.csv' # 'trainPol.csv'
        else:
            filename_ = filename + 'trainedPolicy.csv' # 'trainPol.csv'
        
        filenameSPE = './{}/results/static/SPE_aggr_{}_{}_{}_{}_trueQ.csv'.format(env, alpha, rho, lmbd, pwin)
        trueSPEQ = pd.read_csv(filenameSPE, header = [0, 1, 2], index_col = [0, 1, 2])
        
        filenameSPE = './{}/results/static/SPE_aggr_{}_{}_{}_{}_truePolicy.csv'.format(env, alpha, rho, lmbd, pwin)
        trueSPEPol = pd.read_csv(filenameSPE, header = [0, 1, 2], index_col = [0, 1, 2])
        
    else:
        if ablationID is not None:
            K, step_size = ablationID
            filename_ = filename + '{}_{}_trainedPolicy.csv'.format(K, step_size)
        else:
            K, step_size = 50, 5.
            filename_ = filename + 'trainedPolicy.csv'
    
    #print(filename_)
    policy_val = pd.read_csv(filename_, header = [0, 1, 2], index_col = [0, 1, 2])
    #print(policy_val)
    idx = pd.IndexSlice
    

    Q0 = dict()
    Q1 = dict()
    Freq0 = dict()
    Freq1 = dict()
    trueQ0 = dict()
    trueQ1 = dict()
    V = []
    trueV = []
    
    for runID in policy_val.index.get_level_values(0):
        print('seed: ', runID)
        if len(str(runID)) < 14:
            runID = '0'+str(runID)
        
        filename = './{}/results/static/{}_{}.csv'.format(env, algo, runID)
        seedData = pd.read_csv(filename, header = [0, 1, 2], index_col = [0, 1, 2])
        
        if algo == 'SPERL':
            trainQ = seedData.loc[idx[:, ['q_i']], :]
            trainFreq = seedData.loc[idx[:, ['visit_freq_total']], :]
            
            colIDs = trainQ.columns[:2*15][swingIdx]
            colSPE = trueSPEQ.columns[:2*15][swingIdx]
            
            count = 0
            for c in colIDs: # t, x
                if count %2 == 0:
                    
                    c0, c1, c2 = c
                    label0 = (int(c0), int(c1), int(c2))
                    
                    q0 = np.array(trainQ.loc[:, c])
                    if label0 not in Q0.keys():
                        Q0[label0] = [list(q0)]
                    else:
                        Q0[label0] += [list(q0)]
                    
                    freq0 = np.array(trainFreq.loc[:, c])
                    if label0 not in Freq0.keys():
                        Freq0[label0] = [list(freq0)]
                    else:
                        Freq0[label0] += [list(freq0)]
                    
                    trueSPEArr = np.array(trueSPEQ.loc[:, colSPE[count]]) # dim (9, 1)
                    if label0 not in trueQ0.keys():
                        trueQ0[label0] = []
                        for elem in trueSPEArr:
                            trueQ0[label0] += [[elem for _ in range(len(q0))]]
                    
                    min_ = np.round(min(trueSPEArr) , 4)
                    max_ = np.round(max(trueSPEArr) , 4)
                    label0 = str(label0) + ': SPE= ' + str((min_, max_))
                    
                else:
                    c0, c1, c2 = c
                    label1 = (int(c0), int(c1), int(c2))
                    
                    q1 = np.array(trainQ.loc[:, c])
                    if label1 not in Q1.keys():
                        Q1[label1] = [list(q1)]
                    else:
                        Q1[label1] += [list(q1)]
                    
                    freq1 = np.array(trainFreq.loc[:, c])
                    if label1 not in Freq1.keys():
                        Freq1[label1] = [list(freq1)]
                    else:
                        Freq1[label1] += [list(freq1)]
                    
                    trueSPEArr = np.array(trueSPEQ.loc[:, colSPE[count]])
                    if label1 not in trueQ1.keys():
                        trueQ1[label1] = []
                        for elem in trueSPEArr:
                            trueQ1[label1] += [[elem for _ in range(len(q1))]]

                    min_ = np.round(min(trueSPEArr) , 4)
                    max_ = np.round(max(trueSPEArr) , 4)
                    label1 = str(label1) + ': SPE= ' + str((min_, max_))
                    
                    plt.figure()
                    plt.plot(q0, label = label0)
                    plt.plot(q1, label = label1)
                    plt.legend()
                    
                    title_ = 'TrainQ Dynamics'
                    title_ += '\n (alpha, delta, lmbd, pwin): ' + str((alpha, rho, lmbd, pwin))# + ', p_filter: ' + str(lbub)
                    title_ += ', \n (p_filter, treshRatio, firstVisit): ' + str((lbub, smoothen, ss_inverted))
                    plt.title(title_, fontsize=10)
                    plt.tight_layout()
                    plt.savefig("./{}/plots/static/{}trainQ_{}_{}_{}_{}_{}_{}_{}_{}{}_{}.png".format(env, algo, alpha, rho, lmbd, pwin, lbub, smoothen, ss_inverted, c0, c1, runID), dpi=100)
                    plt.close()
                    
                count += 1
        
        trueQpol = seedData.loc[idx[:, ['q_true']], :]
        trainPol = seedData.loc[idx[:, ['policy_i']], :]
        q_dat = trueQpol.loc[:, trueQpol.columns[:2]]
        pol_dat = trainPol.loc[:, trainPol.columns[:2]]
        v_dat = np.multiply(q_dat, pol_dat)
        
        v_arr = np.array(v_dat)
        v_arr = np.sum(v_arr, axis = 1)
        
        V += [list(v_arr)]
        
        SPEq_dat = trueSPEQ.loc[:, trueSPEQ.columns[:2]]
        SPEpol_dat = trueSPEPol.loc[:, trueSPEPol.columns[:2]]
        SPEv_dat = np.multiply(SPEq_dat, SPEpol_dat) # 9
        SPEv_arr = np.array(SPEv_dat)
        SPEv_arr = np.sum(SPEv_arr, axis=1)
        
        if len(trueV) == 0:
            for elem in SPEv_arr:
                trueV += [[elem for _ in range(len(v_arr))]]
        
        plt.figure()
        plt.plot(v_arr)
        title_ = 'V0 Dynamics'
        if algo == 'SPERL':
            title_ += '\n (alpha, delta, lmbd, pwin): ' + str((alpha, rho, lmbd, pwin))# + ', p_filter: ' + str(lbub)
            title_ += ', \n (p_filter, treshRatio, firstVisit): ' + str((lbub, smoothen, ss_inverted))
        else:
            title_ += '\n (alpha, delta, lmbd, pwin): ' + str((alpha, rho, lmbd, pwin)) + ', stepSize: ' + str(step_size)
        plt.title(title_, fontsize=10)
        plt.tight_layout()
        
        if algo == 'SPERL':
            plt.savefig("./{}/plots/static/{}V0_{}_{}_{}_{}_{}_{}_{}_{}.png".format(env, algo, alpha, rho, lmbd, pwin, lbub, smoothen, ss_inverted, runID), dpi=200)
        else:
            plt.savefig("./{}/plots/static/{}V0_{}_{}_{}_{}_{}_{}.png".format(env, algo, alpha, rho, lmbd, pwin, step_size, runID), dpi=200)
        plt.close()
    
    for tsa in Q0.keys():
        t, s, a = tsa
        q0_seeds = np.array(Q0[tsa]) # dim: (9, 300)
        q0_avg = np.average(q0_seeds, axis=0) # dim: (1, 300)
        q0_stdev = np.std(q0_seeds, axis=0)

        q1_seeds = np.array(Q1[(t, s, 1)])
        q1_avg = np.average(q1_seeds, axis=0)
        q1_stdev = np.std(q1_seeds, axis=0)
        
        trueq0_seeds = np.array(trueQ0[tsa])
        trueq0_avg = np.average(trueq0_seeds, axis=0)
        #trueq0_stdev = np.std(trueq0_seeds, axis=0)
        
        trueq1_seeds = np.array(trueQ1[(t, s, 1)])
        trueq1_avg = np.average(trueq1_seeds, axis=0)
        #trueq1_stdev = np.std(trueq1_seeds, axis=0)
        
        iters = [i for i in range(len(q0_avg))]
        
        plt.figure()
        plt.plot(iters, q0_avg, label = 'SPERL Q(\'exit\')')
        plt.fill_between(iters, q0_avg - q0_stdev, q0_avg + q0_stdev, alpha = .2)
        
        plt.plot(iters, q1_avg, label = 'SPERL Q(\'gamble\')')
        plt.fill_between(iters, q1_avg - q1_stdev, q1_avg + q1_stdev, alpha = .2)
        
        plt.plot(iters, trueq0_avg, color='tab:blue', linestyle = '--', label = 'SPE Q(\'exit\')')
        #plt.fill_between(iters, trueq0_avg - trueq0_stdev, trueq0_avg + trueq0_stdev, alpha = .2)
        
        plt.plot(iters, trueq1_avg, color='tab:orange', linestyle = '--', label = 'SPE Q(\'gamble\')')
        #plt.fill_between(iters, trueq1_avg - trueq1_stdev, trueq1_avg + trueq1_stdev, alpha = .2)
        
        plt.legend()
        plt.xlim([0, len(q0_avg)])
        plt.tight_layout()
        plt.savefig("./{}/plots/static/{}avgQ_{}_{}_{}_{}_{}_{}_{}_{}{}.png".format(env, algo, alpha, rho, lmbd, pwin, lbub, smoothen, ss_inverted, t, s), dpi=100)
        plt.close()
    
        plt.figure()
        freq0_seeds = np.array(Freq0[tsa])
        freq1_seeds = np.array(Freq1[(t, s, 1)])

        for i in range(9):
            plt.plot(iters, freq0_seeds[i, :], color='tab:blue')
            plt.plot(iters, freq1_seeds[i, :], color='tab:orange')
        
        plt.tight_layout()
        plt.savefig("./{}/plots/static/{}allFreq_{}_{}_{}_{}_{}_{}_{}_{}{}.png".format(env, algo, alpha, rho, lmbd, pwin, lbub, smoothen, ss_inverted, t, s), dpi=100)
        plt.close()
    
    V_seeds = np.array(V)
    V_avg = np.average(V_seeds, axis=0)
    V_stdev = np.std(V_seeds, axis=0)
    
    trueV_seeds = np.array(trueV)
    trueV_avg = np.average(trueV_seeds, axis=0)
    #trueV_stdev = np.std(trueV_seeds, axis=0)
    
    iters = [i for i in range(len(V_avg))]
    
    plt.figure()
    plt.plot(iters, V_avg, label = 'SPERL V0')
    plt.fill_between(iters, V_avg - V_stdev, V_avg + V_stdev, alpha = .2)
    
    plt.plot(iters, trueV_avg, color='tab:blue', linestyle='--', label = 'SPE V0')
    #plt.fill_between(iters, trueV_avg - trueV_stdev, trueV_avg + trueV_stdev, color='grey', alpha = .2)
    
    plt.legend()
    plt.xlim([0, len(V_avg)])
    plt.tight_layout()
    plt.savefig("./{}/plots/static/{}avgV0_{}_{}_{}_{}_{}_{}_{}.png".format(env, algo, alpha, rho, lmbd, pwin, lbub, smoothen, ss_inverted), dpi=200)
    plt.close()
    
'''
for pwin in [.6]: # [.48, .54, .6, .66, .72]:
    #np.linspace(.72, .48, 9): 
    pwin = np.round(pwin, 4)
    
    for p_filter in [.8]: #[1., .8, .75]: # [.99]:
        treshRatio = 0.5
        env = 'barberis'
        nActs = 2
        cptMdpID = (.95, .5, 1.5, pwin) #(.88, .65, 2.25, pwin) #(1.0, 1.0, 1.0, pwin)  (.95, .5, 1.5, pwin)
        
        ablationID = [50, 1*p_filter, treshRatio, 1, 0.6] #[50, 0, 0, 0, 0.6], None # SPE Alignment ERROR appears after ablation
        SPSAablation = None # coNone (trainNum=200), (50, 5.)
        SPEablation = None # None (tB=0, no tresh), 0.5
        
        algonames = ['SPERL', 'SPE', 'SPSA', 'Precomm'] #  ['SPERL', 'SPE', 'SPSA', 'Precomm']  ['SPERL', 'SPE', 'Precomm']
        
        swingIdx = plotvsSPSA(env, nActs, cptMdpID, yAxes = 'policy', algonames = algonames, ablationID = ablationID, 
                              SPSAablation = SPSAablation, SPEablation = SPEablation)
        #plotvsSPSA(env, nActs, cptMdpID, yAxes = 'value', algonames = algonames, ablationID = ablationID, 
        #           SPSAablation = SPSAablation, SPEablation = SPEablation)

        swingIdx[0] = True
        swingIdx[1] = True
        #swingIdx = [True]*(2*15)
        plotLearningCurves(env, cptMdpID, 'SPERL', swingIdx=swingIdx, ablationID = ablationID)
        

# HALT RUNS FOR FILTER WHEN THERE ARE NOT MUCH DIFFERENCE .75, .95, .8!!!
'''        

def plotvsSPSA(env, nActs, cptMdpID, yAxes = 'value', algonames = ['SPERL', 'SPE', 'SPSA', 'Precomm'], 
               ablationID = None, SPSAablation = None, SPEablation = None, isSPERL = True):
    # BFS20 is early version for Barberis
    
    alpha, rho, lmbd, pwin = cptMdpID
    lbub = 0
    smoothen, ss_inverted = 0, 0
    
    envSPERL = env
    
    all_means = []
    all_stdevs = []
    max_stdev = []
    all_ubs = []
    all_lbs = []
    all_tx = None
    rows, cols = None, None
    #rows_ID = None
    
    sumValtx_std = []
    V0_std = []
    
    for algo in algonames:
        print('algo:', algo)
        
        if algo == 'SPERL':
            env = envSPERL
        else:
            env = 'barberis'
            
        filename = './{}/results/static/{}_aggr_{}_{}_{}_{}_'.format(env, algo, alpha, rho, lmbd, pwin)
        
        if algo in ['SPE'] and SPEablation is not None:
            tBtresh = SPEablation
            filename = './{}/results/static/{}{}_aggr_{}_{}_{}_{}_'.format(env, algo, tBtresh, alpha, rho, lmbd, pwin)

        if algo in ['SPERL', 'SPERL300.2'] and ablationID is not None:
            K, lbub, smoothen, ss_inverted, eps = ablationID # K=100
            filename += '{}_{}_{}_{}_{}_'.format(K, lbub, smoothen, ss_inverted, eps)
        
        if algo in ['SPSA']:
            if SPSAablation is not None:
                K, step_size = SPSAablation
                filename += '{}_{}_'.format(K, step_size)
            
        if algo in ['SPE', 'Precomm', 'GainExit']:
            filename_ = filename + 'trueQ.csv'
        elif algo in ['SPERL'] and ablationID is not None:
            filename_ = filename + 'trueQPol.csv' #'trueQ_of_trainedPolicy.csv' # 'trueQPol.csv'
        else:
            filename_ = filename + 'trueQ_of_trainedPolicy.csv' # 'trueQPol.csv'
        
        print(filename_)
        CPT_true = pd.read_csv(filename_, header = [0, 1, 2], index_col = [0, 1, 2])

        if algo in ['SPE', 'Precomm', 'GainExit']:
            filename_ = filename + 'truePolicy.csv'
        elif algo in ['SPERL'] and ablationID is not None:
            filename_ = filename + 'trainPol.csv' #'trainedPolicy.csv' # 'trainPol.csv'
        else:
            filename_ = filename + 'trainedPolicy.csv' # 'trainPol.csv'
        policy_val = pd.read_csv(filename_, header = [0, 1, 2], index_col = [0, 1, 2])
        
        if rows is not None:
            prev_rows, prev_cols = rows, cols
            #prev_rowsID = rows_ID
            rows, cols = CPT_true.shape
            
            if rows != prev_rows or cols != prev_cols:
                print('Unmatched rows or cols!')
                print(rows, cols, prev_rows, prev_cols)
                raise ValueError
                # if have time, implement rowID filter
                
        rows, cols = CPT_true.shape
        rowsID = CPT_true.index.get_level_values(1)
        
        CPT_true_V = []
        idx = pd.IndexSlice 
        for r in rowsID: #range(rows):
            if yAxes == 'value':
                q_dat = CPT_true.loc[idx[:, [r]], :]
            else:
                q_dat = [0, 1]*(cols//nActs)
                
            pol_dat = policy_val.loc[idx[:, [r]], :]
            
            v_dat = np.multiply(q_dat, pol_dat)
            v_arr = np.array(v_dat).transpose()
            #print(v_arr)
            v_arr = np.reshape(v_arr, (cols//nActs, nActs)).transpose() #shape: (2, 21)
            #print(v_arr)
            v_arr = np.sum(v_arr, axis = 0) #shape: (1, 21)
            #print(v_arr)
            
            CPT_true_V += [list(v_arr)]
            
        CPT_true_V = np.array(CPT_true_V)
        print(CPT_true_V.shape)
        
        # AGGR SEED,
        mean = np.average(CPT_true_V, axis = 0)
        stdev = np.std(CPT_true_V, axis = 0)
        maxmingap = np.max(CPT_true_V, axis = 0) - np.min(CPT_true_V, axis = 0)
        #print(CPT_true_V)
        #print(mean)
        #print(stdev)
        #print(maxmingap) #minV, maxV across rowIDs
        
        N = CPT_true_V.shape[0]
        ub = mean + 2*stdev/(N**.5)
        lb = mean - 2*stdev/(N**.5)
        #print(ub)
        #print(lb)
        max_stdev += [np.round(max(maxmingap), 4)] #[np.round(4*max(stdev)/(N**.5), 4)] # consistent tie-break 'value': 
        
        # APPEND TO ARR
        all_means += [list(mean)]
        all_stdevs += [list(stdev)]
        all_ubs += [list(ub)]
        all_lbs += [list(lb)]
        
        if yAxes == 'value' and isSPERL == False:
            sumValtx_std += [np.round(np.std(np.sum(CPT_true_V[:, :15], axis = 1)), 4)] # stdev of 9 elems, after sum over 15 tx
        else: #if yAxes == 'value' and isSPERL == -1 or yAxes == 'policy'
            sumValtx_std += [np.round(np.average(np.std(CPT_true_V[:, :15], axis = 0)), 4)] # stdev of 9 elems, then sum over 15 tx
        
        ##### WE ONLY CARE ABOUT FIRST 15 ELEMENTS!!
        print('sumValtx_std 15 elems?', CPT_true_V[:, :15].shape)
        
        V0_std += [np.round(np.std(CPT_true_V[:, 0]), 4)]
        
        all_tx_ = []
        for colID in list(CPT_true.columns):
            c1, c2, c3 = colID
            
            if int(float(c3)) == 0:
                #print(c1, c2)
                all_tx_ += [algo + str((int(float(c1)), int(float(c2))))]
                
        if all_tx is None:            
            all_tx = np.array(all_tx_).reshape((1, -1))
        else:
            all_tx_ = np.array(all_tx_).reshape((1, -1))
            all_tx = np.concatenate((all_tx, #np.array(['']*(cols//nActs)).reshape(1, -1)
                                     all_tx_), axis = 0)
            
    # ARR SHAPE = (4, |S|)
    
    # COMPUTE METRICS: SPERL - SPE
    #if not isSPERL:
    sumValtx = []
    sumValtx_avg = []
    avgValtx = np.sum(all_means, axis = 0)
    for i in range(len(algonames)):
        sumValtx += [np.round(sum(np.array(all_means)[i, :15]), 4)]
        
        # WE ONLY CARE ABOUT FIRST 15 ELEMS
        print('sumValtx 15 elems?', np.array(all_means)[i, :15].shape)
        
        sumValtx_avg += [ np.round(sum(np.array(all_means)[i, :] - avgValtx), 4)]
        
        '''
        valPolSPSA = np.round(sum(np.array(all_means)[1, :]), 4)
        valPolPrecomm = np.round(sum(np.array(all_means)[2, :]), 4)
        valPolGE = np.round(sum(np.array(all_means)[3, :]), 4)
        '''
    #sumValtx = [valPolSPE, valPolSPSA, valPolPrecomm, valPolGE]
    
    valPolDiff = np.array(all_means)[0, :] - np.array(all_means)[1, :] # dim: (1, |S|)
    valPolDiff = valPolDiff[:len(valPolDiff) - 6]
    # plt.plot(valPolDiff)
    
    if len(algonames) > 2:
        valPolDiffSPSA = np.array(all_means)[2, :] - np.array(all_means)[1, :] 
        valPolDiffSPSA = valPolDiffSPSA[:len(valPolDiffSPSA) - 6]
    else:
        valPolDiffSPSA = np.array([0]*len(valPolDiff))
    ###########################################################################
    
    swingIdx = np.abs(valPolDiff) > 1e-6
    swingIdx = swingIdx.reshape(1, -1)
    swingIdx = np.concatenate((swingIdx, swingIdx), axis = 0)
    swingIdx = swingIdx.transpose().flatten()
    # output by function
    
    ############################################################################
    
    all_tx = all_tx.transpose().flatten() # X_id
    all_means = np.array(all_means).transpose().flatten() # dim: (4,|S|) -> (|S|, 4) -> 4*|S|
    all_stdevs = np.array(all_stdevs).transpose().flatten()
    all_ubs = np.array(all_ubs).transpose().flatten()
    all_lbs = np.array(all_lbs).transpose().flatten()
    
    all_tx = all_tx[:len(all_tx) - len(algonames)*6]
    all_means = all_means[:len(all_means) - len(algonames)*6]
    all_stdevs = all_stdevs[:len(all_stdevs) - len(algonames)*6]
    all_ubs = all_ubs[:len(all_ubs) - len(algonames)*6]
    all_lbs = all_lbs[:len(all_lbs) - len(algonames)*6]
    
    # ERROR BAR: PLOT(X = t,x; Y = mean \pm stdev)
    lower_error = all_means - all_lbs
    upper_error = all_ubs - all_means
    #print(max(upper_error), min(lower_error))
    asymmetric_error = [lower_error, upper_error]
    
    colors = {'SPERL':'blue', 'SPERL300.2': 'blue', 'SPE':'black', 'SPSA': 'red', 'Precomm':'green', 'GainExit': 'orange'}
    color = []
    for algo in algonames:
        color += [colors[algo]]
    color = color*(cols//nActs - 6)
    
    plt.figure(figsize = (16, 4)) # (15, 3)
    plt.scatter(all_tx, all_means, c = color, alpha = .6)
    eb1 = plt.errorbar(all_tx, all_means, yerr=asymmetric_error, fmt = ' ', ecolor=color, linewidth = .6)
    eb1[-1][0].set_linestyle('--')
    
    plt.scatter(all_tx, all_lbs, marker = '_', s = 80, color = color)
    plt.scatter(all_tx, all_ubs, marker = '_', s = 80, color = color)
    
    plt.xticks(rotation=90, fontsize=6)
    #plt.ylabel(r'$V^\pi_{\mathcal{C}}(t, x)$', fontsize=8)
    plt.ylabel(yAxes, fontsize=8)
    plt.xlabel('Eval (t, x)', fontsize=8)
    
    '''
    plt.axvspan(.5, 1 + .5, facecolor = 'gray', alpha = .1)
    plt.axvspan(2 + .5, 4 + .5, facecolor = 'gray', alpha = .1)
    plt.axvspan(5 + .5, 7 + .5, facecolor = 'gray', alpha = .1)
    plt.axvspan(9 + .5, 12 + .5, facecolor = 'gray', alpha = .1)
    '''
    
    '''
    if not isSPERL:
        title_ = '(alpha, delta, lmbd, pwin): ' + str((alpha, rho, lmbd, pwin))
        #title_ += ' | (p_filter, treshRatio, firstVisit): ' + str((lbub, smoothen, ss_inverted)) + ' | SPE tB: ' + str(SPEablation)
        
        if yAxes == 'value':
            title_ += '\n sum Vtx: ' + str(sumValtx)
            title_ += '\n V0 mean: ' + str(np.round(all_means[:len(algonames)], 4))# + ', V0 max-min: ' + str(np.round((all_ubs - all_lbs)[:len(algonames)], 4))
        plt.title(title_, fontsize=10)
        plt.tight_layout()
    else:'''
    
    if yAxes == 'value':
        title_ = '(alpha, delta, lmbd, pwin): ' + str((alpha, rho, lmbd, pwin))
        #title_ += '\n largest Vgap: ' + str(max_stdev)
        title_ += ' | (p_filter, treshRatio, firstVisit): ' + str((lbub, smoothen, ss_inverted)) + ' | SPE tB: ' + str(SPEablation)
        title_ += '\n sum Vtx: ' + str(sumValtx) + ', sum Vtx-avged: ' + str(sumValtx_avg)
        title_ += '\n V0 mean: ' + str(np.round(all_means[:len(algonames)], 4))
        # + ', V0 max-min: ' + str(np.round((all_ubs - all_lbs)[:len(algonames)], 4))

        title_ += '\n sum Vtx-diff*: ' + str(np.round([sum(valPolDiff), sum(valPolDiffSPSA)], 4)) + ', sum abs-Vtx-diff: ' + str(np.round([sum(abs(valPolDiff)), sum(abs(valPolDiffSPSA))], 4))
        #title_ += '\n V0 mean: ' + str(np.round(all_means[:len(algonames)], 4)) + ', V0 max-min: ' + str(np.round((all_ubs - all_lbs)[:len(algonames)], 4))
        # large Vgap: find (t,x) source -> see policy(t',x') after t,x -> see if caused by inconsistent tie-break
    else:
        title_ = '(alpha, delta, lmbd, pwin): ' + str((alpha, rho, lmbd, pwin))
        title_ += ' | (p_filter, treshRatio, firstVisit): ' + str((lbub, smoothen, ss_inverted)) +  ' | SPE tB: ' + str(SPEablation)
        title_ += '\n sum poltx-diff: ' + str(np.round([sum(valPolDiff), sum(valPolDiffSPSA)], 4)) + ', sum abs-poltx-diff*: ' + str(np.round([sum(abs(valPolDiff)), sum(abs(valPolDiffSPSA))], 4))
    plt.title(title_, fontsize=10)
    plt.tight_layout()
    
    env = envSPERL
    
    if SPEablation is not None:
        if yAxes == 'value':
            plt.savefig("./{}/plots/static/SPERLCompareVal_{}_{}_{}_{}_{}_{}_{}_{}.png".format(env, alpha, rho, lmbd, pwin, lbub, smoothen, ss_inverted, tBtresh), dpi=200)    
        else:
            plt.savefig("./{}/plots/static/SPERLComparePol_{}_{}_{}_{}_{}_{}_{}_{}.png".format(env, alpha, rho, lmbd, pwin, lbub, smoothen, ss_inverted, tBtresh), dpi=200)
        
    else:
        
        if yAxes == 'value':
            plt.savefig("./{}/plots/static/SPERLCompareVal_{}_{}_{}_{}_{}_{}_{}.png".format(env, alpha, rho, lmbd, pwin, lbub, smoothen, ss_inverted), dpi=200)    
        else:
            plt.savefig("./{}/plots/static/SPERLComparePol_{}_{}_{}_{}_{}_{}_{}.png".format(env, alpha, rho, lmbd, pwin, lbub, smoothen, ss_inverted), dpi=200)
    plt.close()
    
    
    if isSPERL is not True: #not isSPERL:
        return np.round(all_means[:len(algonames)], 4), sumValtx, sumValtx_avg, sumValtx_std, V0_std
    
    return swingIdx

'''

plt.figure(figsize=(5, 5))
plt.scatter(X, Y)

# SPE-SPSA-LossExit-GainExit (Frontier)

for pwin in [.59, .62, .64]: #[.59, .57, .61, .63, .62, .64]: #[.42, .36, .3]: #[.72, .48, .66, .54, .6]: #
    pwin = np.round(pwin, 4)
    
    for p_filter in [1.]: #[1.]: #[.75, .95, .8, .9, .85]:
        treshRatio = 0 #0 #0.5
        env = 'barberis'
        nActs = 2
        cptMdpID =(.88, .65, 2.25, pwin) #(1.0, 1.0, 1.0, pwin) (.88, .65, 2.25, pwin) (.95, .5, 1.5, pwin)
        
        ablationID = [50, 1*p_filter, treshRatio, 1, 0.6] #[50, 1*p_filter, treshRatio, 1, 0.6] #[50, 0, 0, 0, 0.6], None # SPE Alignment ERROR appears after ablation
        SPSAablation = None #(trainNum=200), (50, 5.)
        SPEablation = None # None (tB=0, no tresh), 0.5
        
        algonames = ['SPERL', 'SPE', 'SPSA', 'Precomm', 'GainExit'] #['SPERL', 'SPE', 'Precomm']   ['SPERL', 'SPE', 'SPSA', 'Precomm']  ['SPERL', 'SPE', 'Precomm']
        
        yAxes = 'value' # 'value', 'policy'
        isSPERL = False # False: Frontier (sum first, then stdev); -1: FrPol, FrVal (stdev first, then sum)
        #VplotvsSPSA(env, nActs, cptMdpID, yAxes = 'policy', algonames = algonames, ablationID = ablationID, 
        #           SPSAablation = SPSAablation, SPEablation = SPEablation, isSPERL = False)
        # ADD sumPoltx_std
        # ADD frontier figure for POLICY
        # plt.savefig("./{}/plots/static/FrPol_{}_{}_{}_{}_{}_{}.png".format(env, alpha, rho, lmbd, pwin, p_filter, treshRatio), dpi=100)    
        
        V0, sumValtx, sumValtx_avg, sumValtx_std, V0_std = plotvsSPSA(env, nActs, cptMdpID, yAxes = yAxes, algonames = algonames, ablationID = ablationID, 
                                  SPSAablation = SPSAablation, SPEablation = SPEablation, isSPERL = isSPERL)

        colors = {'SPERL':'blue', 'SPERL300.2': 'blue', 'SPE':'black', 'SPSA': 'red', 'Precomm':'green', 'GainExit': 'orange'}
        color = []
        for algo in algonames:
            color += [colors[algo]]

        # SUM
        plt.figure(figsize = (10, 10)) # (15, 3)
        #plt.figure()
        plt.scatter(V0, sumValtx, color = color)
        for i in range(len(algonames)):
            if algonames[i] == 'Precomm':
                plt.text(V0[i], sumValtx[i], s = 'LossExit' + '\n ' + str((V0_std[i], sumValtx_std[i])))
            #elif algonames[i] == 'SPERL':
            #    plt.text(V0[i], sumValtx[i]-2, s = algonames[i] + '\n ' + str((V0_std[i], sumValtx_std[i])))
            else:    
                plt.text(V0[i], sumValtx[i], s = algonames[i] + '\n ' + str((V0_std[i], sumValtx_std[i])))
        
        x_adj = (max(V0) - min(V0))/10
        y_adj = (max(sumValtx) - min(sumValtx))/10
        plt.xlim([min(V0)-x_adj, max(V0)+x_adj*3])
        plt.ylim([min(sumValtx)-y_adj, max(sumValtx)+y_adj])
        
        plt.ylabel('sum Vtx')
        plt.xlabel('V0')
        
        title_ = 'SPE vs SPSA vs LossExit vs GainExit'
        title_ += '\n (alpha, delta, lmbd, pwin): ' + str(cptMdpID)
        title_ += '\n sum Vtx: ' + str(sumValtx) + ', std: ' + str(sumValtx_std)
        title_ += '\n V0: ' + str(V0) + ', std: ' + str(V0_std)
        
        plt.title(title_)
        #plt.tight_layout()
        alpha, rho, lmbd, pwin = cptMdpID
        
        if yAxes == 'value' and isSPERL == False:
            plt.savefig("./{}/plots/static/Frontier_{}_{}_{}_{}_{}_{}.png".format(env, alpha, rho, lmbd, pwin, p_filter, treshRatio), dpi=100)    
        elif yAxes == 'value':
            plt.savefig("./{}/plots/static/FrVal_{}_{}_{}_{}_{}_{}.png".format(env, alpha, rho, lmbd, pwin, p_filter, treshRatio), dpi=100)    
        else:
            plt.savefig("./{}/plots/static/FrPol_{}_{}_{}_{}_{}_{}.png".format(env, alpha, rho, lmbd, pwin, p_filter, treshRatio), dpi=100)    
            
        plt.close()

        ###############
        # SUM - MEAN
        plt.figure(figsize = (7, 7)) # (15, 3)
        #plt.figure()
        plt.scatter(V0, sumValtx_avg, color = color)
        for i in range(len(algonames)):
            plt.text(V0[i], sumValtx_avg[i], s = algonames[i])
        
        x_adj = (max(V0) - min(V0))/10
        y_adj = (max(sumValtx_avg) - min(sumValtx_avg))/10
        plt.xlim([min(V0)-x_adj, max(V0)+x_adj])
        plt.ylim([min(sumValtx_avg)-y_adj, max(sumValtx_avg)+y_adj])
        
        plt.ylabel('sum Vtx')
        plt.xlabel('V0')
        
        title_ = 'SPE vs SPSA vs Precomm vs GainExit'
        title_ += '\n (alpha, delta, lmbd, pwin): ' + str(cptMdpID)
        title_ += '\n sum Vtx_avg: ' + str(sumValtx_avg)
        title_ += '\n V0: ' + str(V0)
        
        plt.title(title_)
        #plt.tight_layout()
        alpha, rho, lmbd, pwin = cptMdpID
        plt.savefig("./{}/plots/static/Frontier_{}_{}_{}_{}_{}_{}_avg.png".format(env, alpha, rho, lmbd, pwin, p_filter, treshRatio), dpi=100)    
        plt.close()


'''
        
###############################################################################
# LEARNING PERFORMANCE: SPE MATCHING
# SPETree replaced with Q-curves () in .xls (t=4 low, t=4 high) tabs
# CError can remove, t=4 high tabs sufficiently illustrate
# SPEa, SPEu replaced with SPERLtrainQ vs SPERLtrueQ | visitFreq | SPERLtrainPolicy vs SPEtruePolicy
def plotFrequency(env, dyn = False, plotting = True):
    
    #from os import listdir
    #from os.path import isfile, join
    
    #path = './{}/results/static/'.format(env)
    #filenames = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".csv")]
    
    env = 'barberis'
    algo = 'SPERL'
    
    if not dyn:
        filename = './{}/results/static/{}_aggr_'.format(env, algo)
    else:
        filename = './{}/results/dynamic/{}_aggr_'.format(env, algo)
        
    filename_ = filename + 'visitFrequency_inTrain.csv'
    
    # read file
    
    
    return

def plotTree():
    # use trainQ_SPERL_aggr.xls
    
    algo = 'SPERL'
    
    
    return

def plotCError():
    
    algo = 'QR-PE'
    # add .csv (with fixed policy)
    
    # read [q_i, q_true]
    predicted_C = []
    true_C = []
    
    return
    

# env = 'shiBM', 'barberis'
def aggregate(env, dyn = False, plotting = False):
    from os import listdir
    from os.path import isfile, join
    
    print('Remember to check Env-params->feat_range_z,\n SPERL columns (lvl=1) cannot be merged for diff feat_range_z!')
    
    path = './{}/results/static/'.format(env)
    filenames = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".csv")]

    df_SPERL = pd.DataFrame()
    df_SPE = pd.DataFrame()
    
    for i in range(len(filenames)):
        filename = filenames[i]
        print(filename)
        
        algo = ''
        for char in filename:
            if char == '_':
                break
            
            algo += char
            
        if algo == 'SPERL':
            
            df = pd.read_csv(path + filename, header = [0, 1, 2], index_col = [0, 1, 2])
            df_SPERL = pd.concat([df_SPERL, df], axis = 0)
            # join outer! cz SPERL (t, x, u) visited randomly, columns not equal
            # across runID
            
        elif algo == 'SPE':
            df = pd.read_csv(path + filename, header = [0, 1, 2], index_col = [0, 1, 2])
            df_SPE = pd.concat([df_SPE, df], axis = 0)
            
            # join inner/outer?
            # should be the same, cz SPE (t, x, u) with the same parameters
            # are visited deterministically.
            
            # given runID, get_params(algo + runID.txt)
            # append param columns/index only if necessary
            
        elif algo == 'SPSA':
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        
        #for c in df.columns:
        #    print(c)
        #print(set(df.columns.get_level_values(1)) ) 
        print('---')
        
    '''Implement from CRITIC_EVO_CSV:
        1. learning_curves, each runID, aggregate over runs (w mean/stdev)
        2. policy_final (rowID: q_i)
        3. utility_final (rowID: q_TRUE, pi_i)
        4. policy_init (rowID: q_i)
        5. utility_init (rowID: q_TRUE, pi_i)
        
    Note: 3. & 5. does not apply for SPSA
    -> record pi_EST (softmax(theta)) during training (faster than V_EST)'''
        
    print('Check t,x,u visits SPERL >= SPE:', list(df_SPERL.shape)[1] , list(df_SPE.shape)[1])
    
    ######################## Q(t, x, u)-learning curves #######################
    idx = pd.IndexSlice 
    dfa = df_SPERL.loc[idx[:, ['q_i']], :]
    dfb = df_SPE.loc[idx[:, ['q_true']], :]
    print('SPERL q_i > SPE q_true?', dfa.shape, dfb.shape)
    dfab = pd.concat([dfa, dfb], axis = 0, join = "inner") # dfa's q_i should have ALL t,x
    
    #_, all_txu = dfab.shape
    #fig, axs = plt.subplots(all_txu//5, 5)
    color = ['black', 'blue', 'orange', 'pink', 'green', 'red']
    prev_s, prev_t, prev_z = None, None, None
    fig, axs = plt.subplots(1, 6, sharey=True, figsize = (48, 24))
    
    for t,z,u in dfab.columns:
        print(t, z, u)
        if t != '2':
            continue
        
        s = (t, z)
        
        if prev_s is not None and s!=prev_s:
            fig.suptitle('CPT-est vs true: s=({}, {})'.format(prev_t, prev_z), fontsize=30)
            fig.subplots_adjust(hspace = .4)
            plt.savefig("./shiBM/Bug3/CPTPred_aggre_({}, {}).png".format(prev_t, prev_z), dpi = 100)
            plt.close(fig)
        
        if s!= prev_s:
            fig, axs = plt.subplots(1, 6, sharey=True, figsize = (48, 24))
        
        #plt.figure()
        for runID in set(dfab.index.get_level_values(0)):
            q_i = np.array(dfab.loc[idx[[runID], :, :], (t, z, u)])
            q_i = list(q_i)
            
            if len(q_i) == 1:
                q_i = q_i*101
                axs[int(u)].plot(q_i, color = color[int(u)], linestyle = 'dashed', label = 'a='+str(int(u))+' , y='+str(q_i[0]))
                #plt.plot(q_i, color = 'black', label = str((t,z,u)))
                continue
            
            axs[int(u)].plot(q_i, color = color[int(u)], alpha = len(set(q_i))/101, label = 'visit-freq:'+str(len(set(q_i))))
            #plt.plot(q_i, #color = 'grey', 
            #         label = 'visit_freq: ' + str(len(set(q_i))))
        
        prev_s = s
        #plt.legend()
        prev_t, prev_z, u = np.float(t), np.float(z), np.float(u)
        #plt.title('CPT-est vs true: ({}, {}, {})'.format(t,z,u))
        axs[int(u)].legend(prop = {'size':20})
        axs[int(u)].tick_params(axis = 'both', which = 'major', labelsize = 20)
        
    fig.suptitle('CPT-est vs true: s=({}, {})'.format(prev_t, prev_z), fontsize=30)
    fig.subplots_adjust(hspace = .4)
    plt.savefig("./shiBM/Bug3/CPTPred_trials_({}, {}).png".format(prev_t, prev_z), dpi = 100)
    plt.close(fig)
    
    ######################### VISIT(t, x, u) SPERL vs SPE #####################
    
    # join inner/outer?
    # df_SPE, df_SPERL may visit different (t, x, u) even for the same obj/env params
    # EXAMPLE:
    # q_true: None -> not visited during training
    '''df1 = df_SPERL.loc[(190922111945, 'q_true'), :].dropna(axis = 'columns', how = 'all')
    df1 = df1.dropna(axis = 'rows', how = 'all')
    
    df2 = df_SPERL.loc[(190922112205, 'q_true'), :].dropna(axis = 'columns', how = 'all')
    df2 = df2.dropna(axis = 'rows', how = 'all')
    print('SPERL1 == SPERL2 visits?', df1.shape, df2.shape)
    
    visited_tx1 = set()
    for a, b, c in df1.columns:
        visited_tx1.add((a, b))
    
    visited_tx2 = set()
    for a, b, c in df2.columns:
        visited_tx2.add((a, b))'''
    
    idx = pd.IndexSlice
    df = df_SPERL.loc[idx[:, ['q_true']], :].dropna(axis = 'columns', how = 'all')
    df = df.dropna(axis = 'rows', how = 'all')
    print('SPERL_all visits', df.shape)
    
    visited_tx = set()
    for a, b, c in df.columns:
        visited_tx.add((a, b))
    
    SPE_visited_tx = set()
    for a, b, c in df_SPE.columns:
        SPE_visited_tx.add((a, b))
    
    # Case I: df_SPERL visits MORE than df_SPE
    # @EXAMPLE, (1, -.2) in SPERL but not in SPE
    print('in SPERL, not in SPE', visited_tx - SPE_visited_tx)
    # 2 Alternatives:
    # 1. Do 'join inner', i.e. throw away any t,x,u visited by SPERL that is
    # beyond feat_range_z[t]
    # 2. Implement feat_range_z[t], pre-train of SPERL
    # --> JOIN INNER
    # Do (1) first, out-of-bound counts at each t should be small 
    # (<.05*total_train_trajectories_sampled)
    
    # Case II: df_SPE visits MORE than df_SPERL
    # @EXAMPLE, (2, 4.0-5.2) in SPE but not in SPERL
    print(SPE_visited_tx - visited_tx)
    # inevitable due to on-policy training w/ small explore-rate
    # --> JOIN OUTER
    # but, since we will use 'q/policy_i' instead of 'q_true' to pd.concat,
    # this case won't happen since df_SPERL does not drop any t,x,u.
    # sa_memory just used to count_visits(t, x, u) & the following holds,
    # CPT_val[t, x, u] = [0, 0, ..., 0] for non-visited (t, x, u)
    # CPT_true[t, x, u] = [None, None, ..., None]
        
###############################################################################
def init_csv(model, hyperparams, s0 = (0, 0), dyn = False):
    
    folder_ = './results/'
    if dyn == True:
        folder_ += 'dynamic/'
        
    seed, p_win, alpha, rho1, rho2, lmbd, n_batch, train_num, actor_timescale, target_type_ = hyperparams
    
    x = []
    init = []
    initUtility = []
    
    for key in sorted(model.init_policy.keys()):
        x += [str(key)]
        act_probs, _ = model.init_policy[key]
        init += [act_probs[1]]
        
        _, _, true_cpt = model.evaluate_policy(init_sa = (key, None), 
                                               return_episode_rewards = False, 
                                               for_critic = True)
        initUtility += [true_cpt]
        
    if model.policy.name == 'GreedyPolicy':
        
        f = open(folder_ + 'greedySPERL.csv', 'a', newline = '')
        to_append = [hyperparams + [model.policy.name, s0, 'init'] + init + initUtility]
        writer = csv.writer(f)
        writer.writerows(to_append)
        f.close()
        
    else:
        
        f = open(folder_ + 'gradSPSA.csv', 'a', newline = '')
        to_append = [hyperparams + [model.policy.name, s0, 'init'] + init + initUtility]
        writer = csv.writer(f)
        writer.writerows(to_append)
        f.close()

def record_csv(model, hyperparams, model_for_SPE = None, model_for_Precomm = None, 
               s0 = (0, 0), dyn = False, plotting = True):
    
    folder_ = './results/'
    if dyn == True:
        folder_ += 'dynamic/'
        
    seed, p_win, alpha, rho1, rho2, lmbd, n_batch, train_num, actor_timescale, target_type_ = hyperparams
    
    ## 1. Compute SPE policy, SPE utility, Precomm policy, Precomm Utility
    
    x = []
    
    trainedUtility = []
    trained = []
    
    #if model_for_SPE is not None:
    SPEUtility = []
    trueSPE = []
    
    #if model_for_Precomm is not None:
    PrecommUtility = []
    truePrecomm = []

    #Precomm_w_eps = []
    
    for key in sorted(model.trained_policy.keys()):
        
        x += [str(key)]
        
        act_probs, _ = model.trained_policy[key]
        trained += [act_probs[1]]
        _, _, true_cpt = model.evaluate_policy(init_sa = (key, None), 
                                                       return_episode_rewards = False, 
                                                       for_critic = True)
        
        trainedUtility += [true_cpt]
        
        if model_for_SPE is not None:
            trueSPE += [model_for_SPE.trained_policy[key][1]]
            _, _, true_cpt = model_for_SPE.evaluate_policy(init_sa = (key, None), 
                                                           return_episode_rewards = False, 
                                                           for_critic = True)
            SPEUtility += [true_cpt]
        
        if model_for_Precomm is not None:
            truePrecomm += [model_for_Precomm.trained_policy[key][1]]
            _, _, true_cpt = model_for_Precomm.evaluate_policy(init_sa = (key, None),
                                                               return_episode_rewards = False,
                                                               for_critic = True)
            PrecommUtility += [true_cpt]
        
        #model_for_Precomm.policy.eps = .3 # eps, cpt = (0, .74), (.1, ?), (.3, ?)
        #_, _, true_cpt = model_for_Precomm.evaluate_policy(init_sa = (key, None),
        #                                                   deterministic = False,
        #                                                   return_episode_rewards = False,
        #                                                   for_critic = True)
        #Precomm_w_eps += [true_cpt]
    
    '''
    plt.figure()
    plt.scatter(x, PrecommUtility, label = 'loss-exit (~precomm)', alpha = .7, color = 'magenta')
    plt.scatter(x, Precomm_w_eps, label = 'stochastic loss-exit (eps=' + str(model_for_Precomm.policy.eps) + ')', alpha = .4, color = 'magenta')
    plt.scatter(x, SPEUtility, label = 'trueSPE', alpha = .7, color = 'cyan')
    plt.legend(prop={'size':7.5})
    plt.xticks(rotation=90)
    
    plt.axvspan(.5, 1 + .5, facecolor = 'gray', alpha = .1)
    plt.axvspan(2 + .5, 3 + .5, facecolor = 'gray', alpha = .1)
    plt.axvspan(5 + .5, 7 + .5, facecolor = 'gray', alpha = .1)
    plt.axvspan(9 + .5, 11 + .5, facecolor = 'gray', alpha = .1)
    plt.axvspan(14 + .5, 17 + .5, facecolor = 'gray', alpha = .1) 
    
    title_ = '(p, alpha, rho1, rho2, lmbd): ' + str((p_win, alpha, rho1, rho2, lmbd)) + ', \n seed: ' + str(model.seed) + ', ss_const: ' + str(model.policy.ss_const) + ', n_batch: ' + str(n_batch) + ', n_train: ' + str(train_num)  + ', timescale: ' + str(actor_timescale)
    title_ = 'CPT Utility referenced at t, x \n' + title_
    title_ += ', \n (target, w_qtile, I, critic_lr, empty_memory): ' + str((target_type_, model.critic.with_quantile, model.critic.support_size, model.critic.lr, model.empty_memory))
    plt.title(title_)
    '''
    
    ###########################################################################
    if model.policy.name == 'trueSPE':
        
        ## 1.
        f = open(folder_ + 'greedySPERL.csv', 'a', newline = '')
        
        to_append = [hyperparams + [model.policy.name, s0, None] + trained + trainedUtility] #, 
                     #hyperparams + ['trueSPE', s0, None] + trueSPE + SPEUtility, 
                     #hyperparams + ['loss-exit', s0, None] + truePrecomm + PrecommUtility]
        writer = csv.writer(f)
        writer.writerows(to_append)
        f.close()
        
        # 3.
        trueSPEQ = []
        for key in sorted(model.trained_policy.keys()):
            _, _, true_cpt0 = model.evaluate_policy(init_sa = (key, 0), 
                                                            deterministic = True, 
                                                            return_episode_rewards = False, 
                                                            for_critic = True)
            _, _, true_cpt1 = model.evaluate_policy(init_sa = (key, 1), 
                                                            deterministic = True, 
                                                            return_episode_rewards = False, 
                                                            for_critic = True)
            trueSPEQ += [true_cpt0, true_cpt1]
        
        to_append = [hyperparams + ['trueSPE', s0, None] + trueSPEQ]
        f = open(folder_ + 'greedySPERL_critic_evo.csv', 'a', newline = '')
        writer = csv.writer(f)
        writer.writerows(to_append)
        f.close()
        
    if model.policy.name == 'truePrecomm':  
        
        # 1.
        f = open(folder_ + 'greedySPERL.csv', 'a', newline = '')
        
        to_append = [hyperparams + [model.policy.name, s0, None] + trained + trainedUtility] #, 
                     #hyperparams + ['trueSPE', s0, None] + trueSPE + SPEUtility, 
                     #hyperparams + ['loss-exit', s0, None] + truePrecomm + PrecommUtility]
        writer = csv.writer(f)
        writer.writerows(to_append)
        f.close()
        
        # 3.
        truePrecommQ = []
        for key in sorted(model.trained_policy.keys()):
            _, _, true_cpt0 = model.evaluate_policy(init_sa = (key, 0), 
                                                                deterministic = True, 
                                                                return_episode_rewards = False, 
                                                                for_critic = True)
            _, _, true_cpt1 = model.evaluate_policy(init_sa = (key, 1), 
                                                                deterministic = True, 
                                                                return_episode_rewards = False, 
                                                                for_critic = True)
            truePrecommQ += [true_cpt0, true_cpt1]
            
        to_append = [hyperparams + ['truePrecomm', s0, None] + truePrecommQ]
    
        f = open(folder_ + 'greedySPERL_critic_evo.csv', 'a', newline = '')
        writer = csv.writer(f)
        writer.writerows(to_append)
        f.close()
        
    if model.policy.name == 'GreedyPolicy':
        
        ## 1.
        f = open(folder_ + 'greedySPERL.csv', 'a', newline = '')
        
        to_append = [hyperparams + [model.policy.name, s0, None] + trained + trainedUtility, 
                     hyperparams + ['trueSPE', s0, None] + trueSPE + SPEUtility, 
                     hyperparams + ['loss-exit', s0, None] + truePrecomm + PrecommUtility]
        writer = csv.writer(f)
        writer.writerows(to_append)
        f.close()
        
        ## 2.
        f = open(folder_ + 'greedySPERL_learning_curve.csv', 'a', newline = '')
        to_append = [hyperparams + [model.policy.name, s0, 'expected'] + model.stats['mean_rewards'], 
                     hyperparams + [model.policy.name, s0, 'cpt'] + model.stats['cpt_rewards']]
        writer = csv.writer(f)
        writer.writerows(to_append)
        f.close()
        
        ## 3. Critic Evolution & Bias; MAKE ALL STATES INCL IN KEY OF CPT_VAL, CPT_TRUE instead of by visits!!
        trueSPEQ = []
        truePrecommQ = []
        
        trainQ_val = []
        trainQ_true = []
        
        #for key in sorted(model_for_SPE.trained_policy.keys()):
        for key in sorted(model.trained_policy.keys()):
            trainQ_val_0 = list(np.array(model.CPT_val[(key, 0)])[:, 0]) # convert ':' to N rows!
            trainQ_val_1 = list(np.array(model.CPT_val[(key, 1)])[:, 0])
            trainQ_val += [trainQ_val_0]
            trainQ_val += [trainQ_val_1]
            
            trainQ_true_0 = list(np.array(model.CPT_true[(key, 0)])[:, 0])
            trainQ_true_1 = list(np.array(model.CPT_true[(key, 1)])[:, 0])
            trainQ_true += [trainQ_true_0]
            trainQ_true += [trainQ_true_1]
            
            if model_for_SPE is not None:
                _, _, true_cpt0 = model_for_SPE.evaluate_policy(init_sa = (key, 0), 
                                                                deterministic = True, 
                                                                return_episode_rewards = False, 
                                                                for_critic = True)
                _, _, true_cpt1 = model_for_SPE.evaluate_policy(init_sa = (key, 1), 
                                                                deterministic = True, 
                                                                return_episode_rewards = False, 
                                                                for_critic = True)
                trueSPEQ += [true_cpt0, true_cpt1]

            if model_for_Precomm is not None:            
                _, _, true_cpt0 = model_for_Precomm.evaluate_policy(init_sa = (key, 0), 
                                                                    deterministic = True, 
                                                                    return_episode_rewards = False, 
                                                                    for_critic = True)
                _, _, true_cpt1 = model_for_Precomm.evaluate_policy(init_sa = (key, 1), 
                                                                    deterministic = True, 
                                                                    return_episode_rewards = False, 
                                                                    for_critic = True)
                truePrecommQ += [true_cpt0, true_cpt1]
            
        row_header = [hyperparams + [model.policy.name, s0, 'train_val']]
        num_rows = np.array(trainQ_val).shape[1]
        row_header *= num_rows
        to_append1 = np.concatenate((np.array(row_header), np.transpose(np.array(trainQ_val))), axis = 1) # dim: (200, header_size + txu_num)
        
        row_header = [hyperparams + [model.policy.name, s0, 'train_true']]
        row_header *= num_rows
        to_append2 = np.concatenate((np.array(row_header), np.transpose(np.array(trainQ_true))), axis = 1)
        
        to_append = np.concatenate((to_append1, to_append2), axis = 0).tolist()
        if model_for_SPE is not None:
            to_append += [hyperparams + ['trueSPE', s0, None] + trueSPEQ]
        if model_for_Precomm is not None:
            to_append += [hyperparams + ['truePrecomm', s0, None] + truePrecommQ]
        
        f = open(folder_ + 'greedySPERL_critic_evo.csv', 'a', newline = '')
        writer = csv.writer(f)
        writer.writerows(to_append)
        f.close()
        
    else:
        ## 1.
        f = open(folder_ + 'gradSPSA.csv', 'a', newline = '')
        to_append = [hyperparams + [model.policy.name, s0, None] + trained + trainedUtility] #, 
                     #hyperparams + ['trueSPE', s0, None] + trueSPE + SPEUtility, 
                     #hyperparams + ['loss-exit', s0, None] + truePrecomm + PrecommUtility]
        writer = csv.writer(f)
        writer.writerows(to_append)
        f.close()
        
        ## 2.
        f = open(folder_ + 'gradSPSA_learning_curve.csv', 'a', newline = '')
        to_append = [hyperparams + [model.policy.name, s0, 'expected'] + model.stats['mean_rewards'], 
                     hyperparams + [model.policy.name, s0, 'cpt'] + model.stats['cpt_rewards']]
        writer = csv.writer(f)
        writer.writerows(to_append)
        f.close()
        
        ## 3. Implement if actor-critic;
        
################################ UNUSED FUNCTIONS ############################     

'''# EDGE CASES: filter checks
# I 401 is the critical case that needs us to code this: REWRITE RHS cause positive values to replace actual negative values
quantiles401_0 = [-10.84,-10.58,-10.48,-10.34,-10.62,-10.44,-10.32,-10.12,-10.22,-9.92,-9.68,-9.24,-8.36,-6.08,-0.58,4.72,7.54,9.06,9.16,9.26,9.36,9.48,9.2,9.28,9.36,9.44,9.52,9.6,9.68,9.72,9.78,9.86,9.94,10.02,10.1,10.16,10.22,9.9,9.94,10,10.1,10.18,10.26,10.3,10.4,10.48,10.58,10.66,10.7,10.82]
quantiles = list(quantiles401_0) 

quantiles401_1 = [-10.72,-10.54,-10.34,-10.2,-10.04,-9.88,-9.76,-9.64,-9.48,-9.34,-9.26,-9.14,-9,-7.58,-2.62,2.2,4.42,5.42,6.24,7.04,7.84,8.6,9.34,9.88,9.98,10.02,10.04,10.06,10.06,10.1,10.12,10.12,10.16,10.18,10.2,10.2,10.26,10.3,10.34,10.38,10.36,10.38,10.44,10.48,10.54,10.58,10.64,10.7,10.72,10.8]
quantiles = list(quantiles401_1) 

quantiles401_2 = [-10.76,-10.58,-10.52,-10.46,-10.38,-10.32,-10.28,-10.22,-10.16,-10.04,-9.88,-9.62,-8.94,-6.84,-1.48,3.78,6.4,8.16,8.96,9.04,9.12,9.2,9.28,9.36,9.44,9.52,9.6,9.68,9.76,9.82,9.9,9.94,10,10.06,10.1,10.14,10.18,10.22,10.28,10.3,10.3,10.32,10.34,10.38,10.42,10.48,10.56,10.64,10.74,10.82]
quantiles = list(quantiles401_2)

# II 4201 good performance can be due to 'RHS BIAS' bf HALF2 REWRITE
quantiles4201_0 = [1.57, 2.93, 4.75,7.57,9.33,9.47,9.51,9.57,9.59,9.63,9.65,9.69,9.71,10.23,11.61,13.03,14.71,16.17,18.27,19.91,22.03,24.09,25.97,26.77,27.07,27.37,27.63,27.89,28.15,28.41,28.65,28.89,29.11,29.27,29.41,29.53,29.65,29.71,29.79,29.85,29.95,30.01,30.05,30.15,30.19,30.29,30.35,30.41,30.55,30.73]
quantiles = list(quantiles4201_0) # CANNOT HELP HERE, p.filter seems to be too small...

# p.filter=.75
quantiles4201_1 = [2.75,6.25,9.41,9.55,9.65,9.77,9.87,9.95,10.05,10.13,10.25,10.35,10.99,13.79,16.53,20.33,25.57,28.43,28.79,29.07,29.41,29.61,29.77,29.85,29.91,29.99,30.01,30.05,30.07,30.09,30.11,30.13,30.15,30.19,30.21,30.21,30.25,30.29,30.31,30.31,30.35,30.37,30.43,30.45,30.51,30.55,30.55,30.63,30.71,30.81]
quantiles = list(quantiles4201_1)

quantiles4201_2 = [2.91,7.67,9.37,9.49,9.57,9.61,9.65,9.75,9.79,9.85,9.91,9.97,10.33,11.33,12.31,13.27,14.51,18.67,24.33,29.13,29.31,29.41,29.53,29.61,29.67,29.73,29.79,29.89,29.91,29.93,30.03,30.09,30.07,30.13,30.21,30.25,30.19,30.27,30.31,30.23,30.33,30.35,30.39,30.37,30.39,30.47,30.53,30.61,30.71,30.83]
quantiles = list(quantiles4201_2)

quantiles4201_b = [2.59,5.39,9.11,9.39,9.45,9.47,9.51,9.55,9.59,9.61,9.65,9.69,9.71,9.75,9.79,9.79,9.83,9.85,11.75,16.23,19.25,20.97,22.95,24.69,26.59,27.99,28.11,28.25,28.33,28.45,28.57,28.67,28.79,28.87,28.97,29.07,29.19,29.29,29.39,29.53,29.61,29.75,29.85,29.95,30.09,30.23,30.37,30.49,30.67,30.83]
# bad seed learns 30-qtile values wrongly! Flipped cz 4200 n 4201 values v close! justify w 2eps-SPE (inside set of eps-SPE, less related to V0)
quantiles = list(quantiles4201_b)

quantiles4201_g = [2.55,5.35,8.99,9.57,9.77,9.95,10.13,10.29,10.45,10.59,10.75,10.89,11.05,11.23,11.41,11.61,11.81,11.99,14.27,18.67,21.61,23.25,25.15,26.81,28.63,29.95,29.99,30.05,30.05,30.09,30.13,30.15,30.19,30.19,30.21,30.23,30.27,30.29,30.31,30.37,30.37,30.43,30.45,30.47,30.53,30.59,30.65,30.69,30.79,30.87]
quantiles = list(quantiles4201_g)

quantiles4200_g = [7.91,19.73,19.55,19.37,19.19,20.01,19.83,19.65,19.47,19.29,20.11,19.93,19.75,19.57,19.39,20.21,20.03,19.85,19.67,19.49,20.31,20.13,19.95,19.77,19.59,20.41,20.23,20.05,19.87,19.69,20.51,20.33,20.15,19.97,19.79,20.61,20.43,20.25,20.07,19.89,20.71,20.53,20.35,20.17,19.99,20.81,20.63,20.45,20.27,20.09]
quantiles4200_b = [7.96,19.88,19.8,19.72,19.64,19.56,19.48,19.4,19.32,19.24,20.16,20.08,20,19.92,19.84,19.76,19.68,19.6,19.52,19.44,20.36,20.28,20.2,20.12,20.04,19.96,19.88,19.8,19.72,19.64,20.56,20.48,20.4,20.32,20.24,20.16,20.08,20,19.92,19.84,20.76,20.68,20.6,20.52,20.44,20.36,20.28,20.2,20.12,20.04]        


# III 4400 has many negative qvalGaps
quantiles4400_0 = [3.89,11.67,19.45,27.23,35.01,39.79,39.57,39.35,40.13,39.91,39.69,39.47,40.25,40.03,39.81,39.59,39.37,40.15,39.93,39.71,39.49,40.27,40.05,39.83,39.61,40.39,40.17,39.95,39.73,40.51,40.29,40.07,39.85,40.63,40.41,40.19,39.97,40.75,40.53,40.31,40.09,39.87,40.65,40.43,40.21,39.99,40.77,40.55,40.33,40.11]
quantiles = list(quantiles4400_0)

# IV 000 more 'continuous' distribution (more x'|x,a); truncation induce bias at 0, 0, 1;
# may be problematic for cptMdp w 001 ~= 000
quantiles001_0 = [-12.08,-10.02,-8.48,-4.72,-2.4,-1.04,0.06,0.38,0.74,1.34,2.92,4.52,6,7.62,9,9.86,10.62,11.42,12.34,13.58,16.44,20.18,24.24,27.66,29.08,29.3,29.48,29.68,29.92,30.08,30.48,31,31.58,32.32,33.38,35.24,36.68,37.18,37.56,37.94,38.32,38.68,39.04,39.42,39.78,40.12,40.28,40.4,40.54,40.74]
quantiles = list(quantiles001_0)

# IF TRESH_ TOO LARGE (1.36 FOR 4201) --> SIGN OF PRE-CONVERGED
plt.plot(quantiles4201_b, label = '4201-b')
plt.plot(quantiles4201_g, label = '4201-g')
plt.legend()
plt.plot(quantiles4201_2, label = '4201-2')

plt.plot(quantiles401_0, label = '401-0')
#plt.plot(qval_filtered, label = '401-1 filtered')
plt.plot(quantiles401_1, label = '401-1')
plt.plot(quantiles401_2, label = '401-2')

plt.plot(quantiles4400_0, label = '4400-0')
plt.plot(qval_filtered, label = '4400-0 filtered')

plt.plot(quantiles001_0, label = '001-0')
plt.plot(qval_filtered, label = '001-0 filtered')
plt.legend()

plt.plot(quantiles, label = 'unfiltered')
for p_filter in np.linspace(.6, .9, 10):
    filteredqtiles = filtering(quantiles, p_filter)
    plt.plot(filteredqtiles, label = str(p_filter))
    
plt.legend()
'''

'''# Check CPT_compute(): C(40) > C([30, 50]) - p = .5, bet = 10;
compute_CPT([30]*250 + [50]*250)
compute_CPT([30, 50])
prob_weight(1/2, True) * utility(50, True) + (1 - prob_weight(1/2, True)) * utility(30, True)
compute_CPT([40])
compute_CPT([40]*500)
utility(-40, False)
compute_CPT([-40])
prob_weight(1/2, False) * utility(-50, False) + (1 - prob_weight(1/2, False)) * utility(-30, False)
compute_CPT([-50, -30])

# Check cptVal variations with quantile noises (within small noise thresh; cf. filter_ formula in 'filtering')
print('cptID:', alpha, rho1, lmbd)
cptList4 = []
for _ in range(10):
    np.random.shuffle(quantiles4_mono)
    cptList4 += [compute_CPT(quantiles4_mono, sort=False)]
print('maxgap:', min(cptList4), max(cptList4))

cptList5 = []    
for _ in range(10):
    np.random.shuffle(quantiles5_mono)
    cptList5 += [compute_CPT(quantiles5_mono, sort=False)]
print('maxgap:', min(cptList5), max(cptList5))
'''
  
def filtering_debug2(quantiles, p_filter = .75): # + 1e-16): #, mono_filter = -.03):
    # expect list of len K, unsorted
    
    #Insert discreteDistribFilter_
    qval_gaps = np.array(quantiles)[1:] - np.array(quantiles)[:-1] # len: K-1
    tresh_ = np.quantile(qval_gaps, p_filter) #np.round(np.quantile(qval_gaps, p_filter), 4)
    filter_ = qval_gaps <= tresh_ + 1e-6 #NUMERICAL ERROR wo the addition!! | np.sum(filter_) ~36 (if K = 50)
    #print(tresh_, qval_gaps)
    #print(sum(filter_), filter_)
    
    #Insert monofilter w distance from thresh_ (like average) considered
    filter_ = np.multiply(filter_,  1 - np.multiply(qval_gaps < -1e-6, np.abs(qval_gaps - tresh_) > 1e-6))
    filter_ = np.append([False], filter_)
    #print(sum(filter_), filter_)
    
    #Start filter quantiles
    # if True, copy quantiles
    qval_filtered = np.multiply(filter_, np.array(quantiles))
    
    # if False, copy closest RHS
    rev_0ID = np.where(qval_filtered == 0)[0][::-1]
    for i in rev_0ID:
        
        if i+1 > len(qval_filtered)-1:
            continue
        
        z = qval_filtered[i+1]
        qval_filtered[i] = z
     
    # if no RHS, copy LHS    
    zeroID = np.where(qval_filtered == 0)[0]
    for i in zeroID:
        z = qval_filtered[i-1]
        qval_filtered[i] = z
        
    '''# APPLY ON qval_filtered!
    # MAY NOT BE NEEDED (esp. for quantiles3! shd skip); TRY IF NOT ORDERED HOW MUCH IS COMPUTE_CPT AFFECTED
    # w monotresh != 0, we need to reorder!
    ids = np.where(filter_ == 0)[0][::-1]
    for i in range(len(ids)):
        #id_ = ids[i]
        #ids_next = ids[i+1]
        
        if i-1 < 0: # > len(ids)-1:
            X = quantiles[ids[i]+1:]
            
            # check if X is increasing, if not..
            # lbub w curmin loop
            
            print(np.average(X))
            #print(sorted(X))
            if len(X)>0:
                quantiles[ids[i]+1:] = sorted(X)
            
        else:
            X = quantiles[ids[i]+1:ids[i-1]]
            if len(X)>0:
                quantiles[ids[i]+1:ids[i-1]] = sorted(X)
    
        print(X)
    
    print(compute_CPT(quantiles, sort=False), np.array(quantiles))
    '''
    return list(qval_filtered)