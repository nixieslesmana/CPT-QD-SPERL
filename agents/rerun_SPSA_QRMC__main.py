# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 12:38:31 2022

@author: NIXIESAP001
"""

#import gym
import itertools
#import matplotlib.pyplot as plt
import numpy as np
#import tensorflow as tf
import scipy.special as sp
import collections
import random
import math
#import plotting
import record_csv
from datetime import datetime

import sys
import os

cwd = os.getcwd()
sys.path.append(cwd) # nb: append cwd or "../" not enough, need to add __init__ to the cwd;
if "../" not in sys.path:
    sys.path.append("../")
    
from lib.envs.barberis_casino import barberisCasino
from stable_baselines.common.misc_util import set_global_seeds

def barberisFeaturize(obs):
    t, z = obs
    obs_feature = np.zeros(env.observation_space.spaces[0].n * env.observation_space.spaces[1].n)
    loc = int(10 * t + 5 + z // env.bet)
    obs_feature[loc] = 1
    
    return obs_feature, loc

################################## POLICY #####################################

class SPSAPolicy():
    """
    Policy Function approximator | 
    @stable-baselines: .common.policies.PolicyType(sess, ob_space, ...)
    policy.value(obs, ...) returns Q(obs, self.action)
    """
    
    def __init__(self, env, step_size = 1.0, perturb_const = 1.9, theta_init = None, THETA_MIN = .1, THETA_MAX = 1.0): 
        
        self.name = 'SPSAPolicy'
        self.vf_pair = [0.0, 0.0]
        self.is_even = False
        
        self.beta = 1
        self.ss_const = step_size
        self.pc_const = perturb_const
        self.step_size = None
        self.perturb_const = None
        self._get_step_size()
        self._get_perturb_const()
        
        self.grads =  np.zeros((env.observation_space.spaces[0].n * env.observation_space.spaces[1].n, env.action_space.n))# vector of length d, init to 0^d
        
        self.THETA_MIN = THETA_MIN
        self.THETA_MAX = THETA_MAX
        
        if theta_init is not None:
            self.theta = theta_init + self.grads
            # alt: vectors of len d, init to 0/.5^d
        else:
            self.theta = self.THETA_MIN + (self.THETA_MAX - self.THETA_MIN) * np.random.random(self.grads.shape)
            print('theta[:4, :]:', self.theta[:4, :])
        
        self.init_theta = np.copy(self.theta)
        
        self.perturb_noise = None
        self.thetaSPSA = None
        self._get_perturb_noise()
        self._get_thetaSPSA()
    '''
    def init_to_precomm(self):
        
        theta = np.zeros(self.grads.shape)
        
        for t in :
            for z in:
                _, loc = barberisFeaturize((t, z))
                
                if z < 0:
                    theta[loc, :] = [1.0, 0.0]
                else:
                    theta[loc, :] = [0.0, 1.0]
        
        
        return theta'''
    
    def _get_is_even(self):
        
        self.is_even = not self.is_even
        
    def _get_perturb_noise(self):
        
        if self.is_even is False:
            
            '''print('---')
            print('noise content updated, only once per 2 is_even?')
            print('check is_even:', self.is_even)'''
            
            self.perturb_noise = np.zeros(self.theta.shape)
            
            if len(self.perturb_noise.shape) == 1:
                for i in range(self.theta.shape[0]):
                    rng = random.randrange(2)
                    if rng == 0:
                        rng = -1
                    self.perturb_noise[i] = rng
            elif len(self.perturb_noise.shape) == 2:
                for i in range(self.theta.shape[0]):
                    for j in range(self.theta.shape[1]):
                        rng = random.randrange(2)
                        if rng == 0:
                            rng = -1
                        self.perturb_noise[i, j] = rng
            else:
                print('@SPSAPolicy, theta/thetaSPSA/perturb_noise dim issue!!')
                raise NotImplementedError
                
            '''print('check perturb_noise:', self.perturb_noise[:4, :])'''
    
    def _get_thetaSPSA(self):
        # Call at init, after theta_init & perturb_noise!
        # Call each time policy.update(), policy.perturb_noise.update()
        
        '''print('---')
        print('thetaSPSA updated...')'''
        
        if self.thetaSPSA is not None:
            '''print('bf:', self.thetaSPSA[:4, :])'''
        
        if self.is_even == False:
            self.thetaSPSA = self.theta + self.perturb_const * self.perturb_noise
        else:
            self.thetaSPSA = self.theta - self.perturb_const * self.perturb_noise
        
        '''print('aft:', self.thetaSPSA[:4, :])'''
        
    def _get_step_size(self, n = 0):#, const = 1.0):
        # satisfies condition (A3), init  to 1.0
        
        if self.is_even:
            return
        
        n_iterations = math.ceil(n / 2)
        self.step_size = math.pow(n_iterations + 50, -1) * self.ss_const
        
    def _get_perturb_const(self, n = 0):#, const = 1.9):
        # satisfies (A3), init to 1.9
        
        if self.is_even: 
        # @Prash, clarify frequency if its only done once per outerloop (nSPSA)
            return
        
        '''print('---')
        print('calls _get_perturb_const, only once per 2 is_even?')
        print('check is_even:', self.is_even)'''
        
        n_iterations = math.ceil(n / 2)
        self.perturb_const = math.pow(n_iterations + 1, -.101) * self.pc_const
        # @Prash, no +1 --> div by 0 error, clarify!
        
    def _predict(self, state, deterministic = False):
        
        if deterministic:
            cur_theta = self.theta # test -> use theta
        else:
            cur_theta = self.thetaSPSA # train -> use thetaSPSA
        
        state, _ = barberisFeaturize(state) # dim = (1, 6 * 11)
        out = np.matmul(state, cur_theta) # dim = (1, 2)
        action_probs = sp.softmax(out * self.beta) # dim = (1, 2)
        
        return action_probs
    
    def _clip(self, _theta):
        # @clip: function to prevent theta from going outside \Theta = [.1, 1]^d.
        # not mentioned how to set in paper -> SEE CODE;
        if _theta <= self.THETA_MIN:
            return self.THETA_MIN
        elif _theta >= self.THETA_MAX:
            return self.THETA_MAX
        else:
            return _theta
    
    def _update(self):
        
        if self.is_even == False:
            
            '''print('---')
            print('vf_pair used @policy._update')
            print('vf_pair:', self.vf_pair)
            
            
            print('---')
            print('Bf update, theta[:4, :]:', self.theta[:4, :])
            print('vf_pair used:', self.vf_pair)
            '''
            update_factor = (self.vf_pair[0] - self.vf_pair[1]) / (2 * self.perturb_const)
            
            if len(self.theta.shape) == 1:
                
                for i in range(self.theta.shape[0]):
                    self.grads[i] = update_factor / self.perturb_noise[i]
                    self.theta[i] = self._clip(self.theta[i] + self.step_size * self.grads[i])
            
            elif len(self.theta.shape) == 2:
                
                for i in range(self.theta.shape[0]):
                    for j in range(self.theta.shape[1]):
                        self.grads[i, j] = update_factor / self.perturb_noise[i, j]
                        self.theta[i, j] = self._clip(self.theta[i, j] + self.step_size * self.grads[i, j])
            
            else:
                print('@SPSAPolicy, theta/thetaSPSA/perturbnoise dim issue!!')
                raise NotImplementedError
            '''
            print('Aft update, theta[:4, :]:', self.theta[:4, :])
            print('=====')
            '''
##################################### CRITIC ##################################

class CPTCritic():
    
    def __init__(self, env, support_size = 10, with_quantile = False, with_huber = False,
                 param_init = 0.0, lr = .01):
        self.env = env
        self.with_quantile = with_quantile
        
        self.lr = lr
        self.cpt_theta = None
        self.with_huber = None
        self.support_size = None
        self.qtile_theta = None
        
        self.qtile_theta_pos = None
        self.qtile_theta_neg = None
        
        self.lossgrad_evolution = {}
        
        # do not predict C[Z | s, a] but C[Z | s]
        # loss_grad, qtile_theta change dim!! to (6*11, 1)
        # ...
        # for all indexing related to action, FIX TO 0!
        # outside, change data/CPT_dict keys --> merging (s, 0), (s, 1) to (s, 0);
        
        if not self.with_quantile:
            self.loss_grad = np.zeros((env.observation_space.spaces[0].n * env.observation_space.spaces[1].n, 
                                      env.action_space.n)) # (6*11, 2)
            self.cpt_theta = param_init + np.zeros(self.loss_grad.shape) # dim(cpt_theta) = dim(loss_grad) = 66, |A|
        else:
            self.with_huber = with_huber
            self.support_size = support_size # I
            self.loss_grad = np.zeros((env.observation_space.spaces[0].n * env.observation_space.spaces[1].n, 
                                      env.action_space.n, self.support_size)) # (6*11, 2, I)
            self.qtile_theta = param_init + np.zeros(self.loss_grad.shape) # dim(qtile_theta) = dim(loss_grad) = 66, |A|, I
            
            # assert qtile_theta.shape = qtile_theta_pos.shape = qtile_theta_neg.shape
            self.loss_grad_pos = np.zeros((env.observation_space.spaces[0].n * env.observation_space.spaces[1].n, 
                                      env.action_space.n, self.support_size))
            self.qtile_theta_pos = param_init + np.zeros(self.loss_grad.shape)
            self.loss_grad_neg = np.zeros((env.observation_space.spaces[0].n * env.observation_space.spaces[1].n, 
                                      env.action_space.n, self.support_size))
            self.qtile_theta_neg = param_init + np.zeros(self.loss_grad.shape)
            
    def cpt_featurize(self, state, action):
        featurized_sa = np.zeros(self.cpt_theta.shape) # dim = (66, |A|)
        _, loc = barberisFeaturize(state)
        featurized_sa[loc, action] = 1
        return featurized_sa
    
    def qtile_featurize(self, state, action, i):
        featurized_sai = np.zeros(self.qtile_theta.shape) # dim = (66, |A|, I)
        _, loc = barberisFeaturize(state)
        featurized_sai[loc, action , i] = 1
        return featurized_sai
    
    def CPTpredict(self, state, action, is_even = None): # cf. policy._predict()
        
        if not self.with_quantile:
            cpt_estimate = np.tensordot(self.cpt_featurize(state, action), self.cpt_theta, 
                                  axes = len(self.cpt_theta.shape)) # shape = (66, 2), num_axes = 2
            cpt_estimate = np.float(cpt_estimate)
            
        else:
            cpt_estimate = compute_CPT([self.qtilepredict(state, action, i, is_even) for i in range(self.support_size)])
    
        return cpt_estimate
    
    def qtilepredict(self, state, action, i, is_even = None): # by loss-grad @Dabney, use tabularFA, 
        # assert with_quantile == True;
        if is_even == None:
            qtile_estimate = np.tensordot(self.qtile_featurize(state, action, i), self.qtile_theta, 
                                          axes = len(self.qtile_theta.shape)) # shape = (66, 2, I), num_axes = 3
        elif is_even == False:
            #raise NotImplementedError
            qtile_estimate = np.tensordot(self.qtile_featurize(state, action, i), self.qtile_theta_pos, 
                                          axes = len(self.qtile_theta_pos.shape)) 
        else:
            #raise NotImplementedError
            qtile_estimate = np.tensordot(self.qtile_featurize(state, action, i), self.qtile_theta_neg, 
                                          axes = len(self.qtile_theta_neg.shape)) 
        
        return qtile_estimate
    
    def compute_Loss(self, data):
        raise NotImplementedError
    
    def compute_LossGrad(self, data, is_even = None):
        ## manual backprop, given tabularFA of qtile_estimate;
        lossgrad = np.zeros(self.loss_grad.shape) # always clear grad
            
        if not self.with_quantile: # [MC-only]
            
            for state, action in data.keys(): # cf. CPT_val loop - @.evaluate_critic()
                # CHANGE CPT_dict.keys to ((., .), .)!!!
                
                cur_cpt_estimate = self.CPTpredict(state, action)
                cpt_target = compute_CPT(data[(state, action)])
                
                _, loc = barberisFeaturize(state)
                lossgrad[loc, action] += (cpt_target - cur_cpt_estimate)
                
                '''
                for k in range(lossgrad.shape[0]):
                    for l in range(lossgrad.shape[1]):
                        _, loc = barberisFeaturize(state)
                        if k == loc and l == action:
                            lossgrad[k, l] += (cpt_target - cur_cpt_estimate) # loss = sq_diff
                '''
            
        elif not self.with_huber: # [MC = TD?]
            for state, action in data.keys():
                qtile_targets = data[(state, action)] # this contains all i's, BUT len=I only if TD???
                
                for i in range(self.support_size): # 1-1 <= i <= self.support_size-1 
                    cur_qtile_estimate = self.qtilepredict(state, action, i, is_even)
                    
                    tau_i = compute_cdf(i, self.support_size, pos = False) # = i/I
                    tau_i_next = compute_cdf(i+1, self.support_size, pos = False) # = (i+1)/I
                    midpoint_i = (tau_i + tau_i_next) / 2 
                    
                    _, loc = barberisFeaturize(state)
                    lossgrad[loc, action, i] += np.sum([midpoint_i - (z < cur_qtile_estimate) for z in qtile_targets])
                    
                    ### debug ###
                    init_t, init_g = state
                    if init_t == 0 and init_g in [0] and i in [1, 2, 12, 13, 24, 25, 36, 37, 47, 48]: # and action == 1:
                        if ((init_t, init_g), action, i, is_even) not in self.lossgrad_evolution.keys():
                            self.lossgrad_evolution[((init_t, init_g), action, i, is_even)] = [lossgrad[loc, action, i]]
                        else:
                            self.lossgrad_evolution[((init_t, init_g), action, i, is_even)] += [lossgrad[loc, action, i]]
                        
                        #if i == 48:
                        #    raise NotImplementedError
                    
                    '''
                    for k in range(lossgrad.shape[0]):
                        for l in range(lossgrad.shape[1]):
                            for m in range(lossgrad.shape[2]):
                                _, loc = barberisFeaturize(state)
                                if k == loc and l == action and m == i:
                                    lossgrad[k, l, m] += np.sum([midpoint_i - (z < cur_qtile_estimate) for z in qtile_targets])
                    '''        
        else:
            raise NotImplementedError
        
        return lossgrad
    
    def _update(self, data, is_even = None):
        # [MC, w/ & w/o qtile]: call after loop 'm' (move out);
        # [TD, w/ qtile]: call inside loop 'm';
        # data (= CPTdict): pairs of predictors-targets, predictor: (s, a), target: Z^{k+1}(s, a); 
        
        if is_even == None:
            self.loss_grad = self.compute_LossGrad(data, is_even) 
            
            if not self.with_quantile: # [MC-only];    
                self.cpt_theta = self.cpt_theta + self.lr * self.loss_grad 
            else: # [MC, w/ qtile] = [TD, w/ qtile];
                # TD vs MC only affects 'data' s.t. for TD,
                # np.sum @.computelossgrad performs E_j[r + gamma * q(s',a',j)]
                self.qtile_theta = self.qtile_theta + self.lr * self.loss_grad
        elif is_even == False:
            #raise NotImplementedError
            self.loss_grad_pos = self.compute_LossGrad(data, is_even)
            #print('@update...')
            #print('is_even:', is_even)
            #print('loss_grad:', self.loss_grad_pos.flatten())
            self.qtile_theta_pos = self.qtile_theta_pos + self.lr * self.loss_grad_pos
        else:
            #raise NotImplementedError
            self.loss_grad_neg = self.compute_LossGrad(data, is_even)
            #print('@update...')
            #print('is_even:', is_even)
            #print('loss_grad:', self.loss_grad_neg.flatten())
            self.qtile_theta_neg = self.qtile_theta_neg + self.lr * self.loss_grad_neg
  
################################### MAIN AGENT ################################

class QPG_CPT_SPSA():
    """
    Prashant's CPT Policy Optimization Algorithm w/ SPSA Policy.
    
    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized 
        discount_factor: Time-discount factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    def __init__(self, env, estimator_critic, estimator_policy, 
                 seed = None, discount_factor = 1.0, target_type = None, state_only = False,
                 support_size = 10, with_quantile = False, with_huber = False, 
                 param_init = 0.0, critic_lr = .1, empty_memory = True, 
                 step_size = 5.0, perturb_const = 1.9, actor_timescale = 1, 
                 use_buffer = 0, THETA_MIN = .1, THETA_MAX = 1.0):
        
        # env attr.
        self.env = env
        self.s0 = (0, env.init_wealth)
        self.horizon = self.env.T # self.T in barberisCasino
        
        self.seed = seed
        self.set_random_seed(seed = seed)
        
        # train (vf) attr.
        self.gamma = discount_factor
        self.CPT_dict = None
        self.ZList = None
        self.CPT_val = None
        self.CPT_true = None
        
        self.vf_dict_init()
        
        self.empty_memory = empty_memory
        self.critic_class = estimator_critic
        self.critic = self.critic_class(self.env, support_size = support_size,
                                        with_quantile = with_quantile,
                                        with_huber = with_huber,
                                        param_init = param_init, lr = critic_lr)
        self.target_type = target_type
        self.state_only = state_only
        
        # train (policy) attr.
        self.policy_class = estimator_policy
        self.policy = self.policy_class(self.env, step_size = step_size, 
                                        perturb_const = perturb_const, 
                                        THETA_MIN = THETA_MIN, THETA_MAX = THETA_MAX)
        self.actor_update_timescale = actor_timescale
        
        self.init_policy = None
        self.init_theta = None
        self.trained_policy = None
        self.trained_theta = None
        
        self.n_train_eps = None
        self.i_episode = None
        self.n_batch = None # batch size to obtain qtile (~vf) estimates
        self.use_buffer = use_buffer # alt: add .buffer_size
        self.logs = None # buffer logs containing Transitions
        
        # eval attr.
        self.eval_env = env
        self.n_eval_eps = None
        self.eval_freq = None
        self.best_mean_reward = -float("inf")
        self.verbose = None
        self.prev_hist_tuple = None
        self.prev_theta_histo = None
        
        self.stats = None
        self.SPSAwCritic_Check = None
        
        self.policy_dict_init()
    
    
    def vf_dict_init(self):
        self.CPT_dict = {}
        self.ZList = {} # w/o qtile, ZList = CPT_dict
        self.CPT_val = {}
        self.CPT_true = {}
        self.policy_val = {}
        #self.CPT_true_for_SPE = {}
        
        self.visitFreq = {}
        self.curVisitCounts = {}
        
        for t in range(self.env.observation_space.spaces[0].n):
            gain_supp = np.linspace(-self.env.bet * t, self.env.bet * t, t + 1)
            
            for id_ in range(self.env.observation_space.spaces[1].n):
                g = (id_ - self.env.T) * self.env.bet
                if g not in gain_supp:
                    continue
                
                state = (t, g)
                
                for action in range(self.env.action_space.n):
                    #self.CPT_val[(state, action)] = []
                    self.policy_val[(state, action)] = []
                    self.CPT_true[(state, action)] = []
                    self.visitFreq[(state, action)] = []
                    self.curVisitCounts[(state, action)] = 0
    '''
    def vf_dict_init(self):
        self.CPT_dict = {}
        self.ZList = {} # w/o qtile, ZList = CPT_dict
        self.CPT_val = {}
        self.CPT_true = {}'''
    
    def policy_dict_init(self):
        self.stats = {'mean_rewards': [],
                      'std_rewards': [],
                      'cpt_rewards': []}
        self.SPSAwCritic_Check = {'QR': [], 'SQ': [], 'True': []}
        
    def set_random_seed(self, seed = None):
        
        if seed is None:
            return
        
        # Seed python, numpy and tf random generator
        set_global_seeds(seed) 
        
        if self.env is not None:
            self.env.reset(seed = seed) # set env seed
            self.env.action_space.seed(seed) # seed act_space: useful when selecting random actions
        # self.action_space.seed(seed)
    
    def _get_n_batch(self, n, const = 1):
        
        pass
        '''
        # to clarify: m_n = m_0 * num_iter**(-v) formula, (ii) m_n = trajectory len @Prash's code = 500 or sth else
        if self.policy.is_even:
            pass
        
        n_iterations = math.ceil(n / 2)
        self.n_batch = math.pow(n_iterations, -.1) * const # gamma = .101, alpha = 1 -> v < -.202'''
    
    def evaluate_policy(self, init_sa = None, deterministic = True, 
                        return_episode_rewards = True, for_critic = False):
        
        episode_rewards, episode_lengths = [], []
        
        for i in range(self.n_eval_eps):
            
            state = self.eval_env.reset()
            action = None
            init_t, init_g = state
            
            if init_sa is not None:
                state, action = init_sa
                init_t, init_g = state # 0, 0 if not init_sa?
                self.eval_env.reset(init_time = init_t, init_wealth = init_g)
            
            episode_reward = 0.0
            episode_length = 0
            
            for t in itertools.count(): 
                
                if t < init_t:
                    continue
                
                if t > init_t or action == None:
                    action_probs = self.policy._predict(state, deterministic)
                    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                    
                next_state, reward, done, _ = self.eval_env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
                    
                state = next_state
                
            episode_rewards.append(episode_reward + init_g)
            episode_lengths.append(episode_length)
        
        ## Policy Update Diagnostics
        if not for_critic:
            #print('-----')
            #print('fixed policy? step_size:', self.policy.step_size)
            '''
            bins = np.linspace(-10, 10, num = 11) # {-10, -8, ..., 0, 2, .., 10}
            hist_tuple = np.histogram(np.array(episode_rewards), 
                                      bins = bins, density = True)
            plt.plot(hist_tuple[0], color = 'gray', 
                     alpha = .2 * self.i_episode/self.n_train_eps)
            print('---')
            '''
            if self.i_episode == 1:
                '''
                plt.plot(hist_tuple[0], color = 'k', label = 'theta0')
                '''
                
                init_policy = {}
                
                for t in range(self.eval_env.observation_space.spaces[0].n):
                    gain_supp = np.linspace(-self.eval_env.bet * t, self.eval_env.bet * t, t + 1)
                    
                    for id_ in range(self.eval_env.observation_space.spaces[1].n):
                        g = (id_ - self.env.T) * self.eval_env.bet
                        
                        # remove unvisited states e.g. (0, -10), at which trained policies are irrelevant;
                        if g not in gain_supp:
                            continue
                        
                        state = (t, g)
                        action_probs = self.policy._predict(state, deterministic)
                        
                        if np.argmax(action_probs) == 1:
                            action = '~gamble'
                        else:
                            action = '~exit'
                        
                        if t == self.eval_env.T:
                            action = '~exit (default)'
                            
                        init_policy[state] = (list(action_probs), action)
                        
                print('Check init_policy:\n', init_policy) 
                self.init_policy = init_policy
                self.init_theta = self.policy.theta
            
            else: #if self.i_episode == self.n_train_eps + 1:
                '''
                plt.plot(hist_tuple[0], color = 'r', label = 'theta*')
                '''
                final_policy = {}
                
                for t in range(self.eval_env.observation_space.spaces[0].n):
                    gain_supp = np.linspace(-self.eval_env.bet * t, self.eval_env.bet * t, t + 1)
                    
                    for id_ in range(self.eval_env.observation_space.spaces[1].n):
                        g = (id_ - self.env.T) * self.eval_env.bet
                        
                        # remove unvisited states e.g. (0, -10), at which trained policies are irrelevant;
                        if g not in gain_supp:
                            continue
                        
                        state = (t, g)
                        action_probs = self.policy._predict(state, deterministic)
                        
                        if np.argmax(action_probs) == 1:
                            action = '~gamble'
                        else:
                            action = '~exit'
                        
                        if t == self.eval_env.T:
                            action = '~exit (default)'
                            
                        final_policy[state] = (list(action_probs), action)
                        
                #print('Check train_policy:\n', final_policy) 
                self.trained_policy = final_policy
                self.trained_theta = self.policy.theta
            
        if return_episode_rewards:                                            
            return episode_rewards, episode_lengths
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        cpt_reward = compute_CPT(episode_rewards)
        
        if not for_critic:
            self.stats['mean_rewards'] += [mean_reward]
            self.stats['std_rewards'] += [std_reward]
            self.stats['cpt_rewards'] += [cpt_reward]
        
        return mean_reward, std_reward, cpt_reward #, true_qtiles
    
    def evaluate_actor_(self, is_even = None): # there is no critic! SPSA is actor-only approach;
        for state, action in self.CPT_true.keys():
            #self.CPT_val[(state, action)] += [[self.critic.CPTpredict(state, action, is_even)]]
            
            _, _, true_CPT = self.evaluate_policy(init_sa = (state, action), 
                                            deterministic = True, 
                                            return_episode_rewards = False,
                                            for_critic = True)
            self.CPT_true[(state, action)] += [[true_CPT]]
            self.visitFreq[(state, action)] += [[self.curVisitCounts[(state, action)]]]
            
            if self.trained_policy is not None:
                actProbs, _ = self.trained_policy[state]
                
                # IS THIS TRAINED_POLICY NOT RECORDING THE UPDATES?
                
            else:
                actProbs, _ = self.init_policy[state]
            self.policy_val[(state, action)] += [[actProbs[action]]]
    '''
    def evaluate_critic(self, is_even = None):
        
        for state, action in self.CPT_dict.keys():
            if self.state_only: 
                # only use 1 action-id to keep data for C(s)
                action = 1
                
            XList = self.ZList[(state, action)]
            CPT_target = compute_CPT(XList)
            cur_cpt_est = self.critic.CPTpredict(state, action, is_even)
            
            if (state, action) not in self.CPT_val.keys():
                self.CPT_val[(state, action)] = [[cur_cpt_est, CPT_target, is_even]] # @bf_critic: CPT_target
            else:
                self.CPT_val[(state, action)] += [[cur_cpt_est, CPT_target, is_even]]
            
            if self.state_only:
                _, _, true_CPT = self.evaluate_policy(deterministic = True,
                                                      return_episode_rewards = False,
                                                      for_critic = True)
                _, _, true_CPT_perturbed = self.evaluate_policy(deterministic = False,
                                                                return_episode_rewards = False,
                                                                for_critic = True)
            else:
                _, _, true_CPT = self.evaluate_policy(init_sa = (state, action), 
                                                deterministic = True, 
                                                return_episode_rewards = False,
                                                for_critic = True)
                _, _, true_CPT_perturbed = self.evaluate_policy(init_sa = (state, action), 
                                                          deterministic = False,
                                                          return_episode_rewards = False,
                                                          for_critic = True)
            
            if (state, action) not in self.CPT_true.keys():
                self.CPT_true[(state, action)] = [[true_CPT, true_CPT_perturbed, is_even]]
            else:
                self.CPT_true[(state, action)] += [[true_CPT, true_CPT_perturbed, is_even]]
    '''
    def train_policy(self, aggr_episodes, i_episode):
        
        ########################## Policy Evaluation ##########################
        ## i.e. Estimate CPT_val {C(s, a): s \in S, a \in A} given fixed policy
        
        XList = []
        
        for m in range(self.n_batch):
            
            accum_reward_ = np.sum([aggr_episodes[m][t].reward * self.gamma**t 
                                    for t in range(self.horizon + 1)
                                    if aggr_episodes[m][t] is not None])
            
            for t in range(self.horizon + 1):
                
                if aggr_episodes[m][t] == None:
                    continue
                
                state, action, reward, next_state, done = aggr_episodes[m][t]
                
                accum_reward_t = accum_reward_ 
                
                state = tuple(state)
                if self.state_only:
                    action = 1
                
                if (state, action) not in self.CPT_dict.keys():
                    self.CPT_dict[(state, action)] = [accum_reward_t]
                    self.ZList[(state, action)] = [accum_reward_]
                else:
                    self.CPT_dict[(state, action)] += [accum_reward_t]
                    self.ZList[(state, action)] += [accum_reward_]
                
            XList += [accum_reward_]
           
        ## [QRMC-PE (MC-loc)]
        #print('updating MC-loc critic...')
        self.critic._update(data = self.CPT_dict, is_even = self.policy.is_even)
        #self.evaluate_critic(is_even = self.policy.is_even)
        if self.empty_memory: # default = True
            self.CPT_dict = {}
            self.ZList = {}
        print('---')
        
        # IMPLEMENT ONLY W/ TD-LOC!!! @for t in range(self.horizon + 1);
        # [cf. QRTD script]
        # @SPERL: greedy, at all states (t, g) \in T x S;
        # ...
        
        # @SPERL: PG-Normal, at all states (t, g) \in T x S;
        # ...
        
        #################### Precomm-SPSA Policy Improvement ##################
        
        ## SPSA Diagnostics
        self.SPSAwCritic_Check['SQ'] += [[compute_CPT(XList), self.policy.is_even]] 
        
        _, _, grad_true_perturbed = self.evaluate_policy(deterministic = False,
                                                  return_episode_rewards = False,
                                                  for_critic = True)
        self.SPSAwCritic_Check['True'] += [[grad_true_perturbed, self.policy.is_even]]
        
        s0 = (0, 0)
        act_probs = self.policy._predict(s0, deterministic = False)
        if self.state_only:
            XList_qr = [self.critic.qtilepredict(s0, 1, i, self.policy.is_even) for i in range(self.critic.support_size)]
        else:
            XList_qr = []
            for i in range(self.critic.support_size):
                qtile_i = np.sum([self.critic.qtilepredict(s0, a, i, self.policy.is_even) * act_probs[a] for a in np.arange(len(act_probs))])
                XList_qr += [qtile_i]
        self.SPSAwCritic_Check['QR'] += [[compute_CPT(XList_qr), self.policy.is_even]]
        
        ## Switching +/-
        self.policy._get_is_even()
        
        n = (i_episode - 1) // (2 * self.n_batch)
        
        if (n+1) % self.actor_update_timescale == 0: # and .ss_const > 1e-16:
            # Gradient Estimation
            #print('Update vf_pair w/ C{+/-}...')
            if self.target_type == 'MC':
                CPT_val_s0 = compute_CPT(XList_qr)
            elif self.target_type == 'True':
                CPT_val_s0 = grad_true_perturbed
            else:
                CPT_val_s0 = compute_CPT(XList)
            vf_pair_idx = int(not self.policy.is_even)
            self.policy.vf_pair[vf_pair_idx] = CPT_val_s0
            #print('---')
            
            # Actor parameter update
            self.policy._update()
            
            # For next iteration w/ scheduling & +/- switching
            self._get_n_batch((n + 1)//self.actor_update_timescale)
            self.policy._get_step_size((n + 1)//self.actor_update_timescale)
            if self.policy.ss_const > 1e-16:
                self.policy._get_perturb_const((n + 1)//self.actor_update_timescale)
                self.policy._get_perturb_noise()
        
        self.policy._get_thetaSPSA()
        
        ##################### Precomm-PG Policy Improvement ###################
        # ...

        policy_loss = None 
        vf_loss = None
        
        return policy_loss, vf_loss
            
    def learn(self, n_train_eps, init_sa = None, n_batch = 5, n_eval_eps = 50, eval_freq = 10, verbose = True):
        
        self.n_train_eps = n_train_eps
        self.n_batch = n_batch
        self.n_eval_eps = n_eval_eps
        self.eval_freq = eval_freq
        
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        
        for i_episode in range(1, n_train_eps + 1 + 1): # end+1 for eval records
            
            self.i_episode = i_episode
            
            if (i_episode - 1) % eval_freq == 0:
                
                episode_rewards, episode_lengths = self.evaluate_policy(init_sa = init_sa) # add init_sa for precomm-dyn;
                mean_reward, std_reward, cpt_reward = np.mean(episode_rewards), np.std(episode_rewards), compute_CPT(episode_rewards)
                mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
                
                self.stats['mean_rewards'] += [mean_reward]
                self.stats['std_rewards'] += [std_reward]
                self.stats['cpt_rewards'] += [cpt_reward]
                
                if verbose > 0:
                    print("\Eval @ Episode {}/{}, "
                          "episode_reward={:.2f} +/- {:.2f}".format(i_episode, n_train_eps, mean_reward, std_reward))
                    print("cpt_reward={:.2f}".format(cpt_reward))
                    print("episode_length: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))
                    print("---")
                    
                if mean_reward > self.best_mean_reward:
                    if verbose > 0:
                        print('New best mean reward!')
                        print("---")
                    self.best_mean_reward = mean_reward
                
                self.evaluate_actor_()
            
            ## Start training
            if (i_episode - 1) % self.n_batch == 0:
                aggr_episodes = np.array([[None for _ in range(self.horizon + 1)] for _ in range(self.n_batch)])
                m_id = 0
            
            state = self.env.reset() # add init_sa for precomm-dyn;
            action = None
            init_t, init_g = state
            
            if init_sa is not None:
                state, action = init_sa
                init_t, init_g = state # 0, 0 if not init_sa?
                self.env.reset(init_time = init_t, init_wealth = init_g)
            
            for t in itertools.count():
                
                if t < init_t:
                    continue
                
                if t > init_t or action == None:
                    action_probs = self.policy._predict(state)
                    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                
                next_state, reward, done, _ = self.env.step(action)
                
                self.curVisitCounts[(tuple(state), action)] += 1
                
                aggr_episodes[m_id][t] = Transition(
                  state=state, action=action, reward=reward, next_state=next_state, done=done)
                
                if done:
                    break
                    
                state = next_state
            
            if m_id == self.n_batch - 1:
                #print('HERE')
                policy_loss, value_loss = self.train_policy(aggr_episodes, i_episode)
            
            m_id += 1  
            
            if self.logs is not None:
                self.logs = np.concatenate((self.logs, aggr_episodes), axis = 0)
            elif self.use_buffer:
                self.logs = aggr_episodes
            
        return self
'''
############################### EXHAUSTIVE SEARCH #############################

## Search for theta_init; to improve learning curve / 
## widen param search space; 
## Check exhaustive param;
p_win = .5
env = barberisCasino(p = p_win) 

min_cpt_val = float('inf')
max_cpt_val = -float('inf')

theta_cpt_smeans = []
theta_cpt_svars = []

for seed in range(64):
    n_batch = 50
    n_train_eps = 50 * (2*n_batch)
    step_size = 0.0 # check fixed policy
    
    perturb_const = 1.9 # does not matter, cz cares abt eval_policy(determ = True)
    target_type = 'True' #'MC' , 'True'
    # {'True' | None: SQ | 'MC', 'TD{0/1}{0/1}': QR variants}
    actor_timescale = 1
    n_eval_eps = 2000
    eval_freq = 2*n_batch
    
    THETA_MIN = .1
    THETA_MAX = 2.0
    model_ = QPG_CPT(env, estimator_critic = CPTCritic, estimator_policy = SPSAPolicy, 
                    seed = seed, target_type = target_type, state_only = True,
                    support_size = 100, with_quantile = True, with_huber = False, 
                    param_init = 0.0, critic_lr = .1, empty_memory = True, 
                    step_size = step_size, perturb_const = perturb_const,  
                    actor_timescale = actor_timescale, THETA_MIN = THETA_MIN, THETA_MAX = THETA_MAX)
    model_.learn(n_train_eps = n_train_eps, n_batch = n_batch, 
                 n_eval_eps = n_eval_eps, eval_freq = eval_freq)
    
    theta_cpt = model_.stats['cpt_rewards'][0]
    
    if theta_cpt < min_cpt_val:
        worst_theta = model_.policy.init_theta
        worst_seed = seed # 21
        min_cpt_val = theta_cpt # 6.13
        
    if theta_cpt > max_cpt_val:
        best_theta = model_.policy.init_theta
        best_seed = seed # 81
        max_cpt_val = theta_cpt # 8.12
        
    print('min, max:', min_cpt_val, max_cpt_val)
    
    theta_cpt_smeans += [np.mean(model_.stats['cpt_rewards'])]
    theta_cpt_svars += [np.var(model_.stats['cpt_rewards'])]

plt.figure()
plt.plot(theta_cpt_svars, theta_cpt_smeans, 'bo')
plt.plot([theta_cpt_svars[23]], [theta_cpt_smeans[23]], 'go', label = 'lowvar_init (23)') # cur best
plt.plot([theta_cpt_svars[14]], [theta_cpt_smeans[14]], 'ro', label = 'highvar_init (14)') # cur worst
plt.plot([theta_cpt_svars[50]], [theta_cpt_smeans[50]], 'co', label = 'chosen_init (50)')
plt.plot([theta_cpt_svars[18]], [theta_cpt_smeans[18]], 'yo', label = 'bad_init (18)')
plt.legend()
plt.xlabel('svars')
plt.ylabel('smeans') 

plt.title('CPT(theta) Eval-Policy: Choose theta w/ low svars')


filter_var = [i for i, x in enumerate(theta_cpt_svars) if x < 1]
filter_mean = [i for i, x in enumerate(theta_cpt_smeans) if x < 7] 

for i in range(64):
    if i in filter_var and i in filter_mean:
        print('seed:', i) # 50

# Theta @clip/init_theta_unif = [.1, 1] --> worst_seed = 21, range cpt_val (6.13, 8.12)
# Theta = [.1, 2.0] --> worst_seed = 14, range cpt_val (4.68, 8.24)
'''
#################################### RUNNER ###################################

# append itertools..
# ask JS run on hpc.. parallelize CPTParams or set walltime to inf? or ask JX help?

#CPTParams = list(itertools.product(np.round(np.linspace(1, .95, 2), 4), np.round(np.linspace(1, .65+.05, 4), 4),
#                              np.round(np.linspace(1, 1.4, 3), 4)))
'''
print(len(CPTParams))
if len(CPTParams) < 5:
    print('check CPTParams sweep!')
    raise ValueError()
'''   

CPTParams = [(.95, .5, 1.5)] #[(.88, .65, 2.25)] #[(.95, .5, 1.5)] 
pwinArr = [.36, .3, .42]  #[.62, .59, .64] #[.36, .3, .42] 
#np.linspace(.72, .48, 9) [.48, .72, .6, .66, .54] 
seedArr = range(5, 14)
stepSizeArr = [5.] #[5., 10.]

for alpha, rho1, lmbd in CPTParams:
    rho2 = rho1

    #################################### CPT ######################################
    def compute_cdf(k, q, pos):
        # compute cdf of k-th q quantile: 1 <= k <= q;
        
        if pos == True:
            return 1 - k/q
        else:
            return k/q
    
    def prob_weight(F, pos, rho1 = rho1, rho2 = rho2): 
        # @Prash: rho1 = .61, rho2 = .69 \in[0.3, 1]; assert w(F) monotonic in F;
        # @Barberis: rho1 = rho2 = .5 
    
        if pos == True:
            return F**rho1 / ((F**rho1 + (1-F)**rho1)**(1/rho1))
        else:
            return F**rho2 / ((F**rho2 + (1-F)**rho2)**(1/rho2))
    
    def utility(x, pos, alpha = alpha, lmbd = lmbd):
        # @Prash: alpha = .88 \in [0, 1], lmbd = 2.25 \in [1, 4];
        # @Barberis: alpha = .95, lmbd = 1.5 
        
        # compute u^{+/-}(x), where x is a quantile value: \in support(X);
        # X: accumulated reward r.v.;
        if pos == True:
            return x**alpha
        else:
            return -lmbd * (-x)**alpha 
            
    def compute_CPT(XList):
        
        XList = sorted(XList)
        
        m = len(XList) 
        
        CPT_val_pos = 0
        CPT_val_neg = 0
        
        for id_, x in enumerate(XList):
            
            i = id_ + 1
            
            if x >= 0:
                dF_i_pos = prob_weight(compute_cdf(i-1, m, True), True) - prob_weight(compute_cdf(i, m, True), True)
                CPT_val_pos += utility(x, True) * dF_i_pos # CPT_val_neg += 0
            
            elif x < 0:
                dF_i_neg = prob_weight(compute_cdf(i, m, False), False) - prob_weight(compute_cdf(i-1, m, False), False)
                CPT_val_neg += utility(x, False) * dF_i_neg # CPT_val_pos += 0
            
            else:
                print('@compute_CPT, x invalid!!!')
                raise ValueError
            
        # ver: u^{-} non-decreasing, formula @compute_CPT() changed to C_n^{+} + C_n^{-}
        return CPT_val_pos + CPT_val_neg 
    
    for p_win in pwinArr: # incl. [.6, .63]: 
        #np.linspace(.69, .63, 4): #[.5, .7, .3]:# [18]: #@p = .7: [18, 0]; @p = .5: [1, 0]; @p = .3: [4, 0]
        p_win = np.round(p_win, 4)  
        for step_size in stepSizeArr:
            for seed in seedArr:
                n_batch = 50
                perturb_const = 1.9 # 1e-16, 1.9 
                #step_size = 5.0 # 0.0 | 5.0
                beta = 1
                
                target_type = 'True' #'MC' , 'True' (default SPSA)
                # {'True' | None: SQ | 'MC', 'TD{0/1}{0/1}': QR variants}
                
                #p_win = .7
                env = barberisCasino(p = p_win) 
                
                actor_timescale = 1
                train_num = 300
                n_train_eps = train_num * (2*n_batch) * actor_timescale
                n_eval_eps = 2000
                eval_freq = 2*n_batch
                
                # Default = (.1, 1.) -- changed for learning curve demo [cf. Debug0]
                THETA_MIN = .1
                THETA_MAX = 2.0
                
                hyperparams = [seed, p_win, alpha, rho1, rho2, lmbd, n_batch, train_num, actor_timescale, target_type]
                
                ############################### SPSA ##############################
                envID = 'barberis'
                algoID = 'SPSA'
                runID = datetime.now().strftime("%d%m%Y%H%M%S")
                hyperparams_ = ['seed: ' + str(seed), 'p_win: ' + str(p_win),
                               'alpha: ' + str(alpha), 'delta: ' + str((rho1, rho2)), 'lmbd: ' + str(lmbd), 'B: ' + str(0), 
                               'step_size:' + str(step_size), 'perturb_const:' + str(perturb_const), 
                               #'support_size:' + str(support_size), 'explore_type:' + str(explore_type), 
                               'explore_rate (beta):' + str(beta), #'smoothen_thresh:' + str(smoothen_thresh),
                               #'by_episode:' + str(byEpisode),
                               'n_batch(K):' + str(n_batch), 'train_num:' + str(train_num), 'n_eval:' + str(n_eval_eps),
                               'THETA_MIN,MAX:' + str((THETA_MIN, THETA_MAX))]
                record_csv.record_params(hyperparams_, envID, algoID, runID)
                
                #fig = plt.figure(figsize = (20, 10))
                model = QPG_CPT_SPSA(env, estimator_critic = CPTCritic, estimator_policy = SPSAPolicy, 
                                seed = seed, target_type = target_type, state_only = True,
                                support_size = 100, with_quantile = True, with_huber = False, 
                                param_init = 0.0, critic_lr = .1, empty_memory = True, 
                                step_size = step_size, perturb_const = perturb_const,  
                                actor_timescale = actor_timescale, THETA_MIN = THETA_MIN, THETA_MAX = THETA_MAX)
                model.policy.beta = beta #1, 2, 5, 10
                model.learn(n_train_eps = n_train_eps, n_batch = n_batch, 
                            n_eval_eps = n_eval_eps, eval_freq = eval_freq)
                #plt.legend()
                #title_ = '(p, alpha, rho1, rho2, lmbd): ' + str((p_win, alpha, rho1, rho2, lmbd)) + ', \n seed: ' + str(model.seed) + ', ss_const: ' + str(model.policy.ss_const) + ', n_batch: ' + str(n_batch) + ', timescale: ' + str(actor_timescale)
                #title_ = 'SPSA Policy Evolution: Return Distribution \n' + title_
                #title_ += ', \n (target, w_qtile, I, critic_lr, empty_memory): ' + str((target_type, model.critic.with_quantile, model.critic.support_size, model.critic.lr, model.empty_memory))
                #plt.title(title_)
                #plt.savefig("./Plot/seed{}_p{}_alpha{}_rho{}_lmbd{}_beta{}_0.png".format(seed, p_win, alpha, rho1, lmbd, model.policy.beta), dpi=200)
                #plt.close(fig)
                
                cptMdpParams = [alpha, rho1, lmbd, p_win]
                if n_batch == 50 and step_size == 5.0 and train_num == 200:
                    record_csv.record_results(model, envID, algoID, runID, seed=seed, cptMdpID = cptMdpParams)
                else:
                    ablationParams = [n_batch, step_size]
                    record_csv.record_results(model, envID, algoID, runID, seed=seed, cptMdpID = cptMdpParams, ablationID = ablationParams)
                
                record_csv.record_csv(model, hyperparams)
                #plotting.plot_results(model, hyperparams)
                
                ###################################### DYN-IMPLEMENT ##############
                
                '''
                #s0 = (0, 0)
                
                #if s0 == (0, 0):
                #    record_csv.init_csv(model, hyperparams)
                #record_csv.record_csv(model, hyperparams)
                
                hyperparams = [seed, p_win, alpha, rho1, rho2, lmbd, n_batch, train_num, actor_timescale, target_type]
                
                dyn_env = barberisCasino(p = p_win) 
                s0 = tuple(dyn_env.reset())
                print('reset check, env.time:', dyn_env.time)
                
                model = QPG_CPT_SPSA(env, estimator_critic = CPTCritic, estimator_policy = SPSAPolicy, 
                                seed = seed, target_type = target_type, state_only = True,
                                support_size = 100, with_quantile = True, with_huber = False, 
                                param_init = 0.0, critic_lr = .1, empty_memory = True, 
                                step_size = step_size, perturb_const = perturb_const,  
                                actor_timescale = actor_timescale, THETA_MIN = THETA_MIN, THETA_MAX = THETA_MAX)
                #model_for_SPE = QPG_CPT()
                #model_for_Precomm = QPG_CPT()
                print('reset check, env.time:', dyn_env.time)
                for t in itertools.count():
                    
                    print('@dyn-implement, s0:', s0)
                    model.learn(n_train_eps = n_train_eps, init_sa = (s0, None), n_batch = n_batch, 
                                n_eval_eps = n_eval_eps, eval_freq = eval_freq)
                    #model_for_SPE.compute_SPE(n_eval_eps = 2000, init_sa = (s0, None))
                    #model_for_Precomm.assign_Precomm(n_eval_eps = 2000, init_sa = (s0, None))
                    print('reset check, env.time:', dyn_env.time)
                    
                    if s0 == (0, 0):
                        record_csv.init_csv(model, hyperparams, s0 = s0, dyn = True)
                    record_csv.record_csv(model, hyperparams, s0 = s0, dyn = True)
                    
                    action_probs = model.policy._predict(s0, deterministic = True)
                    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                    print('a0, exit?', action)
                    
                    next_state, reward, done, _ = dyn_env.step(action)
                    
                    if done:
                        break
                    
                    s0 = next_state
                    
                    model.vf_dict_init()
                    model.policy_dict_init()
                '''
                