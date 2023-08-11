import numpy as np
import pandas as pd
import gym
from gym import spaces

####################### MULTI-ASSET, DATA IMPORT ENV ##########################
#preprocessed_path = "https://raw.githubusercontent.com/mengmengGuanZH/FinalYearProject/main/done_data.csv"
#data = pd.read_csv(preprocessed_path, index_col=0)
#print(data.head())
#print(data.size)

'''
if os.path.exists(preprocessed_path):
    data = pd.read_csv(preprocessed_path, index_col=0)
else:
    # if no preprocessed data, then data preprocessing pipeline: load data -> adj price -> add tech indicators
    data = preprocess_data()
    data = add_turbulence(data)
    data.to_csv(preprocessed_path)

print(data.head())
print(data.size)

# 2015/10/01 is the date that validation starts
# 2016/01/01 is the date that real trading starts
# unique_trade_date needs to start from 2015/10/01 for validation purpose
unique_trade_date = data[(data.datadate > 20151001)&(data.datadate <= 20200707)].datadate.unique()
print(unique_trade_date)
run_ensemble_strategy(df=data, 
                      unique_trade_date= unique_trade_date, 
                      rebalance_window = rebalance_window, 
                      validation_window=validation_window)

def data_split(df, start, end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.datadate >= start) & (df.datadate < end)]
    data=data.sort_values(['datadate','tic'],ignore_index=True)
    #data  = data[final_columns]
    data.index = data.datadate.factorize()[0]
    return data

rebalance_window = 63 # unit: months, data: daily

train = data_split(df, start=20090000, end=unique_trade_date[i - rebalance_window - validation_window])
env_train = DummyVecEnv([lambda: StockEnvTrain(train)])

validation = data_split(df, 
                        start=unique_trade_date[i - rebalance_window - validation_window],
                        end=unique_trade_date[i - rebalance_window])
env_val = DummyVecEnv([lambda: StockEnvValidation(validation,
                                                  turbulence_threshold=turbulence_threshold,
                                                  iteration=i)])
trade_data = data_split(df, start=unique_trade_date[iter_num - rebalance_window], end=unique_trade_date[iter_num])
env_trade = DummyVecEnv([lambda: StockEnvTrade(trade_data,
                                               turbulence_threshold=turbulence_threshold,
                                               initial=initial,
                                               previous_state=last_state,
                                               model_name=name,
                                               iteration=iter_num)])
'''

class shiStockEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is the environment described in [Shi15] paper
    """

    metadata = {'render.modes': ['console']}
    # mu = .2, 
    def __init__(self, mu = .2, sig = .4, r = 0, time_unit = 'day', init_wealth = 1, T = 3):
        super(shiStockEnv, self).__init__()
        
        self.STOCK_DIM = 1
        
        if self.STOCK_DIM > 1:
            print('Modif .action_space needed')
            raise NotImplementedError()
        
        # ==== Dynamics/Reward ====
        # 1. SDE SIMULATOR (make daily, from annual dPt = mu dt + sig dWt)
        # size (r, c) = (rebalance_window * 30 days, 1 + STOCK_DIM)
        
        self.T = T # rebalance_window * 30 (no of decision periods, keep unit to 'daily')
        self.init_wealth = init_wealth
        self.time_unit = time_unit # 'mth', 'ann'; if changed, change params!
        self.r = r # self.r_annum * self.time_step
        self.mu = mu
        self.sig = sig
        self.mu_vec = [mu for j in range(self.STOCK_DIM)]
        self.sig_vec = [sig for j in range(self.STOCK_DIM)]
        #self.corr_daily =
        
        # 2. MARKET DATA (daily dynamics)
        # data = 
        
        # ==== Action ====
        # Action: wealth amount to invest in each risky asset
        # Normalized_action: fraction of total wealth (discretized)
        # 0: 0, 1: .1, 2: .2, ..., 10: 1
        # (LB, UB) = (0, 1)
        self.n_actions = 5
        self.action_space = spaces.Discrete(self.n_actions + 1)
        
        # check init_wealth = 1 (vs 0 @Barberis)
        # ...
        
        # ==== State ====
        # v1, (time, wealth)
        # assume wealth (LB, UB) = (-5, 5)
        self.n_tiles = 10
        self.observation_space = spaces.Tuple((spaces.Discrete(self.T + 1), spaces.Discrete(self.n_tiles,)))
        
        # v2,
        # state[0]: INCLUDE TIME!!!!
        # state[0]: cash balance (bond?)
        # state[1:1+STOCK_DIM]: risky assets held (in dollar)
        # state[1+STOCK_DIM:1+2*STOCK_DIM]: (normalized) price of assets (at each .step, update!)
        # state[1+2*STOCK_DIM:]: (optionally) other indicators
        #self.observation_space = spaces.Box(low=-1, high=1,
        #                                    shape=(self.N + 2,), dtype=np.float32)
        
        self.time = None
        self.wealth = None    
        self.prev_time = None
        self.prev_wealth = None   
        self.state = None
        self.path = None
        
    def seed(self):
        pass
    
    def step(self, actions_normalized):
        
        if self.time >= self.T:
            done = True
            
            self.prev_time, self.prev_wealth = self._get_obs()
            
            self.wealth += 0 # no longer evolves
            self.time += 1
            
            reward = self.wealth - self.prev_wealth
            
        else: 
            # cz at t = T-1, we still want to generate action 
            # -> apply step(act_T-1) return done = True, since otherwise 
            # we will generate a_T
            done = False
            
            self.prev_time, self.prev_wealth = self._get_obs()
            
            # optionally, add constraints (e.g. no shorting) to transform
            # 'actions_normalized', i.e. single-asset: a0; multi-asset: [a0, a1, ..., aJ]
            bond_invest = (1 - np.sum(actions_normalized)) * self.prev_wealth
            self.state[0] = bond_invest * (1 + self.r)
            
            stock_invest = actions_normalized * self.prev_wealth
            
            # .path[0][.time + 1]
            # dim = r, c
            # .path[:, j]
            # path is an array, price_incr must be a vector of length STOCK_DIM
            # ...
            
            price_increments = self.path[self.time + 1, :] / self.path[self.time, :] 
            # daily, .time_step alr accounted
            
            self.state[1:1+self.STOCK_DIM] = np.dot(stock_invest, price_increments)
            
            self.wealth = np.sum(self.state[:1+self.STOCK_DIM])
            self.time += 1 # daily
            
            # optionally, append more info to .state and update by
            # self.state[1+STOCK_DIM:1+2*STOCK_DIM] = self.path[self.time]
            
            reward = self.wealth - self.prev_wealth
        
        return self._get_obs(), reward, done, {}
    
    def _get_obs(self):
        
        return self.time, self.wealth
    
    def reset(self, init_time = None, init_wealth = None, seed = None, return_info = False):
        """
        Important: the observation must be a numpy array
        :return: (np.array) 
        """
        super().seed(seed)
        
        self.time = 0
        self.wealth = self.init_wealth
        
        self.path = self.get_stock_prices()
        self.state = [self.init_wealth] + [0]*self.STOCK_DIM #+ self.path.loc[self.time, :]
        
        # case II: self.time, self.wealth = self._sample_obs
        # may want to take care of wealth domain changes with t
        if init_time is not None and init_wealth is not None:
            self.time = init_time
            self.wealth = init_wealth
        
        return np.array([self.time, self.wealth], dtype = np.float32)
    
    def get_stock_prices(self, df = None):
        
        if df == None:
            # 1. SDE SIMULATOR (make daily, from annual dPt = mu dt + sig dWt)
            # size (r, c) = (rebalance_window * 30 days, STOCK_DIM)
            
            # use (t, gt) & longSPERL marketenv (DIRECT REUSE????)
            # w continuous SA spaces, can we still do tabular and naive discretization?
            # alt, use linFA from semi-analyt @Shi15
            
            path = [[1] for j in range(self.STOCK_DIM)]
            
            for j in range(self.STOCK_DIM):
                
                P_cur = 1
                for i in range(1, 1+int(self.T)):
                    P_next = P_cur + P_cur * np.random.normal(self.mu_vec[j], self.sig_vec[j])
                    #(self.mu * self.dt + self.sig * np.random.normal(0, math.sqrt(self.dt)))
                
                    path[j] += [P_next]
                    P_cur = P_next
                    
        else:
            # 2. MARKET DATA (daily dynamics, normalized w P_0 = 1)
            # data = data_split(...)
            raise NotImplementedError()
            
        return np.transpose(path)

    
    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

