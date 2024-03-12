import math
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from casadi import *
from casadi.tools import *
from env.template_model import template_model
from env.template_simulator import template_simulator



import ipywidgets as widgets

# limites y condiciones iniciales del simulador

temp_range = 2 
       
states = { 
    'm_W':{"pos":0, "low":0 , "high":4e5, "init": 10000.0 },
    'm_A':{"pos":1, "low":0 , "high":1e4, "init":  853.0 },
    'm_P':{"pos":2, "low":26.0 , "high":1e3, "init":  26.5},
    'T_R':{"pos":3, "low":298 , "high":400.0, "init": 363.15 - temp_range/2 },
    'T_S':{"pos":4, "low":298 , "high":400.0, "init":  90.0 + 273.15},
    'Tout_M':{"pos":5, "low":298.0 , "high":400.0, "init":  90.0 + 273.15},
    'T_EK':{"pos":6, "low":288.0 , "high":400.0, "init": 35.0 + 273.15 },
    'Tout_AWT':{"pos":7, "low":298.0 , "high":400.0, "init": 35.0 + 273.15 },
    'accum_monom':{"pos":8, "low":0 , "high":30000, "init":  300.0},
    'T_adiab':{"pos":9, "low":0 , "high":382.15, "init":  378.047} #FIXME
}




###Acciones (temp en kelvin)
actions = {
    'm_dot_f':{'pos':0, "low":0, "high":3.0e4} ,
    'T_in_M':{'pos':1, "low":333.15, "high":373.15} ,
    'T_in_EK': {'pos':2, "low":333.15, "high":373.15}
}


class DoMPC_Poly_env(gym.Env):
    """
    Custom Environment that follows gym interface.

    This is a custom environment for a reinforcement learning task, 
    which adheres to the OpenAI Gym interface.

    Attributes:
    -----------
    render : bool
        If True, the environment will be rendered. Default is True.
    ep_max_len : int
        Maximum length of an episode. Default is 1000.
    render_mode : str
        Mode for rendering the environment. Default is "human".
    penalties : list
        List of penalties to apply. Options are:
            1 - action penalty,
            2 - adab penalty,
            3 - reactor temperature penalty.
        An empty list implies no penalties. Examples: [1,2], [3], [].
    hard_constraint : bool
        If True, the episode ends when temperatures go outside specified limits
        (e.g., T_adab > 382.15 or T_R ± desired_temp ± 2º).
        Recommended to be False for testing purposes.

    Methods:
    --------
    __init__(render=True, ep_max_len=1000, render_mode="human", penalties=[], hard_constraint=True)
        Initializes the custom environment.
    """
    def __init__(self,
                 render=True,
                 ep_max_len=1000,
                 render_mode="human",
                 penalties=[],
                 hard_constraint=True,
                 randomize = False):
        super(DoMPC_Poly_env, self).__init__()

        p_d = ["Action Penalty", "Adab Penalty", "Reactor Temp Penalty"]

        self.metadata = {'render.modes': ["human"]}

        print("Penalties added:",[p_d[x] for x in penalties] )
        print("Hard constraint:",hard_constraint )
        print("Ep length:",ep_max_len )
        print("Randomize init?:",randomize )

        # Set the initial state of the controller and simulator:
        self.delH_R_real = 950.0
        self.c_pR = 5.0
        self.render_mode = render_mode

        self.randomize = randomize        

        self.penalties = penalties
        self.hard_constraint = hard_constraint

        self.ep_max_len = ep_max_len
        self.last_reward = 0
        self.time_step = 0

        # Define the scaling factors for each component of the reward function
        self.k1 = 0.001  # Scale for production reward
        self.k2 = 5.0  # Scale for safety penalty
        self.k3 = 10.0  # Scale for operational constraint penalty
        self.k4 = 1.0  # Scale for input change penalty
        # Define the target temperature range
        self.desired_temp = 363.15  # Desired reaction temperature in Kelvin
        self.temp_range = 2.0       

        self.list_actions_raw = []
        self.renderize = render          

        # Define action and observation space
        self.action_space = gym.spaces.Box(low=np.array([x["low"] for x in actions.values()]), # Lower bounds for each action
                                           high=np.array([x["high"] for x in actions.values()]), # Upper bounds for each action
                                            dtype=np.float32
                                            )
        self.observation_space = gym.spaces.Box(
            low=np.array([x["low"] for x in states.values()]),  # Lower bounds (0 for each element)
            high=np.array([x["high"] for x in states.values()]),  # Upper bounds (infinity for each element)
            dtype=np.float64  # Data type of each element
        )
            
        self.reset()        

    def step(self, action):      
        if action.shape != (3, 1):
            # Reshape the array to (3, 1) if it's not already in that shape
            action = action.reshape(3, 1)
        u0 = action

        try:
            y_next = self.simulator.make_step(u0)

        except Exception as e:
                print(e)
                return y_next, -1000, True, False, {"info": "Ended with error: "+str(e)}

        # Example of how to update the data for plotting
        self.data['_x']['T_R'].append(y_next[states["T_R"]["pos"]])  # Update with the new value of T_R
        self.data['_x']['accum_monom'].append(y_next[states["accum_monom"]["pos"]])  # Update with the new value of accum_monom
        self.data['_x']['m_P'].append(y_next[states["m_P"]["pos"]])  # Update with the new value of accum_monom
        self.data['_u']['m_dot_f'].append(u0[0])  # Update with the new value of m_dot_f
        self.data['_u']['T_in_M'].append(u0[1])  # Update with the new value of T_in_M
        self.data['_u']['T_in_EK'].append(u0[2])  # Update with the new value of T_in_EK
        self.data['_x']['T_adiab'].append(y_next[states["T_adiab"]["pos"]])  # 
        
        reward = self.get_reward(y_next, action, self.last_action)
        self.data["REWARD"].append(reward[0])
        self.data["R_prod"].append(reward[1])
        self.data["P_safety"].append(reward[2])
        self.data["P_temp"].append(reward[3])
        self.data["P_input"].append(reward[4])
        self.last_reward = reward[0]
        self.last_action = action        
        self.list_actions_raw.append([action, reward])

        done = reward[5]

        self.time_step += 1

        if(self.time_step>self.ep_max_len):
            truncated = True
        else:
            truncated = False

        #return y_next.reshape(-1), reward[0], done, truncated, {"timestep":self.time_step}

        return y_next.reshape(-1), reward[0], done, truncated, {"timestep":self.time_step}

    def get_reward(self, state, action, previous_action):        
        
        done = False
        R_prod, P_safety, P_temp, P_input = 0, 0 ,0 ,0

        # Extract necessary state variables

        m_P = state[states['m_P']["pos"]][0]  # Polymer mass
        T_adiab =state[states['T_adiab']["pos"]][0]  # Adiabatic temperature
        T_R = state[states['T_R']["pos"]][0]  # Reactor temperature
        self.last_T_R = T_R

        # Penalty for input changes
        if(len(self.last_action)>0):
            input_change_penalty = (0.002 * abs(action[0] - previous_action[0]) + \
                            0.004 * abs(action[1] - previous_action[1]) + \
                            0.002 * abs(action[2] - previous_action[2]))[0]
        else:
            input_change_penalty = 0

        # REWARDS and PENALTIES:
        # Reward for production
        R_prod = math.exp(-(20680 - m_P)*(20680 - m_P)/1e5)

        # Reward for respecting safety constraint T_adab
        R_safety = (T_adiab <= 382.15)

        # Penalty for respecting operational constraints
        # R_temp = math.exp(-(T_R - self.desired_temp)*(T_R - self.desired_temp)/1e2)
        R_temp = self.sigmoid(T_R - (self.desired_temp - temp_range)) - self.sigmoid(T_R - (self.desired_temp + temp_range))

        # Penalty for rapid input changes (assuming previous control input values are stored)
        P_input = self.k4 * input_change_penalty

        # Penalty for safety constraint violation T_adab
        P_safety = self.k2 * (T_adiab > 382.15)

        # Penalty for deviation from operational constraints
        P_temp = self.k3 * max(0, abs(T_R - self.desired_temp) - self.temp_range)
        
        
        ############## TOTAL REWARD ############################
        # Combine rewards and penalties (ajustar según objetivo)

        total_reward = R_prod + R_safety + R_temp
        if(0 in self.penalties): total_reward -= P_input  
        if(1 in self.penalties): total_reward -= P_safety  
        if(2 in self.penalties): total_reward -= P_temp  

        ### SAFETY FIRST CONDITION!:###
        if( ((abs(P_safety) > 0) or ( abs(P_temp) > 0)) and self.hard_constraint):
            done = True

        return total_reward, R_prod, R_safety, R_temp, P_input, done

    def reset(self, init_state=None, **kwargs):
        self.data = {'_x': {'T_R': [], 'accum_monom': [],'m_P':[],'T_adiab':[]}, '_u': {'m_dot_f': [], 'T_in_M': [], 'T_in_EK': []}, 'REWARD':[], 'R_prod': [], 'P_input': [], 'P_safety': [], 'P_temp': []}              

        # self.simulator.reset_history()
        self.model = template_model()
        self.simulator = template_simulator(self.model)

        self.time_step = 0
        self.last_action = []
        
        self.last_T_R = states['T_R']["init"] 
        
        # Reset the simulator to the initial state
        if init_state is not None:
            init_state = init_state.squeeze()
            self.simulator.x0['m_W'] = init_state[0]
            self.simulator.x0['m_A'] = init_state[1]
            self.simulator.x0['m_P'] = init_state[2]
            self.simulator.x0['T_R'] = init_state[3]
            self.simulator.x0['T_S'] = init_state[4]
            self.simulator.x0['Tout_M'] = init_state[5]
            self.simulator.x0['T_EK'] = init_state[6]
            self.simulator.x0['Tout_AWT'] = init_state[7]
            self.simulator.x0['accum_monom'] = init_state[8]
            self.simulator.x0['T_adiab'] = init_state[9]
            return init_state, {"timestep":0}
        else:
            self.simulator.x0['m_W'] = states['m_W']["init"] 
            self.simulator.x0['m_A'] = states['m_A']["init"]
            self.simulator.x0['m_P'] = states['m_P']["init"] 
            self.simulator.x0['T_R'] = states['T_R']["init"] 
            self.simulator.x0['T_S'] = states['T_S']["init"] 
            self.simulator.x0['Tout_M'] = states['Tout_M']["init"]
            self.simulator.x0['T_EK'] = states['T_EK']["init"] 
            self.simulator.x0['Tout_AWT'] =  states['Tout_AWT']["init"] 
            self.simulator.x0['accum_monom'] = states['accum_monom']["init"] 
            self.simulator.x0['T_adiab'] = self.simulator.x0['m_A']*self.delH_R_real/((self.simulator.x0['m_W'] + self.simulator.x0['m_A'] + self.simulator.x0['m_P']) * self.c_pR) + self.simulator.x0['T_R']
        
            if(self.randomize):
                self.simulator.x0['m_W'] = states['m_W']["init"] * np.random.uniform(0.9, 1.1) 
                self.simulator.x0['m_A'] = states['m_A']["init"] * np.random.uniform(0.9, 1.1) 
                self.simulator.x0['m_P'] = states['m_P']["init"] * np.random.uniform(0.9, 1.1) 
                self.simulator.x0['T_R'] = states['T_R']["init"] * np.random.uniform(0.95, 1.05) 
                self.simulator.x0['T_S'] = states['T_S']["init"] * np.random.uniform(0.95, 1.05)
            

            return np.array([states['m_W']["init"] , 
                            states['m_A']["init"],
                            states['m_P']["init"], 
                            states['T_R']["init"], 
                            states['T_S']["init"], 
                            states['Tout_M']["init"], 
                            states['T_EK']["init"], 
                            states['Tout_AWT']["init"], 
                            states['accum_monom']["init"], 
                            states['T_adiab']["init"] ]) , {"timestep":0}        
    



    def print_step(self, mode="human", close=False):        ### adaptación para poder plotear en stable baselines
            
        times = list(range(len(self.data["_x"]["m_P"])))

        if(len(times)>2):

            self.fig, self.ax = plt.subplots(12, sharex=True, figsize=(16, 12))          

            # Plot each variable
            self.ax[0].plot(times, self.data['_x']['m_P'], label='m_P', color = "grey")
            self.ax[1].plot(times, self.data['_x']['T_R'], label='T_R', color = "grey")
            self.ax[2].plot(times, self.data['_x']['accum_monom'], label='accum_monom',  color = "grey")
            self.ax[3].plot(times, self.data['_x']['T_adiab'], label='T_adiab',  color = "grey")
            self.ax[4].plot(times, self.data['_u']['m_dot_f'], label='m_dot_f')
            self.ax[5].plot(times, self.data['_u']['T_in_M'], label='T_in_M')
            self.ax[6].plot(times, self.data['_u']['T_in_EK'], label='T_in_EK')
            self.ax[7].plot(times, self.data['REWARD'], label='REWARD', color="orange")
            self.ax[8].plot(times, self.data['R_prod'], label='R_prod', color="blue")
            self.ax[9].plot(times, self.data['P_input'], label='P_input', color="purple")
            self.ax[10].plot(times, self.data['P_safety'], label='P_safety', color="red")
            self.ax[11].plot(times, self.data['P_temp'], label='P_temp', color="green")

            self.ax[1].axhline(self.desired_temp + self.temp_range, linestyle = "dotted", color = "green")
            self.ax[1].axhline(self.desired_temp - self.temp_range, linestyle = "dotted", color = "green")
            self.ax[3].axhline(382.15, linestyle = "dotted", color = "green")
            self.ax[7].axhline(0, linestyle ="dotted", color = "grey")

            # Set labels
            self.ax[0].set_ylabel('(S): m_P')
            self.ax[1].set_ylabel('(S): T_R [K]')
            self.ax[2].set_ylabel('(S): acc. monom')
            self.ax[3].set_ylabel('(S): T_adiab')
            self.ax[4].set_ylabel('(A): m_dot_f')
            self.ax[5].set_ylabel('(A): T_in_M [K]')
            self.ax[6].set_ylabel('(A): T_in_EK [K]')
            self.ax[7].set_ylabel('REWARD')
            self.ax[8].set_ylabel('R_prod')
            self.ax[9].set_ylabel('P_input')
            self.ax[10].set_ylabel('P_safety')
            self.ax[11].set_ylabel('P_temp')
            self.ax[11].set_xlabel('time')
            # Align y-labels
            self.fig.align_ylabels()
            
            # Show the plot
            plt.show()
    
    def render(self, mode='human'):

        if mode not in self.metadata['render.modes']:
            raise NotImplementedError(f"Render mode '{mode}' is not supported.")
        pass
        
        if(mode == "human"): 
            self.print_step(mode)            

    def close(self):
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-10*x))