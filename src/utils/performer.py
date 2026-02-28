'''
© Edgar Andrade 2023
Email: edgar.andrade@urosario.edu.co

-----------------------------------------------
Class to run, renderize, train and test agents
over environments.
-----------------------------------------------
'''
import numpy as np
import seaborn as sns
import gymnasium as gym
import matplotlib.pyplot as plt
from os import path
from pathlib import Path
from copy import deepcopy
from inspect import currentframe, getframeinfo
from typing import Dict, Tuple, Optional, Union

import environments.envs as E
import agents.agentsCS as ApproxQ
import agents.agentsPG as ApproxP
import agents.agentsNN as ApproxNN
import agents.TableAgents as TableA
#import agents.agentsPG as ApproxPG
from utils.interaction import Episode, Experiment
from utils.plot_utils import Plot

SCRIPT_PATH = Path.cwd() / Path('..').resolve()
OWN_ENV_LIST = ['ABC', 'OneDGridWroldEnv', 'GridworldEnv', 'PitLaberynth']


class Performer :
    '''
    Class to train and run an agent in an environment.
    '''
    def __init__(self,\
                env_name:str,\
                env_parameters:Dict[str, any],\
                state_interpreter:any,\
                agent_name:str,\
                agent_parameters:Dict[str,any]
                ) -> None:
        self.env_name = env_name
        assert(isinstance(env_parameters, dict))
        self.env_parameters = env_parameters
        self.state_interpreter = state_interpreter
        if '.' in agent_name:
          agent_class, agent_name = agent_name.split('.')
        else:           
           agent_class = 'TableAgents'
        self.agent_name = agent_name
        self.agent_class = agent_class
        self.agent_parameters = agent_parameters
        self.deep = None
        self.consolidate_folders()
        self.data = None

    def consolidate_folders(self):
        self.file_name = f'{self.agent_name}_in_{self.env_name}'
        self.image_folder = SCRIPT_PATH / Path('images', self.file_name)
        self.image_folder.mkdir(parents=True, exist_ok=True)
        self.data_folder = SCRIPT_PATH / Path('data', self.file_name)
        self.data_folder.mkdir(parents=True, exist_ok=True)
        self.model_folder = SCRIPT_PATH / Path('models', self.file_name)
        self.model_folder.mkdir(parents=True, exist_ok=True)
        self.video_folder = SCRIPT_PATH / Path('videos', self.file_name)
        self.video_folder.mkdir(parents=True, exist_ok=True)
        self.extension = '.pt' if self.deep else '.json'
        self.file_model = path.join(self.model_folder, f'{self.file_name}{self.extension}')
        self.file_csv = path.join(self.data_folder, f'{self.file_name}.csv')
        self.file_png = path.join(self.image_folder, f'{self.file_name}.png')
        self.file_losses = path.join(self.image_folder, f'{self.file_name}_losses.png')
        self.file_test = path.join(self.image_folder, f'{self.file_name}_test.png')
        self.file_test_csv = path.join(self.data_folder, f'{self.file_name}_test_csv.csv')
        self.file_compare_hist = path.join(self.image_folder, f'comparison_hist.png')
        self.file_compare_rew = path.join(self.image_folder, f'comparison_rew.png')

    def load_env(self, render_mode):
        '''
        Load environment. Render mode is different 
        for training (None) than for running (rgb_array). Render
        mode can only be set when instantiating the environment.
        '''
        if self.env_name in OWN_ENV_LIST:
            exec(f'self.environment = E.{self.env_name}(**self.env_parameters)')              
            self.environment.render_mode = render_mode
        elif self.env_name == 'SpecialCarRacing':
          lap_complete_percent = self.env_parameters['lap_complete_percent']
          initial_round = self.env_parameters['initial_round']
          self.environment = gym.make('CarRacing-v2',\
                          render_mode=render_mode,\
                          continuous=False,
                          lap_complete_percent = lap_complete_percent,\
                          domain_randomize = False)
          # Redefine step() method            
          special_step = self.environment.step
          def step(action):
              result = self.environment.special_step(action)
              state = result[0]
              reward = result[1]
              done = result[2]
              # Modificación de reward y done
              reward = -1
              finished = (self.environment.tile_visited_count)/len(self.environment.track) > self.environment.lap_complete_percent
              if finished:
                 reward = 10
              elif done:
                  reward = -10
              done = done or finished
              return (state,reward,done,result[3],result[4])
          # Redefine reset() method               
          special_reset = self.environment.reset
          def reset():
              state = self.environment.special_reset(options={"randomize":True}, seed=1)
              # Avoid 60 initial rounds with camera centering on car
              for i in range(60):
                  result = self.environment.step(3)
              # Make car press gas for some rounds
              for i in range(initial_round):
                  result = self.environment.step(3)
              # Make car full stop
              for i in range(10):
                  result = self.environment.step(4)
              return result[0]
          setattr(self.environment, 'special_step', special_step)
          setattr(self.environment, 'step', step)
          setattr(self.environment, 'special_reset', special_reset)
          setattr(self.environment, 'reset', reset)
        else:
          try:
            self.environment = gym.make(self.env_name, 
                                render_mode=render_mode,
                                **self.env_parameters)
          except:
            raise Exception(f'Environment {self.env_name} unknown. Please modify the method Performer.load_env() inside the module utils/performer to include environment {self.env_name}.')

    def load_agent(self, from_file:bool=False):
        '''
        Load agent from name
        '''
        if self.agent_class == 'TableAgents':
          line = f'self.agent = TableA.{self.agent_name}(self.agent_parameters)'
          self.deep = False
        elif self.agent_class == 'agentsCS':
          line = f'self.agent = ApproxQ.{self.agent_name}(self.agent_parameters)'
          self.deep = False
        elif self.agent_class == 'agentsNN':
          line = f'self.agent = ApproxNN.{self.agent_name}(self.agent_parameters)'
          self.deep = True
        # elif self.agent_class == 'agentsPG':
        #   line = f'self.agent = ApproxP.{self.agent_name}(self.agent_parameters)'
        else:
           raise Exception(f'Agent class {self.agent_class} is unknown!')
        exec(line)
        if from_file:
          print(f'Loading agent from {self.file_model}')
          self.agent.load(file=self.file_model)
    
    def save_agent(self):
        try:
            self.agent.save(file=self.file_model)
        except Exception as e:
            print('\n\tAn error occurred:\n\t', e,'\n')
            pass

    def shutdown_agent_exploration(self) -> Tuple[float, np.ndarray]:
        backup_epsilon = deepcopy(self.agent.epsilon)
        if hasattr(self.agent, 'policy'):
          backup_policy = deepcopy(self.agent.policy)
          self.agent.epsilon = 0
          for s in range(self.agent.nS):
            self.agent.update_policy(s)
        else:
          backup_policy = None
        return backup_epsilon, backup_policy

    def run(self, 
            from_file:bool=False,
            no_exploration:bool=False,
            visual:bool=True, 
            to_video:bool=False,
            sleep_time:float=0.3,
            num_rounds:int=200) -> None:
        '''
        Run the agent on the environment and displays the behavior.
        Agent does not learn.
        Input:
          - from_file (bool), if true, attemts to load the agent from file
          - no_exploration (bool), if true, makes epsilon = 0
          - visual (bool),
            True: displays the environment as in a video using environment render
            False: displays the behavioral data in the terminal step by step
          - to_video (bool),
            True: saves the rendering to a video file 
            False: displays the environment as in a video using environment render
          - sleep_time (float), determines the speed of the renderization
          - num_rounds (int), number of rounds to display
        '''
        # Load agent from name
        self.load_agent(from_file=from_file)
        # self.agent.debug = True # Uncomment for debugging
        if no_exploration:
          backup_epsilon, backup_policy = self.shutdown_agent_exploration()
        if visual and not to_video:
          '''
          To display the environment as in a video
          '''
          # Create environment
          # self.load_env(render_mode='human')
          self.load_env(render_mode='rgb_array')
          try:
            self.environment._max_episode_steps = num_rounds
          except:
             pass
          # Create episode
          episode = Episode(environment=self.environment,\
                            env_name=self.env_name,\
                            agent=self.agent,\
                            model_name=self.agent_name,\
                            num_rounds=num_rounds,\
                            state_interpreter=self.state_interpreter)
          episode.sleep_time = sleep_time
          episode.renderize(to_video=False)
        elif to_video:
          '''
          To save to a video file
          '''
          # Create environment
          self.load_env(render_mode='rgb_array')
          try:
            self.environment._max_episode_steps = num_rounds
          except:
             pass
          # Create episode
          episode = Episode(environment=self.environment,\
                            env_name=self.env_name,\
                            agent=self.agent,\
                            model_name=self.agent_name,\
                            num_rounds=num_rounds,\
                            state_interpreter=self.state_interpreter)
          episode.renderize(to_video=True,
                            file=self.video_folder)
        else:
          '''
          To display data information in the terminal
          '''
          # Create environment
          self.load_env(render_mode=None)
          try:
            self.environment._max_episode_steps = num_rounds
          except:
             pass
          # Create episode
          episode = Episode(environment=self.environment,\
                            env_name=self.env_name,\
                            agent=self.agent,\
                            model_name=self.agent_name,\
                            num_rounds=num_rounds,\
                            state_interpreter=self.state_interpreter
                            )
          df = episode.run(verbose=4, learn=False)
          self.data = df
        print('Number of rounds:', len(episode.agent.rewards) - 1)
        print('Total reward:', np.nansum(episode.agent.rewards))
        if no_exploration:
          self.agent.epsilon = backup_epsilon
          self.agent.policy = backup_policy
            
    def train(
            self, 
            num_rounds: int=200, 
            num_episodes: int=500, 
            from_file: bool=False,
            cool_down: Optional[Union[str, None]]=None,
            save_data: Optional[bool]=True,
            plot: Optional[bool]=True,
          ) -> None:
        '''
        Trains agent.
        '''
        # Load agent from name
        self.load_agent(from_file=from_file)
        # Create environment
        self.load_env(render_mode=None)
        try:
          self.environment._max_episode_steps = num_rounds
        except:
            pass
        # Create episode
        episode = Episode(
          environment=self.environment,\
          env_name=self.env_name,\
          agent=self.agent,\
          model_name=self.agent_name,\
          num_rounds=num_rounds,\
          state_interpreter=self.state_interpreter
        )
        # lengths = f'#states:{len(episode.agent.states)} -- #actions:{len(episode.agent.actions)} -- #rewards:{len(episode.agent.rewards)} -- #dones:{len(episode.agent.dones)}'
        # print('/>', lengths)
        # Train agent
        if save_data:
           file = self.file_csv
        else:
           file = None
        df = episode.simulate(
          num_episodes=num_episodes, 
          cool_down=cool_down,
          file=file
        )
        self.data = df
        if save_data:
          print(f'Data saved to {self.file_csv}')
        # Save agent to file
        self.save_agent()
        print(f'Agent saved to {self.file_model}')
        # Plot results
        if plot:
          p =  Plot(df)
          if num_episodes == 1:
            p.plot_round_reward(file=self.file_png)    
          else:
            p.plot_rewards(file=self.file_png) 
          print(f'Plot saved to {self.file_png}')
          # Save losses if agent uses NN
          if hasattr(self.agent, 'NN'):
            if hasattr(self.agent.NN, 'losses'):
              losses = self.agent.NN.losses
              fig, ax = plt.subplots(figsize=(4,3.5))
              ax = sns.lineplot(x=range(len(losses)), y=losses)
              ax.set_xlabel("Epoch",fontsize=14)
              ax.set_ylabel("Loss",fontsize=14)
              ax.grid()
              plt.savefig(self.file_losses, dpi=300, bbox_inches="tight")
          elif hasattr(self.agent, 'policy'):
            if hasattr(self.agent.policy, 'losses'):
              losses = self.agent.policy.losses
              fig, ax = plt.subplots(figsize=(4,3.5))
              ax = sns.lineplot(x=range(len(losses)), y=losses)
              ax.set_xlabel("Epoch",fontsize=14)
              ax.set_ylabel("Loss",fontsize=14)
              ax.grid()
              plt.savefig(self.file_losses, dpi=300, bbox_inches="tight")

    def test(self, 
             no_exploration:bool=True,
             from_file:bool=True, 
             num_rounds:int=200, 
             num_episodes:int=100):
        '''
        Test the trained agent.
        '''
        # Load agent from name
        self.load_agent(from_file=from_file)
        if no_exploration:
          # Shutdown exploration
          backup_epsilon, backup_policy = self.shutdown_agent_exploration()
        # Create environment
        self.load_env(render_mode=None)
        try:
          self.environment._max_episode_steps = num_rounds
        except:
            pass
        # Create episode
        episode = Episode(environment=self.environment,\
                          env_name=self.env_name,\
                          agent=self.agent,\
                          model_name=self.agent_name,\
                          num_rounds=num_rounds,\
                          state_interpreter=self.state_interpreter
                          )
        # Run simulation
        df = episode.simulate(num_episodes=num_episodes, learn=False)
        self.data = df
        df.to_csv(self.file_test_csv)
        print(f'Data saved to {self.file_test_csv}')
        # Plot results
        p = Plot(df)
        p.plot_histogram_rewards(self.file_test)
        print(f'Plot saved to {self.file_test}')
        if no_exploration:
          self.agent.epsilon = backup_epsilon
          if hasattr(self.agent, 'policy'):
            self.agent.policy = backup_policy
         
    def sweep(self, 
              parameter:str, 
              values:list, 
              num_rounds:int=200, 
              num_episodes:int=100,
              num_simulations:int=10):
        '''
        Runs a sweep over the specified parameter 
        with the specified values.
        '''
        # Load agent from name
        self.load_agent()
        # Creates environment
        self.load_env(render_mode=None)
        # Creates experiment
        experiment = Experiment(environment=self.environment,\
                                env_name=self.env_name,\
                                num_rounds=num_rounds,\
                                num_episodes=num_episodes,\
                                num_simulations=num_simulations,\
                                state_interpreter=self.state_interpreter
                  )
        # Run sweep
        experiment.run_sweep1(agent=self.agent, \
                       name=self.agent_name, \
                       parameter=parameter, \
                       values=values, \
                       measures=['reward'])
        self.data = experiment.data
        # Plot results
        p = Plot(experiment.data)
        print('Plotting...')
        p.plot_rewards(self.file_compare_rew)
        print(f'Plot saved to {self.file_compare_rew}')
        
    def compare_test(self, 
                     agent_vs_name:str,
                     agent_vs_parameters:Dict,
                     num_rounds:int=200, 
                     num_episodes:int=100):
        '''
        Runs a comparison of two agents
        over an environment.
        Agents are loaded from file.
        '''
        # Load agent 1
        self.load_agent(from_file=True)
        self.shutdown_agent_exploration()
        agent1 = deepcopy(self.agent)
        # Load vs agent
        backup_agent_name = self.agent_name
        backup_agent_parameters = deepcopy(self.agent_parameters)
        self.agent_name = agent_vs_name
        self.agent_parameters = agent_vs_parameters
        self.consolidate_folders()
        try:
          self.load_agent(from_file=True)
        except Exception as e:
          print(e)
          raise Exception(f'An agent of class {agent_vs_name} is required.\nRun a performer on such an agent first.') 
        self.shutdown_agent_exploration()
        agent2 = deepcopy(self.agent)
        self.agent_name = backup_agent_name
        self.agent_parameters = backup_agent_parameters
        self.consolidate_folders()
        # Create environment
        self.load_env(render_mode=None)
        # Create experiment
        experiment = Experiment(environment=self.environment,\
                                env_name=self.env_name,\
                                num_rounds=num_rounds,\
                                num_episodes=num_episodes,\
                                num_simulations=1,\
                                state_interpreter=self.state_interpreter
                  )
        # Run sweep
        experiment.run_experiment(agents=[agent1, agent2], \
                                  names=[self.agent_name, agent_vs_name], \
                                  measures=['hist_reward'],\
                                  learn=False)
        self.data = experiment.data
        # Plot results
        p = Plot(experiment.data)
        print('Plotting...')
        p.plot_histogram_rewards(self.file_compare_hist)
        print(f'Plot saved to {self.file_compare_hist}')

