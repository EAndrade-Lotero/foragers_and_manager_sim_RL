'''
© Edgar Andrade 2024
Email: edgar.andrade@urosario.edu.co

-----------------------------------------------
Helper functions to gather, process and visualize data

Includes:
    - Episode, Runs the environment for a number of rounds and keeps tally of everything.
    - Experiment, Compares given models on a number of measures.
    - EnvfromGameAndPl2
-----------------------------------------------
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep
from copy import deepcopy
from tqdm.auto import tqdm
from collections import deque
from IPython.display import clear_output, display
# from gymnasium.utils.save_video import save_video
from typing import Union, List, Tuple, Optional, Callable

from utils.plot_utils import Plot
from agents.BaseAgent import Agent
from utils.interpreters import id_state

class Episode :
    '''
    Runs the environment for a number of rounds and keeps tally of everything.
    '''

    def __init__(
                self, 
                environment: any,
                env_name: str, 
                agent: Agent, 
                model_name: str, 
                num_rounds: int, 
                id: Optional[int]=0,
                state_interpreter: Optional[Callable]=id_state,
            ) -> None:
        self.environment = environment
        self.env_name = env_name
        self.agent = agent
        self.model_name = model_name
        self.num_rounds = num_rounds
        self.done = False
        self.T = 0
        self.id = id
        self.sleep_time = 0.3
        self._clean_state = state_interpreter
        state_ = self.environment.reset()
        state = self._clean_state(state_)
        self.initial_state = state
        if agent is not None:
            self.agent.restart()
            self.agent.states.append(state)

    def play_round(
                self, 
                verbose: Optional[int]=0, 
                learn: Optional[bool]=True
            ) -> None:
        '''
        Plays one round of the game.
        Input:
            - verbose, to print information.
                0: no information
                1: only number of simulation
                2: simulation information
                3: simulation and episode information
                4: simulation, episode and round information
            - learn, a boolean to determine if agent learning is enabled.
        '''
        # Ask agent to make a decision
        try:
            action = self.agent.make_decision()
        except Exception as e:
            raise Exception('Error: Agent could not choose an action.\n\t', e)
        # Runs the environment and obtains the next_state, reward, done, info
        result = self.environment.step(action)            
        next_state = self._clean_state(result[0])
        reward = result[1]
        done = result[2]
        # Update records
        self.agent.actions.append(action)
        if hasattr(self.agent, 'next_states'):
            self.agent.next_states.append(next_state)
        self.agent.rewards.append(reward)
        self.agent.dones.append(done)
        # Prints info
        if verbose > 3:
            state = self.agent.states[-1]
            print(f'\tThe state is => {state}')
            print(f'\tAgent takes action => {action}')
            print(f'\tThe state obtained is => {next_state}')
            print(f'\tThe reward obtained is => {reward}')
            print(f'\tEnvironment is finished? => {done}')
        # Agent learns
        if learn:
            try:
                self.agent.update()
            except:
                try:
                    self.agent.update(next_state, reward, done)
                except Exception as e:
                    raise Exception(e)
        # Update records
        self.agent.states.append(next_state)
        # Update round counter
        self.T += 1
        # Update environment "is-finished?"
        self.done = done
        # lengths = f'#states:{len(self.agent.states)} -- #actions:{len(self.agent.actions)} -- #rewards:{len(self.agent.rewards)} -- #dones:{len(self.agent.dones)}'
        # print('===>', lengths)

    def run(
                self, 
                verbose: Optional[int]=0, 
                learn: Optional[bool]=True
            ) -> pd.DataFrame:
        '''
        Plays the specified number of rounds.
        Input:
            - verbose, to print information.
                0: no information
                1: only number of simulation
                2: simulation information
                3: simulation and episode information
                4: simulation, episode and round information
            - learn, a boolean to determine if agent learning is enabled.
        '''
        # Restart for a new episode
        self.reset()
        # Play the specified number of rounds
        for round in range(self.num_rounds):
            if not self.done:
                if verbose > 2:
                    print('\n' + '-'*10 + f'Round {round}' + '-'*10 + '\n')
                self.play_round(verbose=verbose, learn=learn)                
            else:
                break
        df = self.to_pandas()
        return df

    def to_pandas(self) -> pd.DataFrame:
        '''
        Creates a pandas dataframe with the information from the current objects.
        Output:
            - pandas dataframe with the following variables:           
                Variables:
                    * episode: a unique identifier for the episode
                    * round: the round number
                    * action: the player's action
                    * reward: the player's reward
                    * done: whether the environment is active or not
                    * model: the model's name
                    * environment: the environment's name
        '''
        # Perform cleansing operations on registers
        if hasattr(self.agent, 'next_states'):
            next_states = [s for s in self.agent.next_states]
        else:
            next_states = None
        states, actions, rewards, dones, next_states = self.cleanse_registers(
            states=[s for s in self.agent.states], 
            actions=[s for s in self.agent.actions], 
            rewards=[s for s in self.agent.rewards], 
            dones=[s for s in self.agent.dones],
            next_states=next_states
        )
        # Gather data
        length = min(self.T, len(states))
        # print(f'length:{length} --- self.T:{self.T} --- len(states):{len(states)}')
        df = pd.DataFrame.from_dict({
            "model": [self.model_name] * length,
            "environment": [self.env_name] * length,
            "episode": [self.id] * length,
            "round": [i for i in range(length)],
            "state": states,
            "action": actions,
            "reward": rewards,
            "done": dones
        })        
        if next_states is not None:
            df['next_state'] = next_states
        return df

    def cleanse_registers(
                self,
                states: List[any],
                actions: List[int],
                rewards: List[float],
                dones: List[bool],
                next_states: List[any]
            ) -> Tuple[List[any], List[any], List[any], List[any]]:
        '''
        Perform sanity checks on agent's memory lists:
            - Sanity check 1: All lists should be equal length
            - Sanity check 2: Do not keep information from previous episodes
        '''
        # states has one more than the rest. Don't use the last register
        states = states[:-1]
        # --------------------------------------------------------------
        # Sanity check 1: Do not keep information from previous episodes
        # --------------------------------------------------------------
        length = min(self.T, len(states))
        states = states[-length:]
        actions = actions[-length:]
        rewards = rewards[-length:]
        dones = dones[-length:]
        states = states[-length:]
        if next_states is not None:
            next_states = next_states[-length:]
        # --------------------------------------------------------------
        # Sanity check 2: all lists should be equal length
        # --------------------------------------------------------------
        lengths = f'#states:{len(states)} -- #actions:{len(actions)} -- #rewards:{len(rewards)} -- #dones:{len(dones)}'
        n = len(states)
        assert(len(actions) == n), lengths
        assert(len(rewards) == n), lengths
        assert(len(dones) == n), lengths
        if next_states is not None:
            assert(len(next_states) == n), lengths
        # Return
        return states, actions, rewards, dones, next_states

    def reset(self) -> None:
        '''
        Reset the episode. This entails:
            reset the environment
            restart the agent 
        '''
        # Reset environment and get initial state
        state = self.environment.reset()
        state = self._clean_state(state)
        # Restart the agent
        self.agent.restart()
        # lengths = f'#states:{len(self.agent.states)} -- #actions:{len(self.agent.actions)} -- #rewards:{len(self.agent.rewards)} -- #dones:{len(self.agent.dones)}'
        # print('///>', lengths)
        # If agent's restart does not keep the state history, create 
        # a new history with initial state
        if len(self.agent.states) <= 1:
            if isinstance(self.agent.states, deque):
                self.agent.states = deque([state], maxlen=self.agent.max_len)
            elif isinstance(self.agent.states, list):
                self.agent.states = [state]
            else:
                raise Exception(f'Error: type of memory not accepted for states ({type(self.agent.states)}).')
        else:
            # Otherwise, append initial state to state history
            self.agent.states.append(state)
        # Restart round count
        self.T = 0
        # Environment is not done
        self.done = False

    def renderize(
                self, 
                to_video:Optional[bool]=False, 
                file:Optional[str]=None
            ) -> None:
        '''
        Plays the specified number of rounds.
        '''
        if to_video:
            assert(file is not None), 'A folder name must be provided with the argument file='
            rm = self.environment.render_mode
            assert(rm == 'rgb_array'), f'To create video, environment render mode should be rgb_array, not {rm}'
            frame_list = []
        # Initialize img object
        img = plt.imshow(np.array([[0, 0], [0, 0]]))
        # Play the specified number of rounds
        for round in range(self.num_rounds):
            if self.done:
                break
            im = self.environment.render()
            if isinstance(im, np.ndarray) and to_video:
                # Save image to fame list
                frame_list.append(im)
            clear_output(wait=True)
            img.set_data(im)
            plt.axis('off')
            display(plt.gcf())
            sleep(self.sleep_time)
            self.play_round(verbose=0, learn=False)                
        if to_video:
            assert(len(frame_list) > 0), 'No frames saved. Check env.render() is providing np.ndarrays.'
            save_video(
                frames=frame_list,
                video_folder=file,
                fps=1/self.sleep_time
            )
        plt.close()
   
    def simulate(
                self, 
                num_episodes: int=1, 
                file: Optional[Union[str, None]]=None, 
                verbose: Optional[int]=0, 
                learn: Optional[bool]=True,
                cool_down: Optional[Union[str, None]]=None
            ) -> pd.DataFrame:
        '''
        Runs the specified number of episodes for the given number of rounds.
        Input:
            - num_episodes, int with the number of episodes.
            - file, string with the name of file to save the data on.
            - verbose, to print information.
                0: no information
                1: only number of simulation
                2: simulation information
                3: simulation and episode information
                4: simulation, episode and round information
            - learn, a boolean to determine if agent learning is enabled.
        Output:
            - Pandas dataframe with the following variables:

                Variables:
                    * id_sim: a unique identifier for the simulation
                    * round: the round number
                    * action: the player's action
                    * reward: the player's reward
                    * done: whether the environment is active or not
                    * model: the model's name
                    * environment: the environment's name
        '''
        # Create the list of dataframes
        data_frames = []
        # Run the number of episodes
        for ep in tqdm(range(num_episodes)):
            if verbose > 1:
                print('\n' + '='*10 + f'Episode {ep}' + '='*10 + '\n')
            self.id = ep
            # check cooldown
            if cool_down is not None:
                self._apply_cool_down(
                    cool_down=cool_down,
                    num_episodes=num_episodes,
                    ep=ep
                )
            # Run the episode
            # lengths = f'#states:{len(self.agent.states)} -- #actions:{len(self.agent.actions)} -- #rewards:{len(self.agent.rewards)} -- #dones:{len(self.agent.dones)}'
            # print('//>', lengths)
            df = self.run(verbose=verbose, learn=learn)
            # Include episode in list of dataframes
            data_frames.append(df)
        # Concatenate dataframes
        data = pd.concat(data_frames, ignore_index=True)
        if file is not None:
            data.to_csv(file)
        return data
    
    def _apply_cool_down(
                self, 
                cool_down: str, 
                num_episodes: int,
                ep: int
            ) -> None:
        if 'staircase-' in cool_down:
            h = int(cool_down.split('-')[1])
            step_epsilon = 1 / h
            step = num_episodes / h
            bins_episode = [step * i for i in range(h)]
            # print('===>', bins_episode)
            bin = np.digitize([ep], bins_episode).item() - 1
            # print('===>', bin)
            self.agent.epsilon = 1 - (bin / h)
            print(f"Simulating with agent's epsilon = {self.agent.epsilon}")


class Experiment :
    '''
    Compares given models on a number of measures.
    '''

    def __init__(self, \
                 environment:any, \
                 env_name:str, \
                 num_rounds:int, \
                 num_episodes:int, \
                 num_simulations:int, \
                 state_interpreter=id_state):
        '''
        Input:
            - environment, object with the environment on which to test the agents.
            - env_name, the environment's name.
            - num_rounds, int with the number of rounds.
            - num_episodes, int with the number of episodes.
            - num_simulations, int with the number of times the environment should be
                restarted and run the episodes again.
        '''
        self.environment = environment
        self.env_name = env_name
        self.num_rounds = num_rounds
        self.num_episodes = num_episodes
        self.num_simulations = num_simulations
        self.state_interpreter = state_interpreter
        self.data = None

    def run_experiment(self, \
                       agents:List[any], \
                       names:List[str], \
                       measures:List[str], \
                       learn:bool=True) -> None:
        '''
        For each agent, runs the simulation the stipulated number of times,
        obtains the data and shows the plots on the given measures.
        Input:
            - agents, list of agent objects.
            - names, list of names of the models implemented by the agents.
            - measures, list of measures, which could contain the following strings:
                * 'reward'
                * 'round_reward'
                * 'hist_reward'
            - learn, a boolean to enable learning.
        '''
        # Creates the list of dataframes
        data_frames = []
        # Run simulations
        for k in tqdm(range(self.num_simulations), desc='Running simulations'):
            # Reset all agents
            if learn:
                for agent in agents:
                    agent.reset()
            # Iterate over episodes
            for ep in tqdm(range(self.num_episodes), desc='\tRunning episodes', leave=False):
                # Initialize Episode
                sim_core = Episode(environment=self.environment, \
                                   env_name=self.env_name, \
                                   agent=None, \
                                   model_name=None,\
                                   num_rounds=self.num_rounds,\
                                   state_interpreter=self.state_interpreter)
                # Keep a unique id number for the episode
                sim_core.id = ep
                counter_agent = -1
                for agent in agents:
                    counter_agent += 1
                    # Copy Episode and place agent
                    sim = deepcopy(sim_core)
                    # Restart agent for a new episode
                    agent.restart()
                    sim.agent = agent
                    sim.agent.states.append(sim.initial_state)
                    sim.model_name = names[counter_agent]
                    # Run episode over agent
                    df = sim.run(verbose=False, learn=learn)
                    df["simulation"] = k
                    df["model"] = names[counter_agent]
                    data_frames.append(df)
        # Consolidate data
        data = pd.concat(data_frames, ignore_index=True)
        self.data = data
        # Create plots
        for m in measures:
            if m == 'reward':
                ax = Plot(data).plot_rewards(m)
            if m == 'round_reward':
                ax = Plot(data).plot_round_reward(m)
            if m == 'hist_reward':
                ax = Plot(data).plot_histogram_rewards(m)
            try:
                ax.set_title(m)
            except:
                pass
            plt.show()
        return agents

    def run_sweep1(self, \
                    agent:any, \
                    name:str, \
                    parameter:str, \
                    values:List[Union[int, float, str]], \
                    measures:List[str], \
                    learn:bool=True) -> None:
        '''
        For each agent, runs a parameter sweep the stipulated number
        of times, obtains the data and shows the plots on the given measures.
        Input:
            - agent, an object agent.
            - name, the name of the model implemented by the agent.
            - parameter, a string with the name of the parameter.
            - values, a list with the parameter's values.
            - measures, list of measures, which could contain the following strings:
                * 'reward'
                * 'round_reward'
            - learn, a boolean to enable learning.
        '''
        # Creates list of agents
        agents = []
        for value in values:
            agent_ = deepcopy(agent)
            instruction = f'agent_.{parameter} = {value}'
            exec(instruction)
            agents.append(agent_)
        # Creates list of names
        names = [f'({name}) {parameter}={value}' for value in values]
        # Run the experiment
        self.run_experiment(agents=agents,\
                            names=names,\
                            measures=[],\
                            learn=learn)            


class EnvfromGameAndPl2:
    '''
    Implementa un entorno a partir de un juego y del segundo jugador.
    '''
    def __init__(self, game:any, other_player:any):
        self.other_player = other_player
        self.initial_game = deepcopy(game)
        self.game = game
        self.state = self.game.estado_inicial
        self.list_acciones = None

    def reset(self):
        self.game = deepcopy(self.initial_game)
        self.state = self.game.estado_inicial
        self.other_player.reset()
        self.other_player.states.append(self.state)
        return self.state

    def render(self):
        self.game.render(self.state)

    def test_objetivo(self, state):
        if not self.game.es_terminal(state):
            return False
        else:
            player = self.game.player(state)
            return self.utilidad(state, player) > 0
        
    def acciones_aplicables(self, state):
        return self.game.acciones(state)

    def step(self, action):
        if self.list_acciones is not None:
            action_ = self.list_acciones[action] 
        else:
            action_ = action
        state = self.state
        playing = self.game.player(state)
        # print(f'player {playing} in state {state} makes move {action}')
        # First player made a move. Get new state, reward, done
        try:
            new_state = self.game.resultado(state, action_)
        except Exception as e:
            if action_ not in self.game.acciones(state):
                # Punish agent for playing an impossible action
                return state, -1000, True
            print(state)
            raise Exception(e)
        # self.game.render(new_state)
        # print(f'obtains {new_state}')
        reward = self.game.utilidad(new_state, playing)
        reward = reward if reward is not None else 0
        done = self.game.es_terminal(new_state)
        # If not done, second player makes a move
        if not done:
            playing = self.game.player(new_state)
            # Actualize second player with previous move
            self.other_player.states.append(new_state)
            if hasattr(self.other_player, 'choices'):
                possible_actions = self.game.acciones(new_state)
                self.other_player.choices = possible_actions
            # Second player makes a move
            other_player_action = self.other_player.make_decision()
            if self.other_player.debug:
                print(f'Negras mueven en {other_player_action}')
            # print(f'player {playing} in state {new_state} makes move {other_player_action}')
            # Get new state, reward, done
            new_state = self.game.resultado(new_state, other_player_action)
            # print(f'obtains {new_state}')
            reward = self.game.utilidad(new_state, playing)
            reward = reward if reward is not None else 0
            done = self.game.es_terminal(new_state)
        # Bookkeeping
        self.state = new_state
        return new_state, reward, done   
    
