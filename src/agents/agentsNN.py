'''
Classes for implementing the learning methods for 
large and for continuum state spaces using Neural Networks
as approximation function for Q values.
We assume a discrete action space.
We assume epsilon-greedy action selection.
'''
import torch
import numpy as np
from pathlib import Path
from copy import deepcopy
from collections import deque
from termcolor import colored
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader

class ExperienceDataset(Dataset):
    '''
    Creates the dataset out of the experience stream
    '''
    def __init__(
                self, 
                states:List[torch.Tensor], 
                actions:List[int], 
                updates:List[float]
            ) -> None:
        self.states = states
        self.actions = actions
        self.updates = [torch.tensor(u, dtype=torch.float32) for u in updates]
        n = len(self.states)
        assert (len(self.actions) == n)
        assert (len(self.updates) == n)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx:int):
        state = self.states[idx].to(torch.float32)
        action = int(self.actions[idx])
        update = self.updates[idx].to(torch.float32) 
        return state, action, update

       
class DQN() :
    '''
    Implements the Deep Q Network with 
    experience replay and target network.
    '''
    def __init__(self, parameters:Dict[str, any]):
        self.parameters = parameters
        self.nA = self.parameters['nA']
        self.gamma = self.parameters['gamma']
        self.epsilon = self.parameters['epsilon']
        self.NN = self.parameters['NN']
        assert(hasattr(self.NN, 'predict')), 'NN must be an object with a predict() method'
        assert(hasattr(self.NN, 'values_vector')), 'NN must be an object with a values_vector() method'
        assert(hasattr(self.NN, 'learn')), 'NN must be an object with a learn() method'
        assert(hasattr(self.NN, 'save')), 'NN must be an object with a save() method'
        assert(hasattr(self.NN, 'load')), 'NN must be an object with a load() method'
        assert(hasattr(self.NN, 'reset')), 'NN must be an object with a reset() method'
        # Create memory registers
        self.max_len = self.parameters['max_len']
        self.states = deque(maxlen=self.max_len)
        self.actions = deque(maxlen=self.max_len)
        self.next_states = deque(maxlen=self.max_len)
        self.rewards = deque(maxlen=self.max_len)
        self.dones = deque(maxlen=self.max_len)
        # Create model file
        self.model_folder = Path.cwd() / Path('..').resolve() / Path('..').resolve() / Path('models', 'MLP')
        self.model_folder.mkdir(parents=True, exist_ok=True)
        self.model_file = self.model_folder / Path('mlp.pt')
        # Create experience replay
        self.len_exp = parameters["len_exp"]
        assert(self.len_exp <= self.max_len)
        self.batch_size = parameters["batch_size"]
        assert(self.batch_size <= self.len_exp)
        self.num_epochs = parameters["num_epochs"]
        # check priority experience replay
        if 'use_priority' in parameters.keys():
            self.use_priority = parameters['use_priority']
            if 'w' in parameters.keys():
                self.w = parameters['w']
        else:
            self.use_priority = False
            self.w = 1
        # Create target network
        self.target_network_latency = parameters["target_network_latency"]
        self.NN_hat = deepcopy(self.NN)
        # Define how often to train network
        if 'skip_rounds' in parameters.keys():
            self.skip_rounds = parameters['skip_rounds']
        else:
            self.skip_rounds = 1
        self.seed = None
        self.debug = False
        # Start turn counter
        self.turn = 0


    def restart(self):
        # # Set the inital state marker
        # self.states.append(np.nan)
        # self.actions.append(np.nan)
        # self.next_states.append(np.nan)
        # self.rewards.append(np.nan)
        # self.dones.append(np.nan)
        # Restarts the NN
        self.NN.restart()
        # Restart turn counter
        self.turn = 0
    
    def reset(self) -> None:
        '''
        Resets the agent for a new simulation.
        '''
        self.states = deque(maxlen=self.max_len)
        self.actions = deque(maxlen=self.max_len)
        self.next_states = deque(maxlen=self.max_len)
        self.rewards = deque(maxlen=self.max_len)
        self.dones = deque(maxlen=self.max_len)
        # Resets the NN
        self.NN.reset()
        # Restart turn counter
        self.turn = 0
    
    def save(self, file:Path) -> None:
        self.NN.save(file=file)

    def load(self, file:Path) -> None:
        self.NN.load(file=file)

    def make_decision(self, state:Optional[any]=None):
        '''
        Agent makes an epsilon greedy accion based on Q values.
        '''
        if np.random.uniform(0,1) < self.epsilon:
            return np.random.choice(range(self.nA))
        else:
            if state is None:
                state = self.states[-1]
            return self.argmaxQ(state)        

    def argmaxQ(self, state):
        '''
        Determines the action with max Q value in state s
        '''
        Qvals = self.NN.values_vector(state)
        maxQ = max(Qvals)
        opt_acts = [a for a, q in enumerate(Qvals) if np.isclose(q, maxQ)]
        if self.seed is not None:
            np.random.seed(self.seed)
        try:
            argmax = np.random.choice(opt_acts)
            return argmax
        except Exception as e:
            print('')
            print(colored('%'*50, 'red'))
            print(colored(f'Error in argmaxQ ====>', 'red'))
            print(colored(f'state:\n\t{state}', 'red'))
            print('')
            print(colored(f'Qvals:{Qvals}', 'red'))
            print(colored(f'len:{len(Qvals)} --- type:{type(Qvals)}', 'red'))
            print('')
            print(colored(f'maxQ:{maxQ}', 'red'))
            print(colored(f'type:{type(maxQ)}', 'red'))
            print('')
            print(colored(f'opt_acts:{opt_acts}', 'red'))
            print(colored(f'opt_acts:{[a for a, q in enumerate(Qvals) if np.isclose(q, maxQ)]}', 'red'))
            print(colored('%'*50, 'red'))
            print('')
            raise Exception(e)      
            
    def update(self) -> None:
        '''
        Agent updates NN with experience replay and updates target NN.
        '''
        # Update turn counter
        self.turn += 1
        #####################################
        # Obtain length of experience
        #####################################
        n = len(self.actions)
        k = self.max_len
        if n < k:
            #####################################
            #agent only learns with enough experience
            #####################################
            # print('agent only learns with enough experience')
            pass
        else:
            #####################################
            #agent only learns every skip round
            #####################################
            if self.turn % self.skip_rounds == 0:
                print('We have to learn', self.turn, self.skip_rounds)
                #####################################
                # Create the experience stream with
                # random indices from the whole history
                #####################################
                states, actions, next_states, rewards, dones = self.create_experience_stream(
                    use_priority=self.use_priority,
                    w=self.w
                )
                #####################################
                # Create the dataset and dataloader
                #####################################
                ds = self.create_dataset(states, actions, next_states, rewards, dones)
                ds_loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
                #####################################
                # Train for number of epochs
                #####################################
                # print('Training...')
                for e in range(self.num_epochs):
                    self.NN.learn(ds_loader)
            #####################################
            # Check if it's turn to update the target network
            #####################################
            if len(self.actions) % self.target_network_latency == 0:
                # print('Updating Target Network')
                self.NN_hat = deepcopy(self.NN)

    def create_experience_stream(
                self,
                use_priority: Optional[bool]=True,
                w: Optional[float]=1
            ) -> Tuple[List[any],List[any],List[any],List[any],List[any]]:
        #####################################
        # Get records
        #####################################
        states = list(self.states).copy()
        actions = list(self.actions).copy()
        next_states = list(self.next_states).copy()
        rewards = list(self.rewards).copy()
        dones = list(self.dones).copy()
        # # ###############################################################
        # # # Make sure indices don't correspond to initial states markers
        # # ###############################################################
        # states, actions, next_states, rewards, dones = DQN._avoid_indices_for_initial_states_markers(
        #     states, actions, next_states, rewards, dones
        # )
        if self.debug:
            DQN._check_lists_in_good_shape(states, actions, next_states, rewards, dones)
        #############################################
        # Create mask with random indices
        # We use priority indexes according to delta
        #############################################
        if use_priority:
            p = self.create_probability_vector(
                states=states, 
                next_states=next_states, 
                actions=actions, 
                rewards=rewards, 
                w=w
            )
        else:
            p = np.ones(len(states)) * 1/len(states)
        # print(len(p), p)
        # print(len(states))
        size = min(self.len_exp, len(states))
        mask = np.random.choice(
            range(len(states)), 
            size=size, 
            p=p,
            replace=False
        )
        #########################################
        # Fitler according to mask
        #########################################
        states, actions, next_states, rewards, dones = DQN._select_from_mask(
            mask, states, actions, next_states, rewards, dones
        )
        if self.debug:
            # Check lists are in good shape
            DQN._check_lists_in_good_shape(states, actions, next_states, rewards, dones)
        return states, actions, next_states, rewards, dones

    def create_dataset(
                self, 
                states:List[torch.Tensor], 
                actions:List[int], 
                next_states:List[torch.Tensor], 
                rewards:List[float], 
                dones:List[bool]
            ) -> ExperienceDataset:
        updates = [self.get_update(next_states[i], rewards[i], dones[i]) for i in range(len(rewards))]
        return ExperienceDataset(states, actions, updates)

    def get_update(self, next_state:torch.Tensor, reward:float, done:bool):
        if done:
            #####################################
            # Episode is finished. No need to bootstrap update
            #####################################
            G = reward
        else:
            #####################################
            # Episode is active. Bootstrap update using Target Network
            #####################################
            Qvals = self.NN_hat.values_vector(next_state)
            # print(f'get_update Qvals:{Qvals}')
            maxQ = max(Qvals)
            G = reward + self.gamma * maxQ
        # print(f'get_updates G:{G}')
        return G    
    
    def create_probability_vector(
                self,
                states: any,
                next_states: any,
                actions: any,
                rewards: any,
                w: float
            ) -> List[float]:
        probs = [self.get_preference(state, next_state, action, reward, w) 
                 for state,next_state,action,reward in zip(states,next_states,actions,rewards)]
        probs = np.array(probs)
        return probs/sum(probs)
    
    def get_preference(
                self,
                state: any,
                next_state: any,
                action: any,
                reward: float,
                w: Optional[int]=1
            )-> float:
        Qvals = self.NN_hat.values_vector(state)
        assert(len(np.array(Qvals).shape) == 1), f'Qvals.shape = {len(np.array(Qvals).shape)}'
        Qvals_ns = self.NN_hat.values_vector(next_state)
        assert(len(np.array(Qvals_ns).shape) == 1), f'Qvals_ns.shape = {len(np.array(Qvals_ns).shape)}'
        maxQ_ns = max(Qvals_ns)
        assert(not np.isnan(action) and not np.isnan(reward)), f'Warning: action={action} --- reward={reward}'
        result = abs(reward + self.gamma * maxQ_ns - Qvals[action]) ** w
        return result

    def print_dataset(self,
                use_priority: Optional[bool]=True,
                w: Optional[float]=1
            ) -> None:
        #####################################
        # Download records
        #####################################
        states, actions, next_states, rewards, dones = self.create_experience_stream(
            use_priority=use_priority,
            w=w
        )
        #####################################
        # Create the dataset
        #####################################
        ds = self.create_dataset(states, actions, next_states, rewards, dones)
        #####################################
        # Create the dataloader
        #####################################
        ds_loader = DataLoader(ds, batch_size=1, shuffle=False)
        for state, action, update in ds_loader:
            print('')
            print('='*30)
            print('state:')
            print(state)
            print('-'*20)
            current = self.NN.predict(state, action)
            print(f'Action: {action.item()}\t Update: {update.item()}\t Current value: {current}')

    @staticmethod
    def _avoid_indices_for_initial_states_markers(
                states:List[torch.Tensor], 
                actions:List[int], 
                next_states:List[torch.Tensor], 
                rewards:List[float], 
                dones:List[bool]
            ) -> Tuple[List[any],List[any],List[any],List[any],List[any]]:
        indices_no_terminal_states = [i for i, a in enumerate(actions) if not np.isnan(a)]
        # Filtering lists
        states, actions, next_states, rewards, dones = DQN._select_from_mask(
            indices_no_terminal_states, states, actions, next_states, rewards, dones
        )           
        return states, actions, next_states, rewards, dones

    @staticmethod
    def _avoid_indices_for_terminal_states(
                states:List[torch.Tensor], 
                actions:List[int], 
                next_states:List[torch.Tensor], 
                rewards:List[float], 
                dones:List[bool]
            ) -> Tuple[List[any],List[any],List[any],List[any],List[any]]:
        indices_no_terminal_states = [i for i, a in enumerate(actions) if not np.isnan(a)]
        # Filtering lists
        states, actions, next_states, rewards, dones = DQN._select_from_mask(
            indices_no_terminal_states, states, actions, next_states, rewards, dones
        )           
        return states, actions, next_states, rewards, dones

    @staticmethod
    def _check_lists_in_good_shape(
                states:List[any],
                actions:List[int],
                next_states:List[any],
                rewards:List[float],
                dones:List[bool]
            ) -> None:
        # Check lists are in good shape
        lengths = f'#states:{len(states)} -- #actions:{len(actions)} #next_states:{len(next_states)} -- #rewards:{len(rewards)} -- #dones:{len(dones)}'
        n = len(states)
        assert(len(next_states) == n), lengths
        assert(len(actions) == n), lengths
        assert(len(rewards) == n), lengths
        assert(len(dones) == n), lengths

    @staticmethod
    def _select_from_mask(
                mask,
                states:List[any],
                actions:List[int],
                next_states:List[any],
                rewards:List[float],
                dones:List[bool]
            ) -> Tuple[List[any],List[any],List[any],List[any],List[any]]:
        # Get the randomly selected states
        states = DQN._filter_list_of_any(mask, states)
        # states = DQN._filter_list_of_tensors(mask, states)
        # Get the randomly selected actions
        actions = DQN._filter_list_of_any(mask, actions)
        # Get the randomly selected NEXT states
        next_states = DQN._filter_list_of_any(mask, next_states)
        # next_states = DQN._filter_list_of_tensors(mask, next_states)
        # Get the randomly selected rewards
        rewards = DQN._filter_list_of_any(mask, rewards)
        # Get the randomly selected dones
        dones = DQN._filter_list_of_any(mask, dones)
        return states, actions, next_states, rewards, dones

    @staticmethod
    def _filter_list_of_tensors(
                mask: List[int], 
                tensors: List[torch.tensor]
            ) -> List[torch.tensor]:
        tensor_mask = torch.tensor(mask)
        stacked_tensors = torch.stack(list(tensors))
        filtered_tensors = torch.index_select(stacked_tensors, 0, tensor_mask)
        list_of_tensors = list(torch.unbind(filtered_tensors, dim=0))
        return list_of_tensors
        # return list_of_tensors + [tensors[-1]]

    @staticmethod
    def _filter_list_of_any(
                mask: List[int],
                objects: List[any]
            ) -> List[any]:
        return [object for i, object in enumerate(objects) if i in mask]
        # return np.array(objects.copy())[mask].tolist()
        # return np.array(objects.copy())[mask].tolist() + [objects[-1]]
