'''
Classes for implementing the learning methods for 
large and for continuum state spaces using 
an approximation function for Q values.
We assume a discrete action space.
We assume epsilon-greedy action selection.
'''
import numpy as np
from . BaseAgent import AgentCS

dash_line = '-'*20

class MonteCarloCS(AgentCS):
    '''
    Implements a Monte Carlo learning
    '''
    def __init__(self, parameters:dict):
        super().__init__(parameters)
        self.first_visit = self.parameters['first_visit']
        self.N = np.zeros((self.nS, self.nA))

    def update(self, next_state, reward, done):
        '''
        Agent updates its model.
        '''
        # Only learn when episode is done
        if not done:
            return        
        rewards = [r for r in self.rewards] + [reward]
        T = len(rewards) - 1
        G = 0
        for t in range(T - 1, -1, -1):
            reward = rewards[t+1]
            G  = self.gamma * G + reward
            state = self.states[t]
            if (not self.first_visit) or state not in self.states[:t]:
                action = self.actions[t]
                self.N[state, action] += 1
                prev_Q = self.Q.predict(state, action) 
                self.Q.learn(state, action, G)
                if self.debug:
                    print('')
                    print(dash_line)
                    print(f'Learning log round {t}:')
                    print(f'state:{state}')
                    print(f'action:{action}')
                    print(f'reward:{reward}')
                    print(f'G:{G}')
                    print(f'Previous Q:{prev_Q}')
                    print(f'New Q:{self.Q.predict(state, action)}')
        # Update policy
        for s in range(self.nS):
            self.update_policy(s)

    def restart(self):
        super().restart()
        self.N = np.zeros((self.nS, self.nA))

    def reset(self):
        super().reset()
        self.N = np.zeros((self.nS, self.nA))


class SarsaCS(AgentCS) :
    '''
    Implements a SARSA learning rule.
    '''
    def __init__(self, parameters:dict):
        super().__init__(parameters)

    def update(self, next_state, reward, done):
        '''
        Agent updates according to the SARSA rule.
        '''
        if done:
            # Episode is finished. No need to bootstrap update
            G = reward
        else:
            # Episode is active. Bootstraps update
            next_action = self.make_decision(next_state)
            G = reward + self.gamma * self.Q.predict(next_state, next_action)
        # Update Q values
        state, action = self.states[-1], self.actions[-1]
        self.Q.learn(state, action, G)


class QLearningCS(AgentCS) :
    '''
    Implements a Q-learning rule.
    '''
    def __init__(self, parameters:dict):
        super().__init__(parameters)

    def update(self, next_state, reward, done):
        '''
        Agent updates according to the Q-learning rule.
        '''
        if done:
            # Episode is finished. No need to bootstrap update
            G = reward
        else:
            # Episode is active. Bootstrap update
            max_value_next_state = self.max_Q(next_state)
            G = reward + self.gamma * max_value_next_state
        # Update Q values
        state, action = self.states[-1], self.actions[-1]
        self.Q.learn(state, action, G)


class nStepCS(AgentCS) :
    '''
    Implements a n-step SARSA learning rule.
    '''

    def __init__(self, parameters:dict, Q):
        super().__init__(parameters, Q)
        self.n = self.parameters['n']
        assert(self.n > 0)
        self.T = np.infty
        self.t = 0
        self.tau = 0
   
    def restart(self):
        '''
        Restarts the agent for a new trial.
        Keeps the same Q for more learning.
        '''
        super().restart()
        self.T = np.infty
        self.t = 0
        self.tau = 0

    def update(self, next_state, reward, done):
        '''
        Agent updates its model.
        '''
        def update_tau(tau):
            '''
            - Finds the utility G_{tau:tau+n} 
            - Updates Q[s_tau, a_tau]
            '''
            if tau >= 0:
                # Find utility G_{tau:tau+n} 
                end = min(tau+self.n, self.T)
                G = 0
                # print('\t===>', tau+1, end+1)
                for i in range(tau+1, end+1):
                    discount = self.gamma**(i-tau-1)
                    # print(f'\t\t---{i}---{discount}----')
                    G += discount*rewards[i]  
                # print('\t--->>> G', G)
                if tau + self.n < self.T:
                    s = states[tau + self.n]
                    try:
                        a = self.actions[tau + self.n]
                    except:
                        a = self.make_decision(state=s)
                    # Find bootstrap                    
                    G += self.gamma**self.n * self.Q.predict(s, a)      
                    # print('\tBootstrapping...', G)
                state = states[tau]
                action = self.actions[tau]
                # Update weights
                self.Q.learn(state, action, G, self.alpha)

        # Maintain working list of states and rewards
        states = self.states + [next_state]
        rewards = self.rewards + [reward]
        # Check end of episode
        if done:
            # Update T with t + 1 
            self.T = self.t + 1
            # print('Done\n', 'tau', self.tau, 't', self.t, 'T', self.T)
            # Update using remaining information
            for tau in range(self.tau, self.T - 1):
                update_tau(tau)
        else:
            # Set counter to present minus n rounds
            self.tau = self.t - self.n + 1
            # print('tau', self.tau, 't', self.t, 'T', self.T)
            # If present is greater than n, update
            if self.tau >= 0:
                update_tau(self.tau)
            # Update t for next iteration
            self.t += 1

