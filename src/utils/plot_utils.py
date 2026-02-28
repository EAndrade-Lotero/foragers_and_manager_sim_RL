import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Dict
from seaborn import lineplot, histplot, heatmap, color_palette

from utils.interpreters import id_state

class PlotGridValues :
    
    def __init__(
                self, 
                shape:Tuple[int, int], 
                dict_acciones:Dict[int, str], 
                axes_labels:Optional[List[str]]=[[],[]],
                flipud:Optional[bool]=False
            ) -> None:
        assert(len(shape) == 2)
        self.shape = shape
        self.dict_acciones = dict_acciones
        self.nA = len(dict_acciones.keys())
        self.axes_labels = axes_labels
        self.flipud = flipud
        
    def plot_policy(self, policy, V=None, ax=None):
        try:
            policy = np.array(policy).reshape(self.shape)
        except Exception as e:
            print('Source error:', e)
            raise Exception('Unreadable policy!')
        if self.flipud:
            policy = np.flipud(policy)
        annotations = np.vectorize(self.dict_acciones.get)(policy)
        if V is None:
            values = np.zeros(self.shape)
        else:
            try:
                values = np.array(V).reshape(self.shape)
            except:
                raise Exception('Valores no legibles!')
        if self.flipud:
            values = np.flipud(values)
        xticklabels = self.axes_labels[0]
        yticklabels = self.axes_labels[1]
        if ax is None:
            heatmap(
                values,
                annot=annotations,
                cbar=False,
                fmt="",
                cmap=color_palette("Blues", as_cmap=True),
                linewidths=1.7,
                linecolor="black",
                xticklabels=xticklabels,
                yticklabels=yticklabels,
                annot_kws={"fontsize": "xx-large"},
            ).set(title="Action per state")
            plt.plot()
        else:
            heatmap(
                values,
                annot=annotations,
                cbar=False,
                fmt="",
                cmap=color_palette("Blues", as_cmap=True),
                linewidths=1.7,
                linecolor="black",
                xticklabels=xticklabels,
                yticklabels=yticklabels,
                annot_kws={"fontsize": "xx-large"},
                ax = ax
            ).set(title="Action per state")

    def plot_V_values(self, V, ax=None):
        try:
            V = np.array(V).reshape(self.shape)
        except:
            raise Exception('Valores no legibles!')
        if self.flipud:
            V = np.flipud(V)
        xticklabels = self.axes_labels[0]
        yticklabels = self.axes_labels[1]
        if ax is None:
            heatmap(
                V,
                annot=True,
                cbar=True,
                fmt="",
                cmap=color_palette("Blues", as_cmap=True),
                linewidths=0.7,
                linecolor="black",
                xticklabels=xticklabels,
                yticklabels=yticklabels,
                annot_kws={"fontsize": "x-large"},
            ).set(title="V-values")
        else:
            heatmap(
                V,
                annot=True,
                cbar=True,
                fmt="",
                cmap=color_palette("Blues", as_cmap=True),
                linewidths=0.7,
                linecolor="black",
                xticklabels=xticklabels,
                yticklabels=yticklabels,
                annot_kws={"fontsize": "x-large"},
                ax = ax
            ).set(title="V-values")

    def plot_policy_and_values(self, policy, V):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        self.plot_policy(policy, V, ax=ax[0])
        self.plot_V_values(V, ax=ax[1])
        ax[0].set_title('Policy', fontsize='18')
        ax[1].set_title('Max values', fontsize='18')


class PlotQApprox2D :
    '''
    Plots a function approximation of Q values for a bidimensional
    state space.
    '''
    def __init__(
                self, 
                Q:any, 
                state_scales:List, 
                dict_acciones:Dict,
                mesh_size:Optional[int]=20,
                use_tensors:Optional[bool]=False,
                state_interpreter:Optional[any]=id_state
            ) -> None:
        assert(hasattr(Q, 'predict'))
        self.Q = Q
        assert(len(state_scales) == 2)
        self.state_scales = state_scales
        self.dict_acciones = dict_acciones
        self.mesh_size = mesh_size
        self.use_tensors = use_tensors
        self.state_interpreter = state_interpreter

    def best_actions(self):
        q_table = self.create_q_table()
        # Find best actions
        policy = np.argmax(q_table, axis=2)
        # create plot object
        axes_labels = self.create_labels() 
        plot = PlotGridValues(shape=(self.mesh_size,self.mesh_size), 
                              dict_acciones=self.dict_acciones, 
                              axes_labels=axes_labels)
        plot.plot_policy(policy=policy)

    def best_values(self, max_deci:Optional[int]=2):
        q_table = self.create_q_table()
        # Find best values
        values = np.max(q_table, axis=2)
        if max_deci == 0:
            values = np.vectorize(lambda x: int(x))(values)
        else:
            values = np.vectorize(lambda x: round(x, max_deci))(values)
        # create plot object
        axes_labels = self.create_labels(max_deci=max_deci) 
        plot = PlotGridValues(shape=(self.mesh_size,self.mesh_size), 
                              dict_acciones=self.dict_acciones,
                              axes_labels=axes_labels)
        plot.plot_V_values(V=values)

    def best_actions_with_values(self, max_deci:Optional[int]=2):
        q_table = self.create_q_table()
        # Find best actions
        policy = np.argmax(q_table, axis=2)
        # Find best values
        values = np.max(q_table, axis=2)
        if max_deci == 0:
            values = np.vectorize(lambda x: int(x))(values)
        else:
            values = np.vectorize(lambda x: round(x, max_deci))(values)
        # create plot object
        axes_labels = self.create_labels(max_deci=max_deci) 
        plot = PlotGridValues(shape=(self.mesh_size,self.mesh_size), 
                              dict_acciones=self.dict_acciones,
                              axes_labels=axes_labels)
        plot.plot_policy_and_values(policy=policy, V=values)

    def create_q_table(self):
        # Create mesh from bidimensional states
        mesh_size = self.mesh_size
        X, Y = self.create_mesh()
        # Create q table from Q function
        actions = list(self.dict_acciones.keys())
        n = len(actions)
        q_table = np.zeros((mesh_size, mesh_size, n))
        for j in range(mesh_size):
            for i in range(mesh_size):
                state = (X[j], Y[i])
                if self.use_tensors:
                    state = torch.tensor(state, dtype=torch.float32)
                state = self.state_interpreter(state)
                # print(f'state:{state}')
                if hasattr(self.Q, 'values_vector'):
                    q_table[(mesh_size - 1) - i, j, :] = self.Q.values_vector(state)
                else:
                    nA = len(self.dict_acciones.keys())
                    for a in range(nA):
                        q_table[(mesh_size - 1) - i, j, a] = self.Q.predict(state, a)
        return q_table

    def create_labels(self, max_deci:Optional[int]=2):
        X, Y = self.create_mesh()
        xticklabels = [str(round(x, max_deci)) for x in X]
        yticklabels = [str(round(y, max_deci)) for y in Y]
        return [xticklabels, yticklabels]

    def create_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        x_lims = self.state_scales[0]
        X = np.linspace(x_lims['min'], x_lims['max'], self.mesh_size)
        y_lims = self.state_scales[1]
        Y = np.linspace(y_lims['min'], y_lims['max'], self.mesh_size)
        return X, Y


class PlotPolicyApprox2D :
    '''
    Plots a policy for a bidimensional state space.
    '''
    def __init__(self, policy:any, 
                 state_scales:List, 
                 dict_acciones:Dict,
                 mesh_size:Optional[int]=20) -> None:
        assert(hasattr(policy, 'predict'))
        self.policy = policy
        assert(len(state_scales) == 2)
        self.state_scales = state_scales
        self.dict_acciones = dict_acciones
        self.mesh_size = mesh_size

    def create_policy_table(self):
        # Create mesh from bidimensional states
        mesh_size = self.mesh_size
        x_lims = self.state_scales[0]
        X = np.linspace(x_lims[0], x_lims[1], mesh_size)
        y_lims = self.state_scales[1]
        Y = np.linspace(y_lims[0], y_lims[1], mesh_size)
        # Create policy table from policy
        n = len(self.dict_acciones.keys())
        policy_table = np.zeros((mesh_size, mesh_size, n))
        for j in range(mesh_size):
            for i in range(mesh_size):
                state = (X[j], Y[i])
                policy_table[(mesh_size - 1) - i, j, :] = self.policy.predict(state)
        return policy_table

    def best_actions(self):
        policy_table = self.create_policy_table()
        # Find best actions
        policy = np.argmax(policy_table, axis=2)
        # create plot object
        plot = PlotGridValues((self.mesh_size,self.mesh_size), self.dict_acciones)
        plot.plot_policy(policy=policy)

    def best_values(self, max_deci:Optional[int]=2):
        policy_table = self.create_policy_table()
        # Find best values
        values = np.max(policy_table, axis=2)
        if max_deci == 0:
            values = np.vectorize(lambda x: int(x))(values)
        else:
            values = np.vectorize(lambda x: round(x, max_deci))(values)
        # create plot object
        plot = PlotGridValues((self.mesh_size,self.mesh_size), self.dict_acciones)
        plot.plot_V_values(V=values)

    def best_actions_with_values(self, max_deci:Optional[int]=2):
        policy_table = self.create_policy_table()
        # Find best actions
        policy = np.argmax(policy_table, axis=2)
        # Find best values
        values = np.max(policy_table, axis=2)
        if max_deci == 0:
            values = np.vectorize(lambda x: int(x))(values)
        else:
            values = np.vectorize(lambda x: round(x, max_deci))(values)
        # create plot object
        plot = PlotGridValues((self.mesh_size,self.mesh_size), self.dict_acciones)
        plot.plot_policy_and_values(policy=policy, V=values)



class Plot :
    '''
    Gathers a number of frequently used visualizations.
    '''

    def __init__(self, data:pd.DataFrame):
        self.data = data

    def plot_rewards(self, file:str=None) -> plt.axis:
        '''
        Plots the reward per episode.
        Input:
            - file, string with the name of file to save the plot on.
        Output:
            - axis, a plt object, or None.
        '''
        assert('model' in self.data.columns)
        assert('environment' in self.data.columns)
        models = self.data.model.unique()
        vs_models = True if len(models) > 1 else False
        fig, ax = plt.subplots(figsize=(4,3.5))
        data = self.data.copy()
        if 'simulation' in self.data.columns:
            data = data.groupby(['model', 'environment', 'simulation', 'episode'])["reward"].sum().reset_index()
        else:
            data = data.groupby(['model', 'environment', 'episode'])["reward"].sum().reset_index()
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total reward')
        ax.grid()
        if vs_models:
            ax = lineplot(x='episode', y='reward', hue='model', data=data, errorbar=('ci', 95))
        else:
            ax = lineplot(x='episode', y='reward', data=data, errorbar=('ci', 95))
        if file is not None:
            plt.savefig(file, dpi=300, bbox_inches="tight")
        return ax

    def plot_histogram_rewards(self, file:str=None) -> plt.axis:
        '''
        Plots a histogram with the sum of rewards per episode.
        Input:
            - file, string with the name of file to save the plot on.
        Output:
            - axis, a plt object, or None.
        '''
        assert('model' in self.data.columns)
        assert('environment' in self.data.columns)
        models = self.data.model.unique()
        vs_models = True if len(models) > 1 else False
        fig, ax = plt.subplots(figsize=(4,3.5))
        ax.set_xlabel('Sum of rewards')
        ax.set_ylabel('Frequency')
        ax.grid()
        if vs_models:
            df = self.data.groupby(['model','environment','episode']).reward.sum().reset_index()
            ax = histplot(x='reward', hue='model', data=df)
        else:
            df = self.data.groupby(['environment','episode']).reward.sum().reset_index()
            ax = histplot(x='reward', data=df)
        if file is not None:
            plt.savefig(file, dpi=300, bbox_inches="tight")
        df = self.data.groupby(['environment','model','episode']).reward.sum().reset_index()
        total_reward = df.groupby('model').reward.mean()
        print('Average sum of rewards:\n', total_reward)
        df = self.data.groupby(['environment','model','episode']).done.sum().reset_index()
        df["done"] = df["done"].astype(int)
        termination = df.groupby('model').done.mean()*100
        print('\nEpisode termination percentage:\n', termination)
        return ax
    
    def plot_round_reward(self, file:str=None) -> plt.axis:
        '''
        Plots the reward per round.
        Input:
            - file, string with the name of file to save the plot on.
        Output:
            - axis, a plt object, or None.
        '''
        assert('model' in self.data.columns)
        assert('environment' in self.data.columns)
        models = self.data.model.unique()
        vs_models = True if len(models) > 1 else False
        fig, ax = plt.subplots(figsize=(4,3.5))
        ax.set_xlabel('Round')
        ax.set_ylabel('Reward')
        ax.grid()
        data = self.data.copy()
        if vs_models:
            lineplot(x='round', y='reward', hue='model', data=data, ax=ax)
        else:
            ax = lineplot(x='round', y='reward', data=data, ax=ax)
        if file is not None:
            plt.savefig(file, dpi=300, bbox_inches="tight")
        return ax