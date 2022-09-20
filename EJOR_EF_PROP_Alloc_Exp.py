'''
  File:   EJOR_Alloc_Exp.py
  Author: Nicholas Mattei (nsmattei@gmail.com)
  Date: April 13, 2022

  About
  --------------------
    Simple file to run experiments for our Discrete Alloc paper

'''

import numpy as np
# Load MatPlotLib
import matplotlib
import matplotlib.pyplot as plt
# Load Pandas
import pandas as pd
# Load Stats
from scipy import stats
import seaborn as sns
import gurobipy as gpy
import itertools
# Basic wall time
import time
import random
import itertools
from tqdm import tqdm

"""
Uses the sequential dynamic programming function to find a proportional allocation
of items to agents with different valuations, with a largest sum of utilities (utilitarian value).
The input is a valuation-matrix v, where v[i][j] is the value of agent i to item j.
The states are of the form  (v1, v2, ..., vn) where n is the number of agents.
The "vi" are the value of bundle i to agent i.
Programmer: Erel Segal-Halevi
Since: 2021-12
"""

import dynprog
from dynprog.sequential import SequentialDynamicProgram

import math, logging
from typing import *

logger = logging.getLogger(__name__)


#####
## Erel DP Block
#####

def items_as_value_vectors(valuation_matrix):
    """
    Convert a valuation matrix (an input to a fair division algorithm) into a list of value-vectors.
    Each value-vector v represents an item: v[i] is the value of the item for agent i (i = 0,...,n-1).
    The last element, v[n], is the item index.
    >>> items_as_value_vectors([[11,22,33],[44,55,66]])
    [[11, 44, 0], [22, 55, 1], [33, 66, 2]]
    """
    num_of_agents   = len(valuation_matrix)
    num_of_items    = len(valuation_matrix[0])
    return [  # Each item is represented by a vector of values - a value for each agent. The last value is the item index.
        [valuation_matrix[agent_index][item_index] for agent_index in range(num_of_agents)] + [item_index]
        for item_index in range(num_of_items)
    ]

def add_input_to_bin_sum(bin_sums:list, bin_index:int, input:int):
    """
    Adds the given input integer to bin #bin_index in the given list of bins.
    >>> add_input_to_bin_sum([11, 22, 33], 0, 77)
    (88, 22, 33)
    >>> add_input_to_bin_sum([11, 22, 33], 1, 77)
    (11, 99, 33)
    >>> add_input_to_bin_sum([11, 22, 33], 2, 77)
    (11, 22, 110)
    """
    new_bin_sums = list(bin_sums)
    new_bin_sums[bin_index] = new_bin_sums[bin_index] + input
    return tuple(new_bin_sums)

def add_input_to_agent_value(agent_values:list, agent_index:int, item_values:list):
    """
    Update the state of a dynamic program by giving an item to a specific agent.
    :param agent_values: the current vector of agent values, before adding the new item.
    :param agent_index: the agent to which the item is given.
    :param item_values: a list of values: input[i] represents the value of the current item for agent i.
    >>> add_input_to_agent_value([11, 22, 33], 0, [55,66,77,1])
    (66, 22, 33)
    >>> add_input_to_agent_value([11, 22, 33], 1, [55,66,77,1])
    (11, 88, 33)
    >>> add_input_to_agent_value([11, 22, 33], 2, [55,66,77,1])
    (11, 22, 110)
    """
    return add_input_to_bin_sum(agent_values, agent_index, item_values[agent_index])

def add_input_to_bin(bins:list, agent_index:int, item_index:int):
    """
    Update the solution of a dynamic program by giving an item to a specific agent.
    
    :param bins: the current vector of agent bundles, before adding the new item.
    :param agent_index: the agent to which the item is given.
    :param item_index: the index of the given item.
    Adds the given input integer to bin #agent_index in the given list of bins.
    >>> add_input_to_bin([[11,22], [33,44], [55,66]], 1, 1)
    [[11, 22], [33, 44, 1], [55, 66]]
    """
    new_bins = list(bins)
    new_bins[agent_index] = new_bins[agent_index]+[item_index]
    return new_bins

def utilitarian_proportional_value(valuation_matrix):
    """
    Returns the maximum utilitarian value in a proportional allocation - does *not* return the partition itself.
    Returns -inf if there is no proportional allocation.
    >>> dynprog.sequential.logger.setLevel(logging.WARNING)
    >>> logger.setLevel(logging.WARNING)
    >>> utilitarian_proportional_value([[11,0,11],[33,44,55]])
    110
    >>> utilitarian_proportional_value([[11,22,33,44],[44,33,22,11]])
    154
    >>> utilitarian_proportional_value([[11,0,11,11],[0,11,11,11],[33,33,33,33]])
    88
    >>> utilitarian_proportional_value([[11],[22]])  # no proportional allocation
    -inf
    """
    items = items_as_value_vectors(valuation_matrix)
    return UMPropPartitionDP(valuation_matrix).max_value(items)

def utilitarian_proportional_allocation(valuation_matrix):
    """
    Returns the utilitarian-maximum proportional allocation and its utilitarian value.
    Raises an exception if there is no proportional allocation.
    >>> dynprog.sequential.logger.setLevel(logging.WARNING)
    >>> utilitarian_proportional_allocation([[11,0,11],[0,11,22]])
    (44, [[0], [1, 2]])
    >>> utilitarian_proportional_allocation([[11,22,33,44],[44,33,22,11]])
    (154, [[2, 3], [0, 1]])
    >>> utilitarian_proportional_allocation([[11,0,11,11],[0,11,11,11],[33,33,33,33]])[0]
    88
    >>> utilitarian_proportional_allocation([[11],[11]]) 
    Traceback (most recent call last):
    ...
    ValueError: No proportional allocation
    >>> utilitarian_proportional_allocation([[37,20,34,12,71,17,55,97,79],[57,5,59,63,92,23,4,36,69],[16,3,41,42,68,47,60,39,17]])
    (556, [[1, 7, 8], [0, 3, 4], [2, 5, 6]])
    """
    items = items_as_value_vectors(valuation_matrix)
    (best_state,best_value,best_solution,num_of_states) = UMPropPartitionDP(valuation_matrix).max_value_solution(items)
    # if best_value==-math.inf:
        # raise ValueError("No proportional allocation")
    return (best_value,best_solution)

#### Dynamic program definition for UM

class UMPropPartitionDP(SequentialDynamicProgram):

    # The states are of the form  (v1, v2, ..., vn) where n is the number of agents.
    # The "vi" are the value of bundle i to agent i.

    def __init__(self, valuation_matrix):
        num_of_agents = self.num_of_agents = len(valuation_matrix)
        self.thresholds = [sum(valuation_matrix[i])/num_of_agents for i in range(num_of_agents)]

    def initial_states(self):
        zero_values = self.num_of_agents*(0,)
        return {zero_values}

    def initial_solution(self):
        empty_bundles = [ [] for _ in range(self.num_of_agents)]
        return empty_bundles
   
    def transition_functions(self):
        return [
            lambda state, input, agent_index=agent_index: add_input_to_agent_value(state, agent_index, input)
            for agent_index in range(self.num_of_agents)
        ]

    def construction_functions(self):
        return [
            lambda solution,input,agent_index=agent_index: add_input_to_bin(solution, agent_index, input[-1])
            for agent_index in range(self.num_of_agents)
        ]

    def value_function(self):
        return lambda state: sum(state) if self._is_proportional(state) else -math.inf
    
    def _is_proportional(self, bundle_values:list)->bool:
        return all([bundle_values[i] >= self.thresholds[i] for i in range(self.num_of_agents)])


##### DP For EF.

def utilitarian_envyfree_value(valuation_matrix):
    """
    Returns the maximum utilitarian value in an envy-free allocation - does *not* return the allocation itself.
    Returns -inf if there is no envy-free allocation.
    >>> logger.setLevel(logging.INFO)
    >>> dynprog.sequential.logger.setLevel(logging.INFO)
    >>> utilitarian_envyfree_value([[11,0,11],[33,44,55]])
    110.0
    >>> logger.setLevel(logging.WARNING)
    >>> dynprog.sequential.logger.setLevel(logging.WARNING)
    >>> utilitarian_envyfree_value([[11,22,33,44],[44,33,22,11]])
    154.0
    >>> utilitarian_envyfree_value([[11,0,11,11],[0,11,11,11],[33,33,33,33]])
    88.0
    >>> utilitarian_envyfree_value([[11],[22]])  # no envy-free allocation
    -inf
    """
    items = items_as_value_vectors(valuation_matrix)
    return UMEFPartitionDP(valuation_matrix).max_value(items)

#### Dynamic program definition:

class UMEFPartitionDP(SequentialDynamicProgram):

    # The states are the bundle-differences: (d11, d12, ..., dnn) where n is the number of agents.
    # where dij := vi(Ai)-vi(Aj).

    def __init__(self, valuation_matrix):
        self.num_of_agents = len(valuation_matrix)
        self.valuation_matrix = valuation_matrix
        self.sum_valuation_matrix =  sum(map(sum,valuation_matrix))

    def initial_states(self):
        zero_differences = self.num_of_agents * (self.num_of_agents*(0,),)
        return {zero_differences}

    def initial_solution(self):
        empty_bundles = [ [] for _ in range(self.num_of_agents)]
        return empty_bundles
   
    def transition_functions(self):
        return [
            lambda state, input, agent_index=agent_index: _update_bundle_differences(state, agent_index, input)
            for agent_index in range(self.num_of_agents)
        ]

    def construction_functions(self):
        return [
            lambda solution,input,agent_index=agent_index: add_input_to_bin(solution, agent_index, input[-1])
            for agent_index in range(self.num_of_agents)
        ]

    def value_function(self):
        return lambda state: (sum(map(sum,state))+self.sum_valuation_matrix) / self.num_of_agents if self._is_envyfree(state) else -math.inf
    
    def _is_envyfree(self, bundle_differences:list)->bool:
        return all([bundle_differences[i][j]>=0 for i in range(self.num_of_agents) for j in range(self.num_of_agents)])

def _update_bundle_differences(bundle_differences, agent_index, item_values):
    """
    >>> _update_bundle_differences( ((0,0),(0,0)), 0, [11,33,0]  )
    ((0, 11), (-33, 0))
    >>> _update_bundle_differences( ((0,0),(0,0)), 1, [11,33,0]  )
    ((0, -11), (33, 0))
    """
    num_of_agents = len(bundle_differences)
    new_bundle_differences = [list(d) for d in bundle_differences]
    for other_agent_index in range(num_of_agents):
        if other_agent_index==agent_index: continue
        new_bundle_differences[agent_index][other_agent_index] += item_values[agent_index]
        new_bundle_differences[other_agent_index][agent_index] -= item_values[other_agent_index]
    new_bundle_differences = tuple((tuple(d) for d in new_bundle_differences))
    return new_bundle_differences


#### 
#
## Nick MIP Block
#
####

# UM within Prop model

## Pass in a Matrix of POSITIVE BIDS and NEGATIVE for COI!

def UM_PROP_capacitated_assignment(m, pd_bid_matrix, object_caps, agent_caps, agents, objects):

    # Dicts to keep track of varibles...
    assigned = {}
    utility = {}

    #NOTE THAT THESE ARE BINARY SO WE CAN ONLY ASSIGN EACH AGENT ONCE!!
    # Create a binary variable for every agent/object.
    for a in agents:
        for o in objects:
            assigned[a,o] = m.addVar(vtype=gpy.GRB.BINARY, name='assigned_%s_%s' % (a,o))

    # Create a variable for each agent's utility.
    for a in agents:
        utility[a] = m.addVar(vtype=gpy.GRB.CONTINUOUS, name='utility_%s' % (a))

    # add the variables to the model.
    m.update()

    # Agents can't be assigned negitive objects (no preference).
    for a in agents:
        for o in objects:
            if pd_bid_matrix.loc[a,o] == -1:
                m.addConstr(assigned[a,o] == 0)

    # Enforce that items can only be allocated o times each..
    for o in objects:
        m.addConstr(gpy.quicksum(assigned[a,o] for a in agents) >= object_caps[o][0], 'object_min_cap_%s' % (o))
        m.addConstr(gpy.quicksum(assigned[a,o] for a in agents) <= object_caps[o][1], 'object_max_cap_%s' % (o))

    # Enforce that each agent can't have more than agent_cap items.
    for a in agents:
        m.addConstr(gpy.quicksum(assigned[a,o] for o in objects) >= agent_caps[a][0], 'agent_min_cap_%s' % (a))
        m.addConstr(gpy.quicksum(assigned[a,o] for o in objects) <= agent_caps[a][1], 'agent_max_cap_%s' % (a))

    # Enforce the agent utility computations.
    for a in agents:
        m.addConstr(gpy.quicksum(assigned[a,o] * pd_bid_matrix.loc[a,o] for o in objects) == utility[a], 'agent_%s_utility' % (a))

    m.update()
    
    # Add in a constraint that we have proportionality, specifically that for each agent:
    # u_i(p(i)) \geq ui(o) / n
    for a in agents:
        m.addConstr(utility[a] >= float(pd_bid_matrix.loc[a].sum()) / len(agents))
    
    # Util SW
    m.setObjective(gpy.quicksum(utility[a] for a in agents), gpy.GRB.MAXIMIZE)
    return m, assigned, utility

## EF Within UM Model.

## Pass in a Matrix of POSITIVE BIDS and NEGATIVE for COI!

def UM_EF_capacitated_assignment(m, pd_bid_matrix, object_caps, agent_caps, agents, objects):

    # Dicts to keep track of varibles...
    assigned = {}
    utility = {}

    #NOTE THAT THESE ARE BINARY SO WE CAN ONLY ASSIGN EACH AGENT ONCE!!
    # Create a binary variable for every agent/object.
    for a in agents:
        for o in objects:
            assigned[a,o] = m.addVar(vtype=gpy.GRB.BINARY, name='assigned_%s_%s' % (a,o))

    # Create a variable for each agent's utility.
    for a in agents:
        utility[a] = m.addVar(vtype=gpy.GRB.CONTINUOUS, name='utility_%s' % (a))

    # add the variables to the model.
    m.update()

    # Agents can't be assigned negitive objects (no preference).
    for a in agents:
        for o in objects:
            if pd_bid_matrix.loc[a,o] == -1:
                m.addConstr(assigned[a,o] == 0)

    # Enforce that items can only be allocated o times each..
    for o in objects:
        m.addConstr(gpy.quicksum(assigned[a,o] for a in agents) >= object_caps[o][0], 'object_min_cap_%s' % (o))
        m.addConstr(gpy.quicksum(assigned[a,o] for a in agents) <= object_caps[o][1], 'object_max_cap_%s' % (o))

    # Enforce that each agent can't have more than agent_cap items.
    for a in agents:
        m.addConstr(gpy.quicksum(assigned[a,o] for o in objects) >= agent_caps[a][0], 'agent_min_cap_%s' % (a))
        m.addConstr(gpy.quicksum(assigned[a,o] for o in objects) <= agent_caps[a][1], 'agent_max_cap_%s' % (a))

    # Enforce the agent utility computations.
    for a in agents:
        m.addConstr(gpy.quicksum(assigned[a,o] * pd_bid_matrix.loc[a,o] for o in objects) == utility[a], 'agent_%s_utility' % (a))

    m.update()
    
    # Add in a constraint that we have EnvyFREENESS, specifically that for each agent:
    # u_i(p(i)) >= u_i(p(j)) for all i,j in N
    for i,j in itertools.combinations(agents, 2):
        # Note we need i,j and j,i because this isn't symmetric...
        m.addConstr(utility[i] >= gpy.quicksum(assigned[j,o] * pd_bid_matrix.loc[i,o] for o in objects))
        m.addConstr(utility[j] >= gpy.quicksum(assigned[i,o] * pd_bid_matrix.loc[j,o] for o in objects))
    
    # Util SW
    m.setObjective(gpy.quicksum(utility[a] for a in agents), gpy.GRB.MAXIMIZE)
    return m, assigned, utility



### OLD CODE FOR GENERATING MALLOWS

# Generate a Mallows model with the various mixing parameters passed in
# nvoters is the number of votes we need
# candmap is a candidate map
# mix is an array such that sum(mix) == 1 and describes the distro over the models
# phis is an array len(phis) = len(mix) = len(refs) that is the phi for the particular model
# refs is an array of dicts that describe the reference ranking for the set.
def gen_mallows(nvoters, ncands, phi):	
    # Get the insertion distance values.
    m_insert_dists = compute_mallows_insertvec_dist(ncands, phi)

    # Generate a reference ranking of the candidates
    ref_ranking = list(range(1, ncands+1))
    random.shuffle(ref_ranking)
    
    # Now, generate votes...
    votemap = {}
    for cvoter in range(nvoters):
        #Generate a vote
        insvec = [0] * ncands
        for i in range(1, len(insvec)+1):
            insvec[i-1] = draw(list(range(1, i+1)), m_insert_dists[i])
	
        order = []
        for i in range(ncands):
            order.insert(insvec[i]-1, ref_ranking[i])
        votemap[cvoter] = order
    
    return votemap

def compute_mallows_insertvec_dist(ncand, phi):
	#Compute the Various Mallows Probability Distros
	vec_dist = {}
	for i in range(1, ncand+1):
		#Start with an empty distro of length i
		dist = [0] * i
		#compute the denom = phi^0 + phi^1 + ... phi^(i-1)
		denom = sum([pow(phi,k) for k in range(i)])
		#Fill each element of the distro with phi^i-j / denom
		for j in range(1, i+1):
			dist[j-1] = pow(phi, i - j) / denom
		#print(str(dist) + "total: " + str(sum(dist)))
		vec_dist[i] = dist
	return vec_dist

# This should be refactored to use np..
def draw(values, distro):
	#Return a value randomly from a given discrete distribution.
	#This is a bit hacked together -- only need that the distribution
	#sums to 1.0 within 5 digits of rounding.
	if round(sum(distro),5) != 1.0:
		print("Input Distro is not a Distro...")
		print(str(distro) + "  Sum: " + str(sum(distro)))
		exit()
	if len(distro) != len(values):
		print("Values and Distro have different length")

	cv = 0	
	draw = random.random() - distro[cv]
	while draw > 0.0:
		cv+= 1
		draw -= distro[cv]
	return values[cv]		

if __name__ == "__main__":
    # Generate Some Data

    # ## Build a DataFrame for it..
    # data = {"Agent1": {"Obj1": 4, "Obj2": 3, "Obj3": 2, "Obj4": 1},
    #     "Agent2": {"Obj1": 1, "Obj2": 2, "Obj3": 3, "Obj4": 4},
    #    }
    # df_bid_matrix = pd.DataFrame.from_dict(data, orient="index")

    output_file_name = "./PaperRun_EF-Prop.csv"
    agent_steps = [2, 3, 4, 5, 6, 7]
    # Object Steps are Multiples of 
    object_steps = [1]
    phis = [0.5, 0.75, 1.0]
    iterations = 50
    solveable_only = False
    # Set Random Seed
    random.seed(22)

    all_results = []

    for a,o,p in tqdm(itertools.product(agent_steps, object_steps, phis)):
        print((a,o,p))
        for sample_count in tqdm(range(iterations)):
            result = {}
            result["Agents"] = a
            result["Objects"] = int(a*o)
            result["Phi"] = p
            result["Sample"] = sample_count


            mallows_orders = gen_mallows(a, int(a*o), p)
            # Convert Mallows Orders into a Borda Scored Matrix...
            data = {}
            for current_agent, order in mallows_orders.items():
                pref = {}
                for i in range(len(order)):
                    pref["Obj" + str(order[i])] = len(order) - i - 1
                data["Agent" + str(current_agent)] = pref
            # print(pd.DataFrame.from_dict(data, orient="index"))
            df_bid_matrix = pd.DataFrame.from_dict(data, orient="index")
            #print(df_bid_matrix)

            # Call the allocation function with this set...
            # A simple call...
            agents = list(df_bid_matrix.index)
            objects = list(df_bid_matrix.columns)
            # Make agent and object capacities
            object_caps = {i:(1,1) for i in objects}
            agent_caps = {i:(0,len(objects)) for i in agents}
    
            start = time.time()
            m = gpy.Model('Util')
            m, assigned, utility = UM_EF_capacitated_assignment(m, df_bid_matrix, object_caps, agent_caps, agents, objects)
            m.setParam(gpy.GRB.Param.OutputFlag, False )
            m.optimize()
            end = time.time()
            if m.Status == gpy.GRB.OPTIMAL:
                result["UMinEF_Solution"] = 1
            else:
                result["UMinEF_Solution"] = 0
            result["GRB-UMinEF-Time"] = end - start   
            # print("Elapsed Time {}".format(end - start))

            start = time.time()
            m = gpy.Model('Util')
            m, assigned, utility = UM_PROP_capacitated_assignment(m, df_bid_matrix,object_caps, agent_caps, agents, objects)
            m.setParam(gpy.GRB.Param.OutputFlag, False )
            m.optimize()
            end = time.time()
            if m.Status == gpy.GRB.OPTIMAL:
                result["UMinPO_Solution"] = 1
            else:
                result["UMinPO_Solution"] = 0
            result["GRB-UMinPO-Time"] = end - start  
            # print("Elapsed Time {}".format(end - start))

            if o < 10:
                start = time.time()
                x = utilitarian_proportional_allocation(df_bid_matrix.to_numpy())
                end = time.time()
                # print("Elapsed Time {}".format(end - start))
                result["DP-UMinPO-Time"] = end - start

                start = time.time()
                x = utilitarian_envyfree_value(df_bid_matrix.to_numpy())
                end = time.time()
                # print("Elapsed Time {}".format(end - start))
                result["DP-UMinEF-Time"] = end - start
            else:
                result["DP-UMinPO-Time"] = np.nan
                result["DP-UMinEF-Time"] = np.nan

            all_results.append(result)

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(output_file_name, index=False)
