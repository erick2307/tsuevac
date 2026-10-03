#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

NUM_EPISODES = 100
NUM_STEPS = 100
EPSILON = 0.9
ALPHA = 0.9
GAMMA = 0.9

#Environment
NUM_NODES = 10
NUM_EDGES = 20


def sarsa():
    ""
    #initialize Q(s,a) amd Q(terminal)=0
    init_qvalues()
    terminal = init_terminal()
    for i in range(NUM_EPISODES):
        #Initialize S
        state = init_state()
        #Choose A from S using policy from Q
        action = select_action(state)
        for j in range(NUM_STEPS):
            #take action
            reward, new_state = exec_action(action)
            #Choose new A from new S using policy Q
            new_action = select_action(new_state)
            update_qvalues(state, action, reward, new_state, 
                           new_action, epsilon=EPSILON)
            state = new_state
            action = new_action
            if state == terminal:
                return
    return


def qlearning():
    ""
    #initialize Q(s,a) amd Q(terminal)=0
    q_values = init_qvalues()
    terminal = init_terminal()
    for i in range(NUM_EPISODES):
        #Initialize S
        state = init_state()
        for j in range(NUM_STEPS):
            #Choose A from S using policy from Q
            action = select_action(state, q_values, epsilon=0)
            #take action
            new_state = exec_action(state, action)
            reward = get_reward(new_state, terminal)
            update_qvalues(q_values, state, action, reward, new_state)
            state = new_state
            if state == terminal:
                return
    return


def init_qvalues(rows=10,cols=10,acts=4):
    q_values = np.zeros((rows, cols, acts))
    return q_values


def init_terminal():
    terminal_state = (10, 10)
    return terminal_state


def init_state():
    initial_state = (0, 0)
    return initial_state


def select_action(state, q_values,epsilon):
    return np.argmax(q_values[state[0], state[1]])


def exec_action(state, action):
    if action == 0: # move up
        new_state = (state[0],state[1]-1)
    if action == 1: # move right
        new_state = (state[0]+1,state[1])
    if action == 2: # move down
        new_state = (state[0],state[1]+1)
    if action == 3: # move left
        new_state = (state[0]-1,state[1])
    return new_state


def get_reward(new_state, terminal):
    if new_state == terminal:
        return 100.0
    else:
        return 1.0


def update_qvalues(q_values, state, action, reward, new_state, new_action):
    q_values[state[0], state[1], action] += ALPHA * (reward + GAMMA * 
                                                     q_values[new_state[0],
                                                              new_state[1],
                                                              new_action] -
                                                     q_values[state[0], 
                                                              state[1], 
                                                              action] )

