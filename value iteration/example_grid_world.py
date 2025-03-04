import sys
from grid_world import GridWorld
import random
import numpy as np

def value_iteration(env, gamma, epsilon, state_values, last_state_values):
    policy_matrix = np.zeros((env.num_states, len(env.action_space)))

    ## 策略更新
    ## 先对每个状态，计算每个动作的Q值（注意(x,y)坐标与数组索引顺序是相反的）
    for i in range(env.num_states):
        for j in range(len(env.action_space)):
            next_state, reward = env._get_next_state_and_reward((i%env.env_size[0],i//env.env_size[0]), env.action_space[j])
            x,y=next_state
            q_table[i, j] = reward + gamma * state_values[y*env.env_size[0]+x]

    for i in range(env.num_states):
        policy_matrix[i, np.argmax(q_table[i, :])] = 1
    
    ## 状态值更新
    state_values = np.max(q_table, axis=1)
    
    ## 判断是否收敛
    if(np.all(np.abs(state_values - last_state_values) < epsilon)):
        print("Converged")
        return state_values, policy_matrix, True
    else:
        return state_values, policy_matrix, False

def policy_iteration(env, gamma, epsilon, policy_matrix):
    state_values = np.zeros(env.num_states)
    last_state_values = np.zeros(env.num_states)

    ## 策略评估（用贝尔曼公式迭代计算出当前策略下的 state value）
    while True:
        for i in range(env.num_states):
            next_state, reward = env._get_next_state_and_reward((i%env.env_size[0],i//env.env_size[0]), env.action_space[np.argmax(policy_matrix[i, :])])
            x,y=next_state
            state_values[i] = reward + gamma * state_values[y*env.env_size[0]+x]
        if(np.all(np.abs(state_values - last_state_values) < epsilon)):
            break
        last_state_values = state_values.copy()
    
    ## 策略改进（先计算q_table，再更新策略）并判断策略是否收敛
    policy_stable = True
    for i in range(env.num_states):
        for j in range(len(env.action_space)):
            next_state, reward = env._get_next_state_and_reward((i%env.env_size[0],i//env.env_size[0]), env.action_space[j])
            x,y=next_state
            q_table[i, j] = reward + gamma * state_values[y*env.env_size[0]+x]
    for i in range(env.num_states):
        old_action = np.argmax(policy_matrix[i, :])
        new_action = np.argmax(q_table[i, :])
        if old_action != new_action:
            policy_stable = False
        policy_matrix[i] = np.eye(len(env.action_space))[new_action]

    return state_values, policy_matrix, policy_stable

def truncated_policy_iteration(env, gamma, truncated, policy_matrix, state_values):
    ## state value 进行迭代计算时进行截断，并在下一次迭代时使用上一次的 state value 进行计算
    for t in range(truncated):
        for i in range(env.num_states):
            next_state, reward = env._get_next_state_and_reward((i%env.env_size[0],i//env.env_size[0]), env.action_space[np.argmax(policy_matrix[i, :])])
            x,y=next_state
            state_values[i] = reward + gamma * state_values[y*env.env_size[0]+x]
    
    policy_stable = True
    for i in range(env.num_states):
        for j in range(len(env.action_space)):
            next_state, reward = env._get_next_state_and_reward((i%env.env_size[0],i//env.env_size[0]), env.action_space[j])
            x,y=next_state
            q_table[i, j] = reward + gamma * state_values[y*env.env_size[0]+x]
    for i in range(env.num_states):
        old_action = np.argmax(policy_matrix[i, :])
        new_action = np.argmax(q_table[i, :])
        if old_action != new_action:
            policy_stable = False
        policy_matrix[i] = np.eye(len(env.action_space))[new_action]

    return state_values, policy_matrix, policy_stable

# Example usage:
if __name__ == "__main__":

    gamma = 0.9
    epsilon = 0.01
    truncated = 10
    env = GridWorld()
    state = env.reset()
    q_table = np.zeros((env.num_states, len(env.action_space)))
    state_values = np.zeros(env.num_states)
    last_state_values = np.zeros(env.num_states)
    
    policy_matrix = np.zeros((env.num_states, len(env.action_space)))
    policy_matrix[:, 4] = 1

    converged = False
    policy_stable = False

    while not policy_stable:
        env.render()
        ## 值迭代
        # state_values, policy_matrix, converged = value_iteration(env=env, gamma=gamma, epsilon=epsilon, state_values=state_values,last_state_values=last_state_values)
        # last_state_values = state_values.copy()
        ## 截断策略迭代
        state_values, policy_matrix, policy_stable = truncated_policy_iteration(env=env, gamma=gamma,  truncated = truncated, policy_matrix=policy_matrix, state_values=state_values)
        ## 策略迭代
        # state_values, policy_matrix, policy_stable = policy_iteration(env=env, gamma=gamma, epsilon=epsilon, policy_matrix=policy_matrix)
        
        # Add state values
        env.add_state_values(state_values)
        # Add policy
        policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis]  # make the sum of elements in each row to be 1
        env.add_policy(policy_matrix)
    
    print("Optimal policy")
    env.render(animation_interval=20)