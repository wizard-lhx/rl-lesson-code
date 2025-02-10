import sys
from grid_world import GridWorld
import random
import numpy as np

def MC_basic(env, gamma, delta, T, policy_matrix):
    q_table_sum = np.zeros((env.num_states, len(env.action_space)))
    q_table = np.zeros((env.num_states, len(env.action_space)))
    last_q_table = np.zeros((env.num_states, len(env.action_space)))
    reward_queue = np.zeros(T)
    num_iter = 1

    ## 策略评估（用MC计算出当前策略下的 q value）
    for i in range(env.num_states):
        for j in range(len(env.action_space)):
            while True:
                state =  (i%env.env_size[0],i//env.env_size[0])
                action = env.action_space[j]
                for t in range(T):
                    next_state, reward = env._get_next_state_and_reward(state, action)
                    state=next_state
                    x,y = state
                    action = env.action_space[np.argmax(policy_matrix[y*env.env_size[0]+x, :])]
                    reward_queue[t] = reward
                for t in range(T-1, 0, -1):
                    reward_queue[t-1] = reward_queue[t-1] + gamma * reward_queue[t]
                q_table_sum[i,j] = (reward_queue[0]+q_table_sum[i,j])
                q_table[i,j] = q_table_sum[i,j]/num_iter
                num_iter += 1
                if(np.abs(q_table[i,j] - last_q_table[i,j]) < delta):
                    last_q_table = q_table.copy()
                    break
                last_q_table = q_table.copy()
    
    ## 策略改进（先计算q_table，再更新策略）并判断策略是否收敛
    policy_stable = True
    for i in range(env.num_states):
        old_action = np.argmax(policy_matrix[i, :])
        new_action = np.argmax(q_table[i, :])
        if old_action != new_action:
            policy_stable = False
        policy_matrix[i] = np.eye(len(env.action_space))[new_action]
    
    for i in range(env.num_states):
        state_values[i] = np.max(q_table[i, :])

    return state_values, policy_matrix, policy_stable

def MC_exploring_starts(env, gamma, T, policy_matrix):
    q_table = np.zeros((env.num_states, len(env.action_space)))
    reward_queue = np.zeros(T+1)
    state_index_queue = []
    action_index_queue = []
    num_q_table_visit = np.zeros((env.num_states, len(env.action_space)))
    num_q_table_first_visit = np.zeros((env.num_states, len(env.action_space)))

    ## 策略评估（用MC计算出当前策略下的 q value）
    for i in range(env.num_states):
        for j in range(len(env.action_space)):
            state =  (i%env.env_size[0],i//env.env_size[0])
            action = env.action_space[j]
            state_index_queue.append(i)
            action_index_queue.append(j)
            num_q_table_first_visit.fill(0)
            for t in range(T):
                next_state, reward = env._get_next_state_and_reward(state, action)
                state=next_state
                x,y = state
                action = env.action_space[np.argmax(policy_matrix[y*env.env_size[0]+x, :])]
                state_index_queue.append(y*env.env_size[0]+x)
                action_index_queue.append(np.argmax(policy_matrix[y*env.env_size[0]+x, :]))
                reward_queue[t] = reward
            reward_queue[T] = 0
            state_index = state_index_queue.pop()
            action_index = action_index_queue.pop()
            for t in range(T, 0, -1):
                reward_queue[t-1] = reward_queue[t-1] + gamma * reward_queue[t]
            for t in range(T):
                state_index = state_index_queue.pop(0)
                action_index = action_index_queue.pop(0)
                if(num_q_table_first_visit[state_index,action_index] == 1):
                    continue
                num_q_table_visit[state_index,action_index] += 1
                if state_index == 24 and action_index == 4:
                    print(reward_queue[t])
                q_table[state_index,action_index] = reward_queue[t]+q_table[state_index,action_index]
                num_q_table_first_visit[state_index,action_index] = 1
    q_table /= num_q_table_visit

    ## 策略改进（先计算q_table，再更新策略）并判断策略是否收敛
    policy_stable = True
    for i in range(env.num_states):
        old_action = np.argmax(policy_matrix[i, :])
        new_action = np.argmax(q_table[i, :])
        if old_action != new_action:
            policy_stable = False
        policy_matrix[i] = np.eye(len(env.action_space))[new_action]
    
    for i in range(env.num_states):
        state_values[i] = np.max(q_table[i, :])
    print(q_table)

    return state_values, policy_matrix, policy_stable

# Example usage:
if __name__ == "__main__":

    gamma = 0.9
    delta = 0.01
    T = 15
    env = GridWorld()
    state = env.reset()
    q_table = np.zeros((env.num_states, len(env.action_space)))
    state_values = np.zeros(env.num_states)
    last_state_values = np.zeros(env.num_states)
    
    policy_matrix = np.zeros((env.num_states, len(env.action_space)))
    policy_matrix[:, 2] = 1

    converged = False
    policy_stable = False

    while not policy_stable:
    # while True:
        env.render()    
        ## MC basic
        # state_values, policy_matrix, policy_stable = MC_basic(env=env, gamma=gamma, delta=delta, T=T, policy_matrix=policy_matrix)
        ## MC exploring starts
        state_values, policy_matrix, policy_stable = MC_exploring_starts(env=env, gamma=gamma, T=T, policy_matrix=policy_matrix)
        
        # Add state values
        env.add_state_values(state_values)
        # Add policy
        policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis]  # make the sum of elements in each row to be 1
        env.add_policy(policy_matrix)
    
    print("Optimal policy")
    env.render(animation_interval=20)