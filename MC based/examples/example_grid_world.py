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
    returns = np.zeros((env.num_states, len(env.action_space)))
    num_visited = np.zeros((env.num_states, len(env.action_space)))
    state_values = np.zeros(env.num_states)

    for i in range(env.num_states):
        for j in range(len(env.action_space)):
            state_action_pairs = [(i,j)]
            rewards =  []
            state =  (i%env.env_size[0],i//env.env_size[0])
            action = env.action_space[j]
            for t in range(T):
                next_state, reward = env._get_next_state_and_reward(state, action)
                x,y = next_state
                state = next_state
                action = env.action_space[np.argmax(policy_matrix[y*env.env_size[0]+x, :])]
                state_action_pairs.append((y*env.env_size[0]+x, np.argmax(policy_matrix[y*env.env_size[0]+x, :])))
                rewards.append(reward)
            g = 0
            for t in range(T, 0, -1):
                g = rewards[t-1] + gamma * g
                s = state_action_pairs[t-1][0]
                a = state_action_pairs[t-1][1]
                if (s, a) not in state_action_pairs[:t-1]:
                    returns[s, a] = g+returns[s, a]
                    q_table[s, a] = returns[s, a] / (num_visited[s, a] + 1)
                    num_visited[s, a] += 1

                    ## 需要一直迭代
                    policy_stable = False
                    old_action = np.argmax(policy_matrix[s, :])
                    new_action = np.argmax(q_table[s, :])
                    if old_action != new_action:
                        policy_stable = False
                    policy_matrix[s] = np.eye(len(env.action_space))[new_action]

                    for n in range(env.num_states):
                        state_values[n] = np.max(q_table[n, :])
    return state_values, policy_matrix, policy_stable

# Example usage:
if __name__ == "__main__":

    gamma = 0.9
    delta = 0.01
    T = 50
    env = GridWorld()
    state = env.reset()
    q_table = np.zeros((env.num_states, len(env.action_space)))
    state_values = np.zeros(env.num_states)
    last_state_values = np.zeros(env.num_states)
    
    policy_matrix = np.zeros((env.num_states, len(env.action_space)))
    policy_matrix[:, 0] = 1

    converged = False
    policy_stable = False

    while not policy_stable:
    # while True:
        env.render()    
        ## MC basic
        # state_values, policy_matrix, policy_stable = MC_basic(env=env, gamma=gamma, delta=delta, T=T, policy_matrix=policy_matrix)
        ## MC exploring starts
        state_values, policy_matrix, policy_stable = MC_exploring_starts(env=env, gamma=gamma, T=T, policy_matrix=policy_matrix)
        # state_values, policy_matrix = ExploringStarts_MC(env=env, gamma=gamma, episodes=1000, iterations=1000)
        # Add state values
        env.add_state_values(state_values)
        # Add policy
        policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis]  # make the sum of elements in each row to be 1
        env.add_policy(policy_matrix)
    
    print("Optimal policy")
    env.render(animation_interval=20)