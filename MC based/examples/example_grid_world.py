import sys
from grid_world import GridWorld
import random
import numpy as np

def MC_basic(env, gamma, delta, T, policy_matrix):
    q_table_sum = np.zeros((env.num_states, len(env.action_space)))
    q_table = np.zeros((env.num_states, len(env.action_space)))
    last_q_table = np.zeros((env.num_states, len(env.action_space)))
    reward_queue = np.zeros(T)
    num_visited = np.zeros((env.num_states, len(env.action_space)))

    while True:
        env.render()
        q_table_sum = np.zeros((env.num_states, len(env.action_space)))
        num_visited = np.zeros((env.num_states, len(env.action_space)))
        for i in range(env.num_states):
            ## 策略评估（用MC计算出当前策略下的 q value）
            for j in range(len(env.action_space)):
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
                q_table[i,j] = q_table_sum[i,j]/(num_visited[i,j]+1)
                num_visited[i,j] += 1
            ## 策略改进（先计算q_table，再更新策略）并判断值是否收敛
            new_action = np.argmax(q_table[i, :])
            policy_matrix[i] = np.eye(len(env.action_space))[new_action]

        for i in range(env.num_states):
            state_values[i] = np.max(q_table[i, :])

        # Add state values
        env.add_state_values(state_values)
        # Add policy
        policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis]  # make the sum of elements in each row to be 1
        env.add_policy(policy_matrix)

        if(np.all(np.abs(q_table - last_q_table) < delta)):
            last_q_table = q_table.copy()
            break
        last_q_table = q_table.copy()
    
    return state_values, policy_matrix

def MC_exploring_starts(env, gamma, delta, T, policy_matrix):
    q_table = np.zeros((env.num_states, len(env.action_space)))
    last_q_table = np.zeros((env.num_states, len(env.action_space)))
    returns = np.zeros((env.num_states, len(env.action_space)))
    num_visited = np.zeros((env.num_states, len(env.action_space)))
    state_values = np.zeros(env.num_states)

    # 保证每个状态动作对都被访问到
    for k in range(10):
        env.render()
        
        for i in range(env.num_states):
            for j in range(len(env.action_space)):
                returns = np.zeros((env.num_states, len(env.action_space)))
                num_visited = np.zeros((env.num_states, len(env.action_space)))
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
                    # first visit 方法
                    if (s, a) not in state_action_pairs[:t-1]:
                        returns[s, a] = g+returns[s, a]
                        q_table[s, a] = returns[s, a] / (num_visited[s, a] + 1)
                        num_visited[s, a] += 1

                        new_action = np.argmax(q_table[s, :])
                        policy_matrix[s] = np.eye(len(env.action_space))[new_action]

        for n in range(env.num_states):
            state_values[n] = np.max(q_table[n, :])

        # Add state values
        env.add_state_values(state_values)
        # Add policy
        policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis]  # make the sum of elements in each row to be 1
        env.add_policy(policy_matrix)

        if(np.all(np.abs(q_table - last_q_table) < delta)):
            last_q_table = q_table.copy()
            break
        last_q_table = q_table.copy()
    return state_values, policy_matrix

def MC_epsilon_greedy(env, gamma, T, epsilon, policy_matrix):
    q_table = np.zeros((env.num_states, len(env.action_space)))
    last_q_table = np.zeros((env.num_states, len(env.action_space)))
    returns = np.zeros((env.num_states, len(env.action_space)))
    num_visited = np.zeros((env.num_states, len(env.action_space)))
    state_values = np.zeros(env.num_states)

    while True:
        if epsilon > 0.01:
            epsilon *= 0.99
        env.render()

        returns = np.zeros((env.num_states, len(env.action_space)))
        num_visited = np.zeros((env.num_states, len(env.action_space)))

        state_idx = random.randint(0, env.num_states-1)
        action_idx = random.randint(0, len(env.action_space)-1)
        state_action_pairs = [(state_idx,action_idx)]
        rewards =  []
        state =  (state_idx%env.env_size[0],state_idx//env.env_size[0])
        action = env.action_space[action_idx]
        env.agent_state = state
        # 生成一条轨迹
        for t in range(T):
            next_state, reward = env._get_next_state_and_reward(state, action)
            #_ = env.step(action)
            x,y = next_state
            state = next_state
            action_idx = np.random.choice(len(env.action_space), p=policy_matrix[y*env.env_size[0]+x, :])
            action = env.action_space[action_idx]
            state_action_pairs.append((y*env.env_size[0]+x, action_idx))
            rewards.append(reward)
        g = 0
        # 计算每个状态动作对的回报
        for t in range(T, 0, -1):
            g = rewards[t-1] + gamma * g
            s = state_action_pairs[t-1][0]
            a = state_action_pairs[t-1][1]
            returns[s, a] = g+returns[s, a]
            q_table[s, a] = returns[s, a] / (num_visited[s, a] + 1)
            num_visited[s, a] += 1
            # 策略改进
            action_idx = np.argmax(q_table[s, :])
            policy_matrix[s][action_idx] = 1-epsilon+epsilon/len(env.action_space)
            for i in range(len(env.action_space)):
                if i != action_idx:
                    policy_matrix[s][i] = epsilon/len(env.action_space)

        # 计算状态值
        state_values = np.zeros(env.num_states)
        for i in range(env.num_states):
            for j in range(len(env.action_space)):
                state_values[i] += policy_matrix[i][j] * q_table[i][j]

        # Add state values
        env.add_state_values(state_values)
        # Add policy
        policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis]  # make the sum of elements in each row to be 1
        env.add_policy(policy_matrix)

        if(np.all(np.abs(q_table - last_q_table) < delta)):
            last_q_table = q_table.copy()
            break
        last_q_table = q_table.copy()

        # print(q_table)
    return state_values, policy_matrix
    

# Example usage:
if __name__ == "__main__":

    gamma = 0.9
    delta = 0.01
    T = 1000
    epsilon = 1
    env = GridWorld()
    state = env.reset()
    q_table = np.zeros((env.num_states, len(env.action_space)))
    state_values = np.zeros(env.num_states)
    last_state_values = np.zeros(env.num_states)
    
    policy_matrix = np.zeros((env.num_states, len(env.action_space)))
    #policy_matrix[:, 0] = 1
    for i in range(env.num_states):
        for j in range(len(env.action_space)):
            policy_matrix[i][j] = 1/len(env.action_space)

    converged = False
    policy_stable = False

    ## MC basic
    # state_values, policy_matrix = MC_basic(env=env, gamma=gamma, delta=delta, T=T, policy_matrix=policy_matrix)
    ## MC exploring starts
    # state_values, policy_matrix = MC_exploring_starts(env=env, gamma=gamma,delta=delta, T=T, policy_matrix=policy_matrix)
    ## MC epsilon greedy
    state_values, policy_matrix = MC_epsilon_greedy(env=env, gamma=gamma, T=T, epsilon=epsilon, policy_matrix=policy_matrix)    
        
    
    print("Optimal policy")
    env.render(animation_interval=20)