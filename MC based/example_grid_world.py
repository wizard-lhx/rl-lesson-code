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

def generate_episode(env, policy):
    episode = []
    state = env.pos2state(env.reset())
    while True:
        action = np.random.choice(len(policy[state]), p=policy[state])
        next_pos, reward, done, _ = env.step(env.action_space[action])
        next_state = env.pos2state(next_pos)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode

def policy_update(policy, Q, state, epsilon):
    action_star = np.argmax(Q[state])
    # epsilon-greedy policy
    policy[state][action_star] = 1 - epsilon + epsilon/len(env.action_space)
    for i in range(len(env.action_space)):
        if i != action_star:
            policy[state][i] = epsilon / len(env.action_space)
    return policy

def MC_epsilon_greedy(env, gamma = 0.9, epsilon = 0.1, num_episode = 100):
    Q = np.zeros((env.num_states, len(env.action_space)))
    policy = np.ones((env.num_states, len(env.action_space))) / len(env.action_space)

    for i in range(num_episode):
        returns = np.zeros((env.num_states, len(env.action_space)))
        num_visited = np.zeros((env.num_states, len(env.action_space)))
        episode = generate_episode(env, policy)
        G = 0
        for t in range(len(episode)-1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward
            
            returns[state, action] = returns[state, action] + G
            num_visited[state, action] += 1
            Q[state, action] = returns[state, action] / num_visited[state, action]
            policy = policy_update(policy, Q, state, epsilon)
        print("Episode: ", i)
    return Q, policy

# Example usage:
if __name__ == "__main__":

    gamma = 0.9
    delta = 0.01
    T = 1000
    epsilon = 0.1
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
    env.render()
    Q, policy = MC_epsilon_greedy(env=env, gamma=0.9, epsilon=0.1, num_episode=100)
    
    env.add_policy(policy)
    print("Optimal policy")
    env.render(animation_interval=20)