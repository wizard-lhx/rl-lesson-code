from grid_world import GridWorld
import random
import numpy as np
import matplotlib.pyplot as plt

def draw_metrics(total_rewards, episodes_length):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].plot(total_rewards)
    ax[0].set_title("Total rewards")
    ax[1].plot(episodes_length)
    ax[1].set_title("Episodes length")
    plt.show()

def policy_update(policy, Q, state, epsilon):
    # optimal action
    action_star = np.argmax(Q[state, :])
    # epsilon-greedy policy
    policy[state][action_star] = 1 - epsilon + epsilon/len(env.action_space)
    for i in range(len(env.action_space)):
        if i != action_star:
            policy[state][i] = epsilon / len(env.action_space)
    return policy

def Sarsa(env, num_episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    # for animation
    total_rewards = np.zeros(num_episodes)
    episodes_length = np.zeros(num_episodes)
    # algorithm
    Q = np.zeros((env.num_states, len(env.action_space)))
    policy = np.random.rand(env.num_states, len(env.action_space))
    policy /= policy.sum(axis=1)[:, np.newaxis]  # make the sum of elements in each row to be 1
    for i in range(num_episodes):
        # state = np.random.choice(env.num_states)
        state = env.pos2state(env.reset())
        action = np.random.choice(len(env.action_space), p=policy[state, :])
        done = False
        while not done:
            next_pos, reward, done, _ = env.step(env.action_space[action])
            next_state = env.pos2state(next_pos)
            next_action = np.random.choice(len(env.action_space), p=policy[next_state, :])
            # value update
            Q[state, action] -= alpha * (Q[state, action] - (reward + gamma * Q[next_state, next_action]))
            # policy update
            policy = policy_update(policy, Q, state, epsilon)
            state = next_state
            action = next_action

            total_rewards[i] += reward
            episodes_length[i] += 1
    draw_metrics(total_rewards, episodes_length)
    return Q, policy

def n_step_Sarsa(env, num_episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1, num_steps=1):
    # for animation
    total_rewards = np.zeros(num_episodes)
    episodes_length = np.zeros(num_episodes)
    # algorithm
    Q = np.zeros((env.num_states, len(env.action_space)))
    policy = np.random.rand(env.num_states, len(env.action_space))
    policy /= policy.sum(axis=1)[:, np.newaxis]  # make the sum of elements in each row to be 1
    for i in range(num_episodes):
        # state = np.random.choice(env.num_states)
        state = [env.pos2state(env.reset())]
        action = [np.random.choice(len(env.action_space), p=policy[state[0], :])]
        done = False
        while not done:
            TD_target = 0
            for j in range(num_steps):
                next_pos, reward, done, _ = env.step(env.action_space[action[-1]])
                next_state = env.pos2state(next_pos)
                next_action = np.random.choice(len(env.action_space), p=policy[next_state, :])
                state.append(next_state)
                action.append(next_action)
                TD_target += gamma ** j * reward
            # value update
            TD_target += gamma ** num_steps * Q[next_state, next_action]
            Q[state[0], action[0]] -= alpha * (Q[state[0], action[0]] - TD_target)
            # policy update
            policy = policy_update(policy, Q, state[0], epsilon)
            state = [state[1]]
            action = [action[1]]

            total_rewards[i] += reward
            episodes_length[i] += 1
        print("Episode: ", i, "Total reward: ", total_rewards[i], "Episode length: ", episodes_length[i])
    draw_metrics(total_rewards, episodes_length)
    return Q, policy

# Example usage:
if __name__ == "__main__":             
    env = GridWorld()
    state = env.reset()
    
    env.render()
    # Q, policy = Sarsa(env, num_episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1)
    Q, policy = n_step_Sarsa(env, num_episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1, num_steps=2)
    env.add_policy(policy)
    # Render the environment
    env.render(animation_interval=15)