from grid_world import GridWorld
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

class DQN():
    def __init__(self, env, gamma=0.9, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon

        self.main_network = self.build_model()
        self.target_network = self.build_model()
        for param in self.target_network.parameters():
            param.requires_grad = False
        self.optimizer = optim.SGD(self.main_network.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        self.behavior_policy = np.ones((self.env.num_states, len(self.env.action_space))) / len(self.env.action_space)
        self.target_policy = np.ones((self.env.num_states, len(self.env.action_space))) / len(self.env.action_space)

        self.replay_buffer = []

    def build_model(self):
        model = nn.Sequential(
            # input is state and action, output is Q-value
            nn.Linear(2, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        return model
    
    def get_policy(self):
        return self.target_policy
    
    def generate_replay_buffer(self, num_episodes=1, num_steps = 1000):
        self.replay_buffer = []
        for episode in range(num_episodes):
            state = self.env.pos2state(self.env.reset())
            for step in range(num_steps):
                action = np.random.choice(len(self.env.action_space),p = self.behavior_policy[state])
                next_pos, reward, done,_ = self.env.step(self.env.action_space[action])
                next_state = self.env.pos2state(next_pos)
                self.replay_buffer.append((state, action, reward, next_state))
                state = next_state
                # if done:
                #     break
        
    def train(self):
        q_value, q_target = self.get_batch(batch_size=100)
        loss = self.criterion(q_value, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def get_batch(self, batch_size=32):
        # choose the batch_size {s,a,r,s'}
        random.shuffle(self.replay_buffer)
        batch = random.sample(self.replay_buffer, batch_size)

        # (b,4)->(b,)
        state, action, reward, next_state = zip(*batch)
        # (b,)->(b,2)
        state_action_pair = torch.FloatTensor(list(zip(state,action)))
        q_value = self.main_network(state_action_pair)
        
        # (b,)->(b,1)->(b,1,1)->(b,5,1)
        state_expand = torch.FloatTensor(next_state).unsqueeze(1).unsqueeze(1).repeat(1,5,1)
        # (5,)->(5,1)->(1,5,1)->(b,5,1)
        action_expand = torch.arange(len(self.env.action_space)).unsqueeze(1).unsqueeze(0).repeat(batch_size,1,1)
        # (b,5,1)->(b,5,2)
        next_state_action_pairs = torch.cat([state_expand,action_expand],dim = 2)
        # (b,5,2)->(b,5,1)
        next_q_value = self.target_network(next_state_action_pairs)
        # (b,5,1)->(b,1)
        max_q_value,max_action_idx = torch.max(next_q_value,dim = 1)
        q_target = torch.FloatTensor(reward).unsqueeze(1) + self.gamma * max_q_value

        return q_value, q_target

    def update_policy(self, state, action_star):
        # greedy policy
        self.target_policy[state][action_star] = 1
        for i in range(len(self.env.action_space)):
            if i != action_star:
                self.target_policy[state][i] = 0

    def run(self, num_episodes=100):
        rewards = 0
        self.generate_replay_buffer(num_episodes=1, num_steps = 1000)
        V = np.zeros(self.env.num_states)

        self.main_network.train()
        for episode in range(num_episodes):
            self.train()
            if (episode+1) % 5 == 0:
                self.target_network.load_state_dict(self.main_network.state_dict())
        with torch.no_grad():
            state = [self.env.pos2state(self.env.reset())]
            done = False
            # while not done:
            for i in range(self.env.num_states):
                state_action_pairs = torch.cat([torch.tensor(state,dtype=torch.float).unsqueeze(1).repeat(5,1),
                                                torch.arange(len(self.env.action_space),dtype=torch.float).unsqueeze(1)],dim = 1)
                max_q_value, action_star = torch.max(self.target_network(state_action_pairs),dim=0)
                self.update_policy(state[0],action_star[0])
                next_pos, reward, done, _ = self.env.step(self.env.action_space[action_star])
                # state = [self.env.pos2state(next_pos)]
                state = [i]
                rewards += reward
                V[i] = max_q_value[0]
            return V

# Example usage:
if __name__ == "__main__":             
    env = GridWorld()
    state = env.reset()
    
    env.render()
    dqn = DQN(env)
    V = dqn.run(num_episodes=400)
    env.add_policy(dqn.get_policy())
    env.add_state_values(V)
    # Render the environment
    env.render(animation_interval=15)