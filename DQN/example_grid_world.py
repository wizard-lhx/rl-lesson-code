from grid_world import GridWorld
import random
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)
    
class Qnet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)
    
class DQN:
    ''' DQN算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim,
                          self.action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        self.criterion = nn.MSELoss()  # 使用均方误差损失函数
        # 使用Adam优化器
        self.optimizer = optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):  # 均匀分布的策略采取动作
        action = np.random.randint(self.action_dim)
        return action
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
            -1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
                                                                )  # TD误差目标
        dqn_loss = self.criterion(q_values, q_targets)  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1

# Example usage:
if __name__ == "__main__":
    # 超参数
    lr = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.9
    epsilon = 1
    target_update = 10
    buffer_size = 10000
    minimal_size = 100
    batch_size = 32
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    # 设置环境，且固定随机种子使每次实验重复
    env = GridWorld()
    state = env.reset()
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = 2
    action_dim = 5
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                target_update, device)
    
    policy = np.zeros((env.num_states,len(env.action_space)))
    env.render()
    
    # 因为每个 episode 采取的动作是随机的，所以这里计算的 behaviour policy 的 return 没有意义，可以使用目标策略的 return 作为评估指标
    return_list = []
    # 总共 500 episode，每 50 个 episode 显示当前最优策略
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                # 每个 episode 固定为 1000 步
                for j in range(1000):
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(env.action_space[action])
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    # 当 buffer 数据的数量超过一定值后,才进行 Q 网络训练
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                # 更新进度条
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
            # 每 50 个 episode 显示一次当前最优策略
            for k in range(env.num_states):
                policy[k] = np.eye(len(env.action_space))[agent.q_net(torch.tensor(env.state2pos(k), dtype=torch.float).to(device)).argmax().item()]
            env.add_policy(policy)
            env.render()
    
    env.add_policy(policy)
    # Render the environment
    env.render(animation_interval=15)