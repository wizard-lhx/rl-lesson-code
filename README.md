# 强化学习教程代码
## 1 环境
python 版本：3.11.9<br>
算法验证环境：在[强化学习的数学原理书本及环境代码](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)中设置 5 x 5 网格地图。
## 2 运行方法
```
代码
├── DQN
├── MC based
├── README.md
├── TD
├── demo（只包括对源代码进行少量修改的运行环境以及对其测试）
└── value iteration（值和策略迭代算法）
```
打开对应文件夹的 example_grid_world.py 文件运行即可。
## 3 已知问题
1. MC-epsilon greedy 算法无法收敛
2. multi-step Sarsa 算法无法收敛

## 4 参考链接
[强化学习的数学原理书本及环境代码](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)<br>
[视频教程](https://www.bilibili.com/video/BV1sd4y167NS)<br>
[Hands-on Reinforcement Learning](https://hrl.boyuai.com/chapter/2/dqn%E6%94%B9%E8%BF%9B%E7%AE%97%E6%B3%95/)<br>
[参考代码 1](https://github.com/10-OASIS-01/minrl)<br>
[参考代码 2](https://github.com/ziwenhahaha/Code-of-RL-Beginning)<br>