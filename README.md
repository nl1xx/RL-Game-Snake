# 深度强化学习贪吃蛇项目 (DRL Snake Game)

一个完整的、端到端的深度强化学习应用系统，基于经典贪吃蛇游戏实现。本项目展示了如何从零开始构建一个能够自主学习最优策略的智能体。

## 项目概述

本项目旨在开发一个完整的深度强化学习应用系统，以经典的贪吃蛇游戏为载体，实现一个能够自主学习最优策略的智能体。项目将展示DRL如何在动态环境中实现自适应决策，克服传统规则方法的局限性。

### 核心特性

- **完整的RL生态系统**：从环境搭建、智能体训练到模型部署的全流程实现
- **多算法支持**：集成PPO、DQN、A2C、DDPG、SAC、TD3等多种深度强化学习算法
- **多模式观察空间**：支持特征向量、像素和网格三种观察模式
- **多蛇对抗模式**：支持多条AI控制的蛇在同一环境中竞争，实现多智能体对抗
- **高性能训练**：使用并行环境和算法优化加速训练
- **可视化界面**：提供交互式游戏界面，支持人工控制和AI演示，可显示多条蛇的实时状态
- **全面的评估体系**：包括训练分析、模型评估和泛化能力测试

## 技术栈

### 仿真平台与游戏环境
- **Gymnasium**：标准强化学习环境接口
- **PyGame**：游戏渲染与用户交互

### 深度强化学习
- **算法**：PPO (Proximal Policy Optimization)、DQN (Deep Q-Network)、A2C (Advantage Actor-Critic)、DDPG (Deep Deterministic Policy Gradient)、SAC (Soft Actor-Critic)、TD3 (Twin Delayed DDPG)
- **库**：Stable-Baselines3 (SB3)、rl-baselines3-zoo
- **框架**：PyTorch

### 数据处理与可视化
- **数据处理**：NumPy、Pandas
- **可视化**：Matplotlib、Seaborn

## 项目结构

```
DRL/
├── environment/          # 游戏环境模块
│   └── snake_env.py      # 贪吃蛇环境实现
├── training/             # 训练模块
│   └── train.py          # 多算法训练脚本（支持PPO、DQN、A2C、DDPG、SAC、TD3）
├── controller/           # 控制器模块
│   └── snake_ai.py       # AI控制器实现
├── analysis/             # 数据分析模块
│   └── data_analyzer.py  # 训练日志分析
├── evaluation/           # 评估模块
│   └── evaluate_model.py # 模型评估脚本
├── ui/                   # 用户界面模块
│   └── snake_ui.py       # 游戏界面实现
├── models/               # 模型保存目录
├── logs/                 # 日志保存目录
├── main.py               # 主程序入口
├── requirements.txt      # 项目依赖
└── README.md             # 项目说明
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练智能体

```bash
# 基本训练（默认使用PPO算法）
python main.py train

# 指定算法训练
python main.py train --algorithm ppo  # 使用PPO算法
python main.py train --algorithm dqn  # 使用DQN算法
python main.py train --algorithm a2c  # 使用A2C算法
python main.py train --algorithm ddpg # 使用DDPG算法
python main.py train --algorithm sac  # 使用SAC算法
python main.py train --algorithm td3  # 使用TD3算法

# 多蛇对抗模式训练（2条蛇）
python main.py train --algorithm ppo --num_snakes 2

# 自定义参数训练
python main.py train --algorithm ppo --grid_size 10 --mode feature --total_timesteps 1000000
```

### 3. 启动游戏界面

```bash
# 人工控制模式
python main.py ui

# AI模式（需要先训练模型）
python main.py ui --model_path models/ppo_snake/ppo_snake_final

# 多蛇AI演示模式（2条蛇）
python main.py ui --model_path models/ppo_snake/ppo_snake_final --num_snakes 2 --multi_agent
```

### 4. 评估模型

```bash
# 单蛇模式评估
python main.py evaluate --model_path models/ppo_snake/ppo_snake_final --n_episodes 20

# 多蛇对抗模式评估（2条蛇）
python main.py evaluate --model_path models/ppo_snake/ppo_snake_final --n_episodes 20 --num_snakes 2 --multi_agent
```

### 5. 分析训练结果

```bash
python main.py analyze
```

## 详细使用说明

### 训练参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| --algorithm | 强化学习算法 (ppo/dqn/a2c/ddpg/sac/td3) | ppo |
| --grid_size | 游戏网格大小 | 10 |
| --mode | 观察模式 (feature/pixel/grid) | feature |
| --n_envs | 并行环境数量 | 4 |
| --total_timesteps | 总训练步数 | 1000000 |
| --save_freq | 模型保存频率 | 10000 |
| --eval_freq | 评估频率 | 5000 |
| --num_snakes | 蛇的数量 | 1 |
| --multi_agent | 是否启用多智能体模式 | False |

### 游戏控制

- **方向键**：控制蛇的移动
- **R**：重置游戏
- **A**：切换AI模式
- **ESC**：退出游戏

## 观察模式说明

### 1. Feature模式（特征向量）

- 形状：(12,)
- 内容：
  - 蛇头相对食物位置 (4个特征)
  - 蛇身体方向 (4个特征)
  - 障碍物检测 (4个特征)
- 优点：计算效率高，训练速度快

### 2. Pixel模式（像素观察）

- 形状：(屏幕尺寸, 屏幕尺寸, 3)
- 内容：RGB图像
- 优点：无需特征工程，更接近真实场景

### 3. Grid模式（网格编码）

- 形状：(网格大小, 网格大小)
- 内容：0=空, 1=蛇身, 2=蛇头, 3=食物
- 优点：直观，便于理解

## 模型训练与评估

### 训练流程

1. **环境初始化**：创建并行贪吃蛇环境，支持单蛇或多蛇模式
2. **算法选择**：根据指定的算法(PPO/DQN/A2C等)配置相应的网络结构和超参数
3. **训练过程**：智能体与环境交互，不断优化策略
4. **模型保存**：定期保存模型和训练日志
5. **评估验证**：定期评估模型性能，支持早停机制

### 评估指标

- **得分**：每回合吃到的食物数量
- **回合长度**：每回合持续的步数
- **累计奖励**：每回合获得的总奖励

## 高级功能

### 泛化能力测试

```bash
python main.py evaluate --model_path models/ppo_snake/ppo_snake_final --n_episodes 20 --generalization_test
```

该功能将测试模型在不同网格大小下的表现，评估其泛化能力。

### 超参数优化

可以使用rl-baselines3-zoo工具进行超参数优化：

```bash
python -m rl_zoo3.train --algo ppo --env SnakeEnv-v1 --eval-episodes 5 --eval-freq 10000
```

## 项目应用

本项目可用于：

1. **教学演示**：展示DRL算法的基本原理和应用
2. **算法研究**：作为实验平台测试新的RL算法和改进
3. **游戏AI开发**：为游戏开发智能对手或助手
4. **决策系统开发**：展示如何将RL应用于实际决策问题

## 开发说明

### 环境注册

自定义环境已在`environment/snake_env.py`中注册为`SnakeEnv-v1`，可以直接使用Gymnasium的接口访问。

### 添加新的观察模式

1. 在`SnakeEnv`类的`__init__`方法中扩展观察空间定义
2. 在`_get_observation`方法中实现新的观察转换逻辑
3. 更新训练和评估脚本以支持新的模式

### 自定义奖励函数

修改`SnakeEnv`类的`step`方法中的奖励计算逻辑，尝试不同的奖励塑形方案。

## 性能优化

1. **使用GPU加速**：确保安装了支持CUDA的PyTorch版本
2. **增加并行环境数量**：通过`--n_envs`参数提高训练效率
3. **调整网络结构**：根据观察模式调整策略网络结构
4. **优化超参数**：使用rl-baselines3-zoo进行超参数搜索

## 未来改进方向

- [x] 支持更多RL算法（DQN、A2C等）
- [x] 实现多智能体对抗
- [ ] 添加更复杂的游戏机制（障碍物、多食物等）
- [ ] 开发Web界面便于展示
- [ ] 支持模型导出和部署

## 许可证

MIT License

## 参考文献

1. [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
2. [Gymnasium Documentation](https://gymnasium.farama.org/)
3. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
4. [PyTorch Documentation](https://pytorch.org/docs/stable/)

## 联系方式

如有问题或建议，欢迎提交Issue或Pull Request。
