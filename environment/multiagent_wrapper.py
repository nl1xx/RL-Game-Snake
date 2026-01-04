import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper

class MultiAgentWrapper(gym.Wrapper):
    """
    多智能体环境包装器，将多智能体环境转换为单智能体环境
    使用独立学习（Independent Learning）方法
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        
        # 获取原始环境参数
        self.num_snakes = env.num_snakes
        self.multi_agent = env.multi_agent
        
        if not self.multi_agent:
            raise ValueError("MultiAgentWrapper 只用于多智能体环境")
        
        # 获取单个智能体的观察和动作空间
        if isinstance(env.observation_space, tuple):
            self.observation_space = env.observation_space[0]
        else:
            self.observation_space = env.observation_space
        
        if isinstance(env.action_space, tuple):
            self.action_space = env.action_space[0]
        else:
            self.action_space = env.action_space
        
        # 当前控制的智能体索引（轮换使用）
        self.current_agent = 0
        
        # 存储所有智能体的观察和奖励
        self.all_observations = None
        self.all_rewards = None
        self.all_dones = None
        self.all_infos = None
    
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        
        if isinstance(obs, tuple):
            self.all_observations = list(obs)
        else:
            self.all_observations = [obs]
        
        self.current_agent = 0
        self.all_rewards = [0.0] * self.num_snakes
        self.all_dones = [False] * self.num_snakes
        self.all_infos = info
        
        return self.all_observations[self.current_agent], info
    
    def step(self, action):
        # 构建所有智能体的动作列表
        actions = []
        for i in range(self.num_snakes):
            if i == self.current_agent:
                actions.append(action)
            else:
                # 其他智能体使用随机动作或保持当前方向
                actions.append(np.random.randint(0, 4))
        
        # 执行环境步骤
        obs, rewards, dones, truncated, info = self.env.step(tuple(actions))
        
        # 存储所有智能体的信息
        if isinstance(obs, tuple):
            self.all_observations = list(obs)
        else:
            self.all_observations = [obs]
        
        if isinstance(rewards, tuple):
            self.all_rewards = list(rewards)
        else:
            self.all_rewards = [rewards]
        
        if isinstance(dones, tuple):
            self.all_dones = list(dones)
        else:
            self.all_dones = [dones]
        
        self.all_infos = info
        
        # 切换到下一个存活的智能体
        self._next_agent()
        
        # 返回当前智能体的信息
        current_reward = self.all_rewards[self.current_agent]
        current_done = self.all_dones[self.current_agent]
        all_done = all(self.all_dones) or truncated
        
        return (
            self.all_observations[self.current_agent],
            current_reward,
            current_done or all_done,
            truncated,
            info
        )
    
    def _next_agent(self):
        """切换到下一个存活的智能体"""
        # 找到下一个存活的智能体
        for _ in range(self.num_snakes):
            self.current_agent = (self.current_agent + 1) % self.num_snakes
            if self.current_agent < len(self.all_dones) and not self.all_dones[self.current_agent]:
                return
        
        # 如果所有智能体都死亡，重置为第一个
        self.current_agent = 0


class ParallelMultiAgentWrapper(gym.Wrapper):
    """
    并行多智能体包装器，为每个智能体创建独立的环境副本
    适用于并行训练
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        
        self.num_snakes = env.num_snakes
        self.multi_agent = env.multi_agent
        
        if not self.multi_agent:
            raise ValueError("ParallelMultiAgentWrapper 只用于多智能体环境")
        
        # 获取单个智能体的观察和动作空间
        if isinstance(env.observation_space, tuple):
            self.observation_space = env.observation_space[0]
        else:
            self.observation_space = env.observation_space
        
        if isinstance(env.action_space, tuple):
            self.action_space = env.action_space[0]
        else:
            self.action_space = env.action_space
        
        # 当前智能体索引
        self.current_agent = 0
        
        # 保存对原始环境的引用，用于渲染
        self.unwrapped_env = env
    
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        
        if isinstance(obs, tuple):
            return obs[self.current_agent], info
        else:
            return obs, info
    
    def step(self, action):
        # 为所有智能体构建动作
        actions = [action] * self.num_snakes
        
        # 执行环境步骤
        obs, rewards, dones, truncated, info = self.env.step(tuple(actions))
        
        # 提取当前智能体的信息
        if isinstance(obs, tuple):
            current_obs = obs[self.current_agent]
        else:
            current_obs = obs
        
        if isinstance(rewards, tuple):
            current_reward = rewards[self.current_agent]
        else:
            current_reward = rewards
        
        if isinstance(dones, tuple):
            current_done = dones[self.current_agent]
        else:
            current_done = dones
        
        all_done = all(dones) if isinstance(dones, tuple) else dones
        
        return current_obs, current_reward, current_done or all_done, truncated, info


def make_multiagent_env(env_id, num_snakes=2, grid_size=10, mode='feature', **kwargs):
    """
    创建多智能体环境并应用包装器
    
    Args:
        env_id: 环境ID
        num_snakes: 蛇的数量
        grid_size: 网格大小
        mode: 观察模式
        **kwargs: 其他环境参数
    
    Returns:
        包装后的多智能体环境
    """
    # 直接导入并创建环境实例，避免使用 gym.make() 时的 PassiveEnvChecker 检查
    from environment.snake_env import SnakeEnv
    env = SnakeEnv(
        render_mode=None,
        grid_size=grid_size,
        mode=mode,
        num_snakes=num_snakes,
        multi_agent=True,
        **kwargs
    )
    return ParallelMultiAgentWrapper(env)


class DiscreteToContinuousActionWrapper(VecEnvWrapper):
    """
    将离散动作空间转换为连续动作空间的包装器
    用于支持DDPG、SAC、TD3等需要连续动作空间的算法
    支持包装VecEnv
    """
    
    def __init__(self, venv):
        super().__init__(venv)
        self.venv = venv
        
        # 获取原始离散动作空间
        if isinstance(venv.action_space, spaces.Discrete):
            self.n_actions = venv.action_space.n
        elif isinstance(venv.action_space, spaces.Box):
            # 如果已经是连续动作空间，直接使用
            self.n_actions = venv.action_space.shape[0]
            self.already_continuous = True
        else:
            raise ValueError("DiscreteToContinuousActionWrapper 只用于离散动作空间")
        
        # 设置连续动作空间（4个动作，每个动作对应一个连续值）
        if not getattr(self, 'already_continuous', False):
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_actions,), dtype=np.float32)
        
        # 保持观察空间不变
        self.observation_space = venv.observation_space
    
    def step_async(self, actions):
        # 如果已经是连续动作空间，直接传递
        if getattr(self, 'already_continuous', False):
            return self.venv.step_async(actions)
        
        # 将连续动作转换为离散动作（选择最大值的索引）
        discrete_actions = np.argmax(actions, axis=1)
        
        # 执行离散动作
        self.venv.step_async(discrete_actions)
    
    def step_wait(self):
        return self.venv.step_wait()
    
    def reset(self, seed=None, options=None):
        return self.venv.reset()