import os
import sys
import numpy as np
from stable_baselines3 import PPO, DQN, A2C, DDPG, SAC, TD3

# 确保能导入自定义环境
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SnakeAI:
    def __init__(self, model_path, mode='feature'):
        """
        贪吃蛇AI控制器
        
        Args:
            model_path: 训练好的模型路径
            mode: 观察模式 ('feature', 'pixel', 'grid')
        """
        self.mode = mode
        self.model = None
        self.algorithm = None
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """加载训练好的模型（支持多种算法）"""
        try:
            # 尝试从路径中检测算法类型
            path_lower = model_path.lower()
            if 'dqn' in path_lower:
                self.model = DQN.load(model_path)
                self.algorithm = 'dqn'
            elif 'ddpg' in path_lower:
                self.model = DDPG.load(model_path)
                self.algorithm = 'ddpg'
            elif 'sac' in path_lower:
                self.model = SAC.load(model_path)
                self.algorithm = 'sac'
            elif 'td3' in path_lower:
                self.model = TD3.load(model_path)
                self.algorithm = 'td3'
            elif 'a2c' in path_lower:
                self.model = A2C.load(model_path)
                self.algorithm = 'a2c'
            else:
                # 默认使用PPO
                self.model = PPO.load(model_path)
                self.algorithm = 'ppo'
            
            # 保存模型的观察空间信息
            self.model_obs_shape = self.model.observation_space.shape
            
            print(f"成功加载{self.algorithm.upper()}模型: {model_path}")
            print(f"模型观察空间形状: {self.model_obs_shape}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            raise
    
    def predict(self, observation):
        """
        根据当前观察预测动作
        
        Args:
            observation: 当前环境观察
            
        Returns:
            action: 预测的动作 (0=上, 1=右, 2=下, 3=左)
        """
        if self.model is None:
            raise ValueError("模型未加载")
        
        obs = observation
        
        # 处理多智能体观察空间
        # 在多智能体模式下，观察空间包含所有蛇的信息（12 + 4*(num_snakes-1)）
        # 但训练好的模型可能期望单智能体观察空间（12维）或多智能体观察空间（16维）
        if isinstance(obs, np.ndarray):
            if obs.ndim == 1:
                # 单个观察向量
                if len(obs) != self.model_obs_shape[0]:
                    # 观察维度不匹配，需要调整
                    if len(obs) > self.model_obs_shape[0]:
                        # 当前观察比模型期望的更大，只提取前N维
                        obs = obs[:self.model_obs_shape[0]]
                    else:
                        # 当前观察比模型期望的更小，需要填充
                        obs = np.pad(obs, (0, self.model_obs_shape[0] - len(obs)), mode='constant')
            elif obs.ndim == 2:
                # 批次观察
                if obs.shape[1] != self.model_obs_shape[0]:
                    # 观察维度不匹配，需要调整
                    if obs.shape[1] > self.model_obs_shape[0]:
                        # 当前观察比模型期望的更大，只提取前N维
                        obs = obs[:, :self.model_obs_shape[0]]
                    else:
                        # 当前观察比模型期望的更小，需要填充
                        padding = ((0, 0), (0, self.model_obs_shape[0] - obs.shape[1]))
                        obs = np.pad(obs, padding, mode='constant')
        
        # 转换为模型期望的格式
        if self.mode == 'feature':
            # 特征向量已经是正确的形状
            pass
        elif self.mode == 'pixel':
            # 像素观察需要添加批次维度
            obs = np.expand_dims(obs, axis=0)
        elif self.mode == 'grid':
            # 网格观察需要添加批次维度
            obs = np.expand_dims(obs, axis=0)
        
        # 使用模型预测动作
        action, _ = self.model.predict(obs, deterministic=True)
        
        # 对于DDPG、SAC、TD3等连续动作算法，需要将连续动作转换为离散动作
        if self.algorithm in ['ddpg', 'sac', 'td3']:
            # 这些算法输出连续动作，选择最大值的索引作为离散动作
            if isinstance(action, np.ndarray) and action.ndim > 1:
                discrete_action = int(np.argmax(action[0]))
            else:
                discrete_action = int(np.argmax(action))
            return discrete_action
        else:
            # 离散动作算法（PPO、DQN、A2C）直接返回动作
            return int(action)
    
    def get_action_name(self, action):
        """获取动作名称"""
        action_names = ['上', '右', '下', '左']
        return action_names[action]
    
    def close(self):
        """关闭模型资源"""
        if hasattr(self.model, 'save'):
            del self.model
            self.model = None
            print("模型已关闭")

# 示例用法
if __name__ == '__main__':
    import gymnasium as gym
    from environment.snake_env import SnakeEnv
    
    # 加载模型
    ai = SnakeAI(model_path=os.path.join('models', 'ppo_snake', 'ppo_snake_final'))
    
    # 创建环境
    env = gym.make('SnakeEnv-v1', render_mode='human', grid_size=10, mode='feature')
    
    # 测试AI
    observation, info = env.reset()
    done = False
    
    while not done:
        action = ai.predict(observation)
        print(f"动作: {ai.get_action_name(action)}")
        observation, reward, done, truncated, info = env.step(action)
        
        if done:
            print(f"游戏结束，得分: {info['score']}")
            break
    
    env.close()
    ai.close()
