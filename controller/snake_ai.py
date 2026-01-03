import os
import sys
import numpy as np
from stable_baselines3 import PPO

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
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """加载训练好的PPO模型"""
        try:
            self.model = PPO.load(model_path)
            print(f"成功加载模型: {model_path}")
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
