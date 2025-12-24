import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import gymnasium as gym

# 确保能导入自定义环境
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.snake_env import SnakeEnv

def train_ppo(args):
    # 创建日志目录
    log_dir = os.path.join('logs', 'ppo_snake')
    os.makedirs(log_dir, exist_ok=True)
    
    # 配置日志
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    
    # 环境配置
    env_kwargs = {
        'grid_size': args.grid_size,
        'mode': args.mode
    }
    
    # 创建并行环境进行训练
    vec_env = make_vec_env(
        'SnakeEnv-v1',
        n_envs=args.n_envs,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv
    )
    
    # 创建评估环境
    eval_env = Monitor(gym.make('SnakeEnv-v1', **env_kwargs))
    
    # 回调函数
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=os.path.join('models', 'ppo_snake'),
        name_prefix='ppo_snake',
        save_replay_buffer=True
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join('models', 'ppo_snake'),
        log_path=os.path.join(log_dir, 'eval'),
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False
    )
    
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,
        min_evals=5,
        verbose=1
    )
    
    # 策略网络配置
    if args.mode == 'feature':
        policy_kwargs = {
            'net_arch': {
                'pi': [64, 64],
                'vf': [64, 64]
            }
        }
    elif args.mode == 'pixel':
        policy_kwargs = {
            'net_arch': {
                'pi': [128, 64],
                'vf': [128, 64]
            }
        }
    elif args.mode == 'grid':
        policy_kwargs = {
            'net_arch': {
                'pi': [128, 64],
                'vf': [128, 64]
            }
        }
    
    # 创建PPO模型
    model = PPO(
        'MlpPolicy' if args.mode == 'feature' else 'CnnPolicy',
        vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    
    # 设置日志器
    model.set_logger(new_logger)
    
    # 开始训练
    print(f"开始训练PPO模型，模式: {args.mode}, 网格大小: {args.grid_size}")
    print(f"日志保存到: {log_dir}")
    print(f"模型保存到: {os.path.join('models', 'ppo_snake')}")
    
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback, stop_callback],
        progress_bar=True
    )
    
    # 保存最终模型
    model.save(os.path.join('models', 'ppo_snake', 'ppo_snake_final'))
    print("训练完成！")
    
    # 关闭环境
    vec_env.close()
    eval_env.close()
    
    return model

def main():
    parser = argparse.ArgumentParser(description='训练贪吃蛇PPO智能体')
    
    # 环境参数
    parser.add_argument('--grid_size', type=int, default=10, help='游戏网格大小')
    parser.add_argument('--mode', type=str, default='feature', choices=['feature', 'pixel', 'grid'], help='观察模式')
    
    # 训练参数
    parser.add_argument('--n_envs', type=int, default=4, help='并行环境数量')
    parser.add_argument('--total_timesteps', type=int, default=1000000, help='总训练步数')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='学习率')
    parser.add_argument('--n_steps', type=int, default=2048, help='每批次收集的步数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--n_epochs', type=int, default=10, help='每批次训练的轮数')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip_range', type=float, default=0.2, help='裁剪范围')
    parser.add_argument('--ent_coef', type=float, default=0.01, help='熵系数')
    parser.add_argument('--vf_coef', type=float, default=0.5, help='值函数系数')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='梯度裁剪')
    
    # 回调参数
    parser.add_argument('--save_freq', type=int, default=10000, help='模型保存频率')
    parser.add_argument('--eval_freq', type=int, default=5000, help='评估频率')
    
    args = parser.parse_args()
    
    # 创建模型保存目录
    os.makedirs(os.path.join('models', 'ppo_snake'), exist_ok=True)
    
    # 开始训练
    train_ppo(args)

if __name__ == '__main__':
    main()
