import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN, A2C, DDPG, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import gymnasium as gym

# 确保能导入自定义环境
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.snake_env import SnakeEnv
from environment.multiagent_wrapper import make_multiagent_env, DiscreteToContinuousActionWrapper

def get_algorithm(algorithm_name):
    """根据算法名称返回对应的算法类"""
    algorithms = {
        'ppo': PPO,
        'dqn': DQN,
        'a2c': A2C,
        'ddpg': DDPG,
        'sac': SAC,
        'td3': TD3
    }
    return algorithms.get(algorithm_name.lower(), PPO)

def get_policy(mode, algorithm_name):
    """根据观察模式和算法选择合适的策略"""
    if mode == 'feature':
        return 'MlpPolicy'
    elif mode == 'pixel':
        if algorithm_name.lower() in ['ppo', 'a2c', 'dqn']:
            return 'CnnPolicy'
        else:  # DDPG, SAC, TD3
            return 'CnnPolicy'
    elif mode == 'grid':
        return 'MlpPolicy'
    return 'MlpPolicy'

def get_policy_kwargs(mode, algorithm_name):
    """根据观察模式和算法设置策略网络参数"""
    continuous_algorithms = ['ddpg', 'sac', 'td3']
    
    if mode == 'feature':
        if algorithm_name.lower() == 'dqn':
            return {
                'net_arch': [64, 64]
            }
        elif algorithm_name.lower() in continuous_algorithms:
            return {
                'net_arch': {
                    'pi': [64, 64],
                    'qf': [64, 64]
                }
            }
        else:
            return {
                'net_arch': {
                    'pi': [64, 64],
                    'vf': [64, 64]
                }
            }
    elif mode == 'pixel':
        if algorithm_name.lower() == 'dqn':
            return {
                'net_arch': [128, 64]
            }
        elif algorithm_name.lower() in continuous_algorithms:
            return {
                'net_arch': {
                    'pi': [128, 64],
                    'qf': [128, 64]
                }
            }
        else:
            return {
                'net_arch': {
                    'pi': [128, 64],
                    'vf': [128, 64]
                }
            }
    elif mode == 'grid':
        if algorithm_name.lower() == 'dqn':
            return {
                'net_arch': [128, 64]
            }
        elif algorithm_name.lower() in continuous_algorithms:
            return {
                'net_arch': {
                    'pi': [128, 64],
                    'qf': [128, 64]
                }
            }
        else:
            return {
                'net_arch': {
                    'pi': [128, 64],
                    'vf': [128, 64]
                }
            }
    return {}

def train_agent(args):
    """训练智能体的通用函数"""
    # 获取算法类
    algorithm = get_algorithm(args.algorithm)
    
    # 创建日志目录
    log_dir = os.path.join('logs', f'{args.algorithm}_snake')
    os.makedirs(log_dir, exist_ok=True)
    
    # 配置日志
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    
    # 环境配置
    env_kwargs = {
        'grid_size': args.grid_size,
        'mode': args.mode,
        'num_snakes': args.num_snakes,
        'multi_agent': args.multi_agent
    }
    
    # 创建并行环境进行训练
    if args.multi_agent:
        # 多智能体模式：使用包装器
        def make_env(rank):
            def _init():
                env = make_multiagent_env(
                    'SnakeEnv-v1',
                    num_snakes=args.num_snakes,
                    grid_size=args.grid_size,
                    mode=args.mode
                )
                return env
            return _init
        
        vec_env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
    else:
        # 单智能体模式：直接创建环境
        def make_env(rank):
            def _init():
                env = SnakeEnv(
                    grid_size=args.grid_size,
                    mode=args.mode,
                    num_snakes=args.num_snakes,
                    multi_agent=args.multi_agent
                )
                return env
            return _init
        
        vec_env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
    
    # 如果是像素模式，对多帧进行堆叠（适用于DQN等需要时序信息的算法）
    if args.mode == 'pixel' and args.algorithm.lower() in ['dqn', 'a2c']:
        vec_env = VecFrameStack(vec_env, n_stack=4)
    
    # 如果是DDPG、SAC、TD3算法，需要将离散动作空间转换为连续动作空间
    if args.algorithm.lower() in ['ddpg', 'sac', 'td3']:
        from stable_baselines3.common.vec_env import VecMonitor
        vec_env = VecMonitor(vec_env)
        vec_env = DiscreteToContinuousActionWrapper(vec_env)
    
    # 创建评估环境
    if args.multi_agent:
        eval_env = make_multiagent_env(
            'SnakeEnv-v1',
            num_snakes=args.num_snakes,
            grid_size=args.grid_size,
            mode=args.mode
        )
    else:
        eval_env = Monitor(SnakeEnv(
            grid_size=args.grid_size,
            mode=args.mode,
            num_snakes=args.num_snakes,
            multi_agent=args.multi_agent
        ))
    
    # 回调函数
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=os.path.join('models', f'{args.algorithm}_snake'),
        name_prefix=f'{args.algorithm}_snake',
        save_replay_buffer=True
    )
    
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,
        min_evals=5,
        verbose=1
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join('models', f'{args.algorithm}_snake'),
        log_path=os.path.join(log_dir, 'eval'),
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
        callback_after_eval=stop_callback
    )
    
    # 策略选择
    policy = get_policy(args.mode, args.algorithm)
    policy_kwargs = get_policy_kwargs(args.mode, args.algorithm)
    
    # 算法特定参数配置
    algorithm_kwargs = {
        'learning_rate': args.learning_rate,
        'gamma': args.gamma,
        'verbose': 1
    }
    
    # 根据不同算法添加特定参数
    if args.algorithm.lower() == 'ppo':
        algorithm_kwargs.update({
            'n_steps': args.n_steps,
            'batch_size': args.batch_size,
            'n_epochs': args.n_epochs,
            'gae_lambda': args.gae_lambda,
            'clip_range': args.clip_range,
            'ent_coef': args.ent_coef,
            'vf_coef': args.vf_coef,
            'max_grad_norm': args.max_grad_norm
        })
    elif args.algorithm.lower() == 'dqn':
        algorithm_kwargs.update({
            'buffer_size': 100000,
            'learning_starts': 5000,
            'batch_size': args.batch_size,
            'target_update_interval': 1000,
            'train_freq': 4,
            'exploration_fraction': 0.1,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.05
        })
    elif args.algorithm.lower() in ['ddpg', 'sac', 'td3']:
        algorithm_kwargs.update({
            'buffer_size': 100000,
            'learning_starts': 10000,
            'batch_size': args.batch_size
        })
    elif args.algorithm.lower() == 'a2c':
        algorithm_kwargs.update({
            'n_steps': args.n_steps,
            'ent_coef': args.ent_coef
        })
    
    # 添加策略参数
    algorithm_kwargs['policy_kwargs'] = policy_kwargs
    
    # 创建模型
    model = algorithm(
        policy,
        vec_env,
        **algorithm_kwargs
    )
    
    # 设置日志器
    model.set_logger(new_logger)
    
    # 开始训练
    agent_mode = "多智能体" if args.multi_agent else "单智能体"
    print(f"开始训练{args.algorithm.upper()}模型，模式: {agent_mode}, 观察模式: {args.mode}, 网格大小: {args.grid_size}")
    if args.multi_agent:
        print(f"蛇的数量: {args.num_snakes}")
    print(f"日志保存到: {log_dir}")
    print(f"模型保存到: {os.path.join('models', f'{args.algorithm}_snake')}")
    
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    # 保存最终模型
    model.save(os.path.join('models', f'{args.algorithm}_snake', f'{args.algorithm}_snake_final'))
    print("训练完成！")
    
    # 关闭环境
    vec_env.close()
    eval_env.close()
    
    return model

def main():
    parser = argparse.ArgumentParser(description='训练贪吃蛇智能体')
    
    # 基础参数
    parser.add_argument('--algorithm', type=str, default='ppo', 
                      choices=['ppo', 'dqn', 'a2c', 'ddpg', 'sac', 'td3'],
                      help='选择强化学习算法')
    
    # 环境参数
    parser.add_argument('--grid_size', type=int, default=10, help='游戏网格大小')
    parser.add_argument('--mode', type=str, default='feature', choices=['feature', 'pixel', 'grid'], help='观察模式')
    parser.add_argument('--num_snakes', type=int, default=1, help='蛇的数量')
    parser.add_argument('--multi_agent', action='store_true', help='是否启用多智能体模式')
    
    # 通用训练参数
    parser.add_argument('--n_envs', type=int, default=4, help='并行环境数量')
    parser.add_argument('--total_timesteps', type=int, default=1000000, help='总训练步数')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    
    # PPO特定参数
    parser.add_argument('--n_steps', type=int, default=2048, help='每批次收集的步数')
    parser.add_argument('--n_epochs', type=int, default=10, help='每批次训练的轮数')
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
    os.makedirs(os.path.join('models', f'{args.algorithm}_snake'), exist_ok=True)
    
    # 开始训练
    train_agent(args)

if __name__ == '__main__':
    main()
