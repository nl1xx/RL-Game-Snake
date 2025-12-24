import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

# 确保能导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controller.snake_ai import SnakeAI
from environment.snake_env import SnakeEnv

def evaluate_model(args):
    """
    评估训练好的模型
    
    Args:
        args: 命令行参数
        
    Returns:
        results: 评估结果字典
    """
    print("开始评估模型...")
    print(f"模型路径: {args.model_path}")
    print(f"网格大小: {args.grid_size}")
    print(f"观察模式: {args.mode}")
    print(f"评估回合数: {args.n_episodes}")
    print(f"渲染: {args.render}")
    print("=" * 50)
    
    # 加载AI控制器
    ai = SnakeAI(model_path=args.model_path, mode=args.mode)
    
    # 创建环境
    env = gym.make(
        'SnakeEnv-v1',
        render_mode='human' if args.render else None,
        grid_size=args.grid_size,
        mode=args.mode,
        num_snakes=args.num_snakes,
        multi_agent=args.multi_agent
    )
    
    # 评估结果
    results = {
        'scores': [],
        'episode_lengths': [],
        'rewards': []
    }
    
    # 开始评估
    for episode in range(args.n_episodes):
        observation, info = env.reset()
        done = truncated = False
        episode_reward = 0
        episode_length = 0
        
        while not (done or truncated):
            # AI预测动作
            action = ai.predict(observation)
            
            # 执行动作
            observation, reward, done, truncated, info = env.step(action)
            
            # 更新统计
            episode_reward += reward
            episode_length += 1
        
        # 记录结果
        score = info.get('score', 0)
        results['scores'].append(score)
        results['episode_lengths'].append(episode_length)
        results['rewards'].append(episode_reward)
        
        print(f"回合 {episode+1:3d}: 得分 = {score:3d}, 长度 = {episode_length:3d}, 奖励 = {episode_reward:6.2f}")
    
    # 关闭环境
    env.close()
    ai.close()
    
    return results

def print_evaluation_summary(results):
    """
    打印评估总结
    
    Args:
        results: 评估结果字典
    """
    print("=" * 50)
    print("评估总结")
    print("=" * 50)
    
    scores = results['scores']
    episode_lengths = results['episode_lengths']
    rewards = results['rewards']
    
    print(f"总回合数: {len(scores)}")
    print(f"平均得分: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"最高得分: {np.max(scores)}")
    print(f"最低得分: {np.min(scores)}")
    print(f"\n平均回合长度: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"最长回合: {np.max(episode_lengths)}")
    print(f"最短回合: {np.min(episode_lengths)}")
    print(f"\n平均奖励: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"最高奖励: {np.max(rewards):.2f}")
    print(f"最低奖励: {np.min(rewards):.2f}")
    
    # 计算胜率（如果有定义）
    # 在贪吃蛇游戏中，我们可以定义"成功"为得分超过某个阈值
    success_threshold = np.mean(scores) + np.std(scores)
    success_count = sum(1 for s in scores if s > success_threshold)
    print(f"\n成功率: {success_count / len(scores) * 100:.2f}%")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='评估贪吃蛇智能体')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    
    # 环境参数
    parser.add_argument('--grid_size', type=int, default=10, help='游戏网格大小')
    parser.add_argument('--mode', type=str, default='feature', choices=['feature', 'pixel', 'grid'], help='观察模式')
    parser.add_argument('--num_snakes', type=int, default=1, help='蛇的数量')
    parser.add_argument('--multi_agent', action='store_true', help='是否启用多智能体模式')
    
    # 评估参数
    parser.add_argument('--n_episodes', type=int, default=20, help='评估回合数')
    parser.add_argument('--render', action='store_true', help='是否渲染游戏')
    parser.add_argument('--generalization_test', action='store_true', help='是否进行泛化能力测试')
    
    args = parser.parse_args()
    
    # 开始评估
    results = evaluate_model(args)
    
    # 打印评估总结
    print_evaluation_summary(results)


if __name__ == '__main__':
    main()