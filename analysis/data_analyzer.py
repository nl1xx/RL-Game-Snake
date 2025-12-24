import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

sns.set_style("darkgrid")
plt.rcParams.update({'font.size': 12})

class DataAnalyzer:
    def __init__(self, log_dir='logs/ppo_snake'):
        """
        数据采集与分析器
        
        Args:
            log_dir: 训练日志目录
        """
        self.log_dir = log_dir
        self.df = None
        self.eval_df = None
        
    def load_training_logs(self):
        """加载训练日志"""
        csv_path = os.path.join(self.log_dir, "progress.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"训练日志文件不存在: {csv_path}")
        
        self.df = pd.read_csv(csv_path)
        print(f"加载训练日志完成，共 {len(self.df)} 条记录")
        return self.df
    
    def load_eval_logs(self):
        """加载评估日志"""
        eval_csv_path = os.path.join(self.log_dir, "eval", "evaluations.csv")
        if not os.path.exists(eval_csv_path):
            raise FileNotFoundError(f"评估日志文件不存在: {eval_csv_path}")
        
        self.eval_df = pd.read_csv(eval_csv_path)
        print(f"加载评估日志完成，共 {len(self.eval_df)} 条记录")
        return self.eval_df
    
    def plot_training_curve(self, window=100, save_path=None):
        """
        绘制训练曲线
        
        Args:
            window: 平滑窗口大小
            save_path: 保存路径，None表示不保存
        """
        if self.df is None:
            self.load_training_logs()
        
        plt.figure(figsize=(12, 8))
        
        # 绘制累计奖励曲线
        plt.subplot(2, 2, 1)
        if 'rollout/ep_len_mean' in self.df.columns:
            plt.plot(self.df['time/total_timesteps'], self.df['rollout/ep_len_mean'], alpha=0.3, label='原始')
            plt.plot(self.df['time/total_timesteps'], self.df['rollout/ep_len_mean'].rolling(window=window).mean(), label=f'平滑({window})')
            plt.xlabel('总步数')
            plt.ylabel('平均回合长度')
            plt.title('训练过程中的平均回合长度')
            plt.legend()
        
        plt.subplot(2, 2, 2)
        if 'rollout/ep_rew_mean' in self.df.columns:
            plt.plot(self.df['time/total_timesteps'], self.df['rollout/ep_rew_mean'], alpha=0.3, label='原始')
            plt.plot(self.df['time/total_timesteps'], self.df['rollout/ep_rew_mean'].rolling(window=window).mean(), label=f'平滑({window})')
            plt.xlabel('总步数')
            plt.ylabel('平均奖励')
            plt.title('训练过程中的平均奖励')
            plt.legend()
        
        # 绘制PPO特定指标
        plt.subplot(2, 2, 3)
        if 'train/value_loss' in self.df.columns:
            plt.plot(self.df['time/total_timesteps'], self.df['train/value_loss'], alpha=0.5)
            plt.xlabel('总步数')
            plt.ylabel('值函数损失')
            plt.title('值函数损失')
        
        plt.subplot(2, 2, 4)
        if 'train/policy_gradient_loss' in self.df.columns:
            plt.plot(self.df['time/total_timesteps'], self.df['train/policy_gradient_loss'], alpha=0.5, label='策略梯度损失')
        if 'train/entropy_loss' in self.df.columns:
            plt.plot(self.df['time/total_timesteps'], self.df['train/entropy_loss'], alpha=0.5, label='熵损失')
        plt.xlabel('总步数')
        plt.ylabel('损失值')
        plt.title('策略和熵损失')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练曲线已保存到: {save_path}")
        
        plt.show()
    
    def plot_eval_results(self, save_path=None):
        """
        绘制评估结果
        
        Args:
            save_path: 保存路径，None表示不保存
        """
        if self.eval_df is None:
            self.load_eval_logs()
        
        plt.figure(figsize=(10, 6))
        
        # 绘制评估奖励
        plt.subplot(1, 2, 1)
        plt.plot(self.eval_df['timesteps'], self.eval_df['mean_reward'], marker='o')
        plt.fill_between(
            self.eval_df['timesteps'],
            self.eval_df['mean_reward'] - self.eval_df['std_reward'],
            self.eval_df['mean_reward'] + self.eval_df['std_reward'],
            alpha=0.2
        )
        plt.xlabel('总步数')
        plt.ylabel('平均奖励')
        plt.title('评估平均奖励')
        plt.grid(True)
        
        # 绘制评估回合长度
        plt.subplot(1, 2, 2)
        plt.plot(self.eval_df['timesteps'], self.eval_df['mean_ep_length'], marker='o')
        plt.fill_between(
            self.eval_df['timesteps'],
            self.eval_df['mean_ep_length'] - self.eval_df['std_ep_length'],
            self.eval_df['mean_ep_length'] + self.eval_df['std_ep_length'],
            alpha=0.2
        )
        plt.xlabel('总步数')
        plt.ylabel('平均回合长度')
        plt.title('评估平均回合长度')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"评估结果已保存到: {save_path}")
        
        plt.show()
    
    def analyze_performance(self):
        """分析模型性能指标"""
        if self.df is None:
            self.load_training_logs()
        
        print("===== 训练性能分析 =====")
        
        # 回合长度分析
        if 'rollout/ep_len_mean' in self.df.columns:
            print(f"最终平均回合长度: {self.df['rollout/ep_len_mean'].iloc[-1]:.2f}")
            print(f"最大平均回合长度: {self.df['rollout/ep_len_mean'].max():.2f}")
            print(f"最小平均回合长度: {self.df['rollout/ep_len_mean'].min():.2f}")
            print(f"平均回合长度增长: {self.df['rollout/ep_len_mean'].iloc[-1] / self.df['rollout/ep_len_mean'].iloc[0]:.2f}x")
        
        # 奖励分析
        if 'rollout/ep_rew_mean' in self.df.columns:
            print(f"\n最终平均奖励: {self.df['rollout/ep_rew_mean'].iloc[-1]:.2f}")
            print(f"最大平均奖励: {self.df['rollout/ep_rew_mean'].max():.2f}")
            print(f"最小平均奖励: {self.df['rollout/ep_rew_mean'].min():.2f}")
            print(f"平均奖励增长: {self.df['rollout/ep_rew_mean'].iloc[-1] / self.df['rollout/ep_rew_mean'].iloc[0]:.2f}x")
        
        # 训练时间分析
        if 'time/total_timesteps' in self.df.columns and 'time/fps' in self.df.columns:
            total_timesteps = self.df['time/total_timesteps'].iloc[-1]
            avg_fps = self.df['time/fps'].mean()
            total_time = total_timesteps / avg_fps / 3600  # 转换为小时
            print(f"\n总训练步数: {total_timesteps:,}")
            print(f"平均FPS: {avg_fps:.2f}")
            print(f"总训练时间: {total_time:.2f} 小时")
    
    def analyze_eval_performance(self):
        """分析评估性能指标"""
        if self.eval_df is None:
            self.load_eval_logs()
        
        print("\n===== 评估性能分析 =====")
        
        # 评估奖励分析
        if 'mean_reward' in self.eval_df.columns:
            print(f"最佳平均奖励: {self.eval_df['mean_reward'].max():.2f}")
            print(f"最终评估奖励: {self.eval_df['mean_reward'].iloc[-1]:.2f}")
            print(f"平均评估奖励: {self.eval_df['mean_reward'].mean():.2f}")
            print(f"评估奖励标准差: {self.eval_df['std_reward'].mean():.2f}")
        
        # 评估回合长度分析
        if 'mean_ep_length' in self.eval_df.columns:
            print(f"\n最佳平均回合长度: {self.eval_df['mean_ep_length'].max():.2f}")
            print(f"最终评估回合长度: {self.eval_df['mean_ep_length'].iloc[-1]:.2f}")
            print(f"平均评估回合长度: {self.eval_df['mean_ep_length'].mean():.2f}")
            print(f"评估回合长度标准差: {self.eval_df['std_ep_length'].mean():.2f}")

def main():
    """示例用法"""
    analyzer = DataAnalyzer()
    
    try:
        # 加载日志
        analyzer.load_training_logs()
        analyzer.load_eval_logs()
        
        # 分析性能
        analyzer.analyze_performance()
        analyzer.analyze_eval_performance()
        
        # 绘制曲线
        analyzer.plot_training_curve(window=100, save_path='analysis/training_curve.png')
        analyzer.plot_eval_results(save_path='analysis/eval_results.png')
        
    except Exception as e:
        print(f"分析过程中出错: {e}")

if __name__ == '__main__':
    main()
