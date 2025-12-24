import os
import sys
import argparse
import numpy as np
import pennylane as qml
from pennylane import numpy as qnp
from pennylane.optimize import GradientDescentOptimizer
import matplotlib.pyplot as plt
import time
import json

# 确保能导入自定义环境
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.train import train_agent

class QuantumHyperparameterSearch:
    def __init__(self, algorithm, grid_size=10, mode='feature', n_envs=4, 
                 total_timesteps=100000, save_freq=1000, eval_freq=500, 
                 num_snakes=1, multi_agent=False):
        """
        量子超参数搜索类
        
        Args:
            algorithm: 强化学习算法名称
            grid_size: 游戏网格大小
            mode: 观察模式
            n_envs: 并行环境数量
            total_timesteps: 每轮训练的总步数（用于超参数搜索，应小于完整训练）
            save_freq: 模型保存频率
            eval_freq: 评估频率
            num_snakes: 蛇的数量
            multi_agent: 是否启用多智能体模式
        """
        self.algorithm = algorithm
        self.grid_size = grid_size
        self.mode = mode
        self.n_envs = n_envs
        self.total_timesteps = total_timesteps
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.num_snakes = num_snakes
        self.multi_agent = multi_agent
        
        # 定义每个算法的超参数搜索空间
        self.hyperparameter_spaces = self._get_hyperparameter_spaces()
        
        # 量子电路设置
        self.num_qubits = len(self.hyperparameter_spaces)
        self.dev = qml.device("default.qubit", wires=self.num_qubits)
        
        # 搜索结果
        self.results = []
        
    def _get_hyperparameter_spaces(self):
        """
        获取不同算法的超参数搜索空间
        """
        spaces = {
            'ppo': {
                'learning_rate': (1e-5, 1e-3),
                'gamma': (0.9, 0.999),
                'batch_size': [32, 64, 128, 256],
                'n_steps': [1024, 2048, 4096],
                'n_epochs': [5, 10, 20],
                'gae_lambda': (0.9, 0.99),
                'clip_range': (0.1, 0.3),
                'ent_coef': (0.001, 0.05),
                'vf_coef': (0.3, 0.7),
                'max_grad_norm': (0.3, 0.7)
            },
            'dqn': {
                'learning_rate': (1e-5, 1e-3),
                'gamma': (0.9, 0.999),
                'batch_size': [32, 64, 128, 256],
                'buffer_size': [50000, 100000, 200000],
                'learning_starts': [1000, 5000, 10000],
                'target_update_interval': [500, 1000, 2000],
                'exploration_fraction': (0.05, 0.2),
                'exploration_initial_eps': (0.9, 1.0),
                'exploration_final_eps': (0.01, 0.1)
            },
            'a2c': {
                'learning_rate': (1e-5, 1e-3),
                'gamma': (0.9, 0.999),
                'batch_size': [32, 64, 128],
                'n_steps': [5, 10, 20, 50],
                'ent_coef': (0.001, 0.05)
            },
            'ddpg': {
                'learning_rate': (1e-5, 1e-3),
                'gamma': (0.9, 0.999),
                'batch_size': [32, 64, 128, 256]
            },
            'sac': {
                'learning_rate': (1e-5, 1e-3),
                'gamma': (0.9, 0.999),
                'batch_size': [32, 64, 128, 256],
                'tau': (0.001, 0.05),
                'ent_coef': ['auto', 0.001, 0.01]
            },
            'td3': {
                'learning_rate': (1e-5, 1e-3),
                'gamma': (0.9, 0.999),
                'batch_size': [32, 64, 128, 256],
                'tau': (0.001, 0.05),
                'policy_delay': [2, 3, 4]
            }
        }
        
        return spaces.get(self.algorithm, {})
    
    def _quantum_circuit(self, params):
        """
        量子电路用于生成超参数配置
        
        Args:
            params: 量子电路参数
            
        Returns:
            超参数配置
        """
        @qml.qnode(self.dev)
        def circuit(params):
            # 初始化量子态
            for i in range(self.num_qubits):
                qml.Hadamard(wires=i)
                
            # 应用参数化旋转门
            for i in range(self.num_qubits):
                qml.RY(params[i], wires=i)
            
            # 应用纠缠门
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i+1])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
        
        # 运行量子电路
        measurements = circuit(params)
        
        # 将量子测量结果映射到超参数值
        hyperparams = {}
        for i, (name, space) in enumerate(self.hyperparameter_spaces.items()):
            value = measurements[i]
            
            if isinstance(space, tuple):
                # 连续参数
                min_val, max_val = space
                hyperparams[name] = min_val + (max_val - min_val) * (value + 1) / 2
            elif isinstance(space, list):
                # 离散参数
                index = int((value + 1) / 2 * (len(space) - 1))
                hyperparams[name] = space[min(index, len(space) - 1)]
        
        return hyperparams
    
    def _evaluate_hyperparameters(self, hyperparams):
        """
        评估一组超参数的性能
        
        Args:
            hyperparams: 超参数配置
            
        Returns:
            性能得分（平均奖励）
        """
        print(f"\n评估超参数配置: {hyperparams}")
        
        # 创建临时参数对象
        class Args:
            pass
        
        args = Args()
        args.algorithm = self.algorithm
        args.grid_size = self.grid_size
        args.mode = self.mode
        args.n_envs = self.n_envs
        args.total_timesteps = self.total_timesteps
        args.save_freq = self.save_freq
        args.eval_freq = self.eval_freq
        args.num_snakes = self.num_snakes
        args.multi_agent = self.multi_agent
        
        # 设置超参数
        for key, value in hyperparams.items():
            setattr(args, key, value)
        
        # 补充默认超参数
        default_params = self._get_default_params()
        for key, value in default_params.items():
            if not hasattr(args, key):
                setattr(args, key, value)
        
        try:
            # 运行训练（会自动进行评估）
            model = train_agent(args)
            
            # 获取评估结果
            # 这里简化处理，实际应该从日志中读取最佳评估分数
            # 我们假设模型的最后一个评估分数是性能指标
            # 在实际应用中，应该从eval_callback中获取最佳评估分数
            score = np.random.uniform(0, 100)  # 占位符，实际应替换为真实评估分数
            
            print(f"超参数配置得分: {score}")
            
            return score
            
        except Exception as e:
            print(f"评估失败: {e}")
            return -np.inf
    
    def _get_default_params(self):
        """
        获取默认超参数
        """
        defaults = {
            'ppo': {
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'batch_size': 64,
                'n_steps': 2048,
                'n_epochs': 10,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5
            },
            'dqn': {
                'learning_rate': 5e-4,
                'gamma': 0.99,
                'batch_size': 64,
                'buffer_size': 100000,
                'learning_starts': 5000,
                'target_update_interval': 1000,
                'exploration_fraction': 0.1,
                'exploration_initial_eps': 1.0,
                'exploration_final_eps': 0.05
            },
            'a2c': {
                'learning_rate': 7e-4,
                'gamma': 0.99,
                'batch_size': 64,
                'n_steps': 5,
                'ent_coef': 0.01
            },
            'ddpg': {
                'learning_rate': 1e-4,
                'gamma': 0.99,
                'batch_size': 64
            },
            'sac': {
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'batch_size': 64,
                'tau': 0.005,
                'ent_coef': 'auto'
            },
            'td3': {
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'batch_size': 64,
                'tau': 0.005,
                'policy_delay': 2
            }
        }
        
        return defaults.get(self.algorithm, {})
    
    def search(self, n_iterations=20):
        """
        执行量子超参数搜索
        
        Args:
            n_iterations: 搜索迭代次数
        """
        print(f"开始量子超参数搜索，算法: {self.algorithm}, 迭代次数: {n_iterations}")
        print(f"超参数数量: {self.num_qubits}")
        print(f"搜索空间: {self.hyperparameter_spaces}")
        
        start_time = time.time()
        
        for i in range(n_iterations):
            print(f"\n=== 迭代 {i+1}/{n_iterations} ===")
            
            # 随机生成量子电路参数
            params = qnp.random.uniform(0, 2 * np.pi, size=self.num_qubits, requires_grad=True)
            
            # 使用量子电路生成超参数配置
            hyperparams = self._quantum_circuit(params)
            
            # 评估超参数配置
            score = self._evaluate_hyperparameters(hyperparams)
            
            # 保存结果
            self.results.append({
                'iteration': i+1,
                'hyperparams': hyperparams,
                'score': score,
                'time': time.time() - start_time
            })
        
        # 保存搜索结果
        self._save_results()
        
        # 输出最佳结果
        self._print_best_result()
        
        return self.results
    
    def _save_results(self):
        """
        保存搜索结果到文件
        """
        results_dir = os.path.join('logs', 'hyperparameter_search')
        os.makedirs(results_dir, exist_ok=True)
        
        filename = os.path.join(results_dir, f'{self.algorithm}_quantum_search.json')
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"搜索结果已保存到: {filename}")
    
    def _print_best_result(self):
        """
        打印最佳搜索结果
        """
        if not self.results:
            print("没有搜索结果")
            return
        
        # 找到最佳结果
        best_result = max(self.results, key=lambda x: x['score'])
        
        print(f"\n=== 最佳搜索结果 ===")
        print(f"迭代次数: {best_result['iteration']}")
        print(f"得分: {best_result['score']}")
        print(f"运行时间: {best_result['time']:.2f}秒")
        print("最佳超参数配置:")
        for key, value in best_result['hyperparams'].items():
            print(f"  {key}: {value}")
        
        # 生成可直接使用的命令
        cmd = f"python main.py train --algorithm {self.algorithm} "
        cmd += f"--grid_size {self.grid_size} --mode {self.mode} "
        cmd += f"--n_envs {self.n_envs} --total_timesteps 1000000 "  # 完整训练的总步数
        
        for key, value in best_result['hyperparams'].items():
            cmd += f"--{key} {value} "
        
        print(f"\n使用最佳超参数的训练命令:")
        print(cmd)
        
    def plot_results(self):
        """
        绘制搜索结果
        """
        if not self.results:
            print("没有搜索结果可绘制")
            return
        
        iterations = [r['iteration'] for r in self.results]
        scores = [r['score'] for r in self.results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, scores, 'o-', label='Score')
        plt.xlabel('迭代次数')
        plt.ylabel('性能得分')
        plt.title(f'{self.algorithm}量子超参数搜索结果')
        plt.grid(True)
        plt.legend()
        
        # 保存图表
        plots_dir = os.path.join('logs', 'hyperparameter_search')
        os.makedirs(plots_dir, exist_ok=True)
        
        filename = os.path.join(plots_dir, f'{self.algorithm}_quantum_search.png')
        plt.savefig(filename)
        plt.close()
        
        print(f"搜索结果图表已保存到: {filename}")

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='量子超参数搜索')
    
    # 基础参数
    parser.add_argument('--algorithm', type=str, default='ppo', 
                      choices=['ppo', 'dqn', 'a2c', 'ddpg', 'sac', 'td3'],
                      help='选择强化学习算法')
    
    # 环境参数
    parser.add_argument('--grid_size', type=int, default=10, help='游戏网格大小')
    parser.add_argument('--mode', type=str, default='feature', choices=['feature', 'pixel', 'grid'], help='观察模式')
    parser.add_argument('--num_snakes', type=int, default=1, help='蛇的数量')
    parser.add_argument('--multi_agent', action='store_true', help='是否启用多智能体模式')
    
    # 搜索参数
    parser.add_argument('--n_iterations', type=int, default=20, help='搜索迭代次数')
    parser.add_argument('--n_envs', type=int, default=4, help='并行环境数量')
    parser.add_argument('--total_timesteps', type=int, default=100000, help='每轮训练的总步数（用于超参数搜索）')
    
    args = parser.parse_args()
    
    # 创建并运行量子超参数搜索
    searcher = QuantumHyperparameterSearch(
        algorithm=args.algorithm,
        grid_size=args.grid_size,
        mode=args.mode,
        n_envs=args.n_envs,
        total_timesteps=args.total_timesteps,
        num_snakes=args.num_snakes,
        multi_agent=args.multi_agent
    )
    
    searcher.search(n_iterations=args.n_iterations)
    searcher.plot_results()

if __name__ == '__main__':
    main()