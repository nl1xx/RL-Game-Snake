import os
import sys
import argparse
import subprocess

def print_banner():
    """打印项目横幅"""
    banner = """
    +======================================+
    |                                      |
    |        深度强化学习贪吃蛇项目         |
    |         Deep Reinforcement Learning  |
    |               Snake Game             |
    |                                      |
    +======================================+
    """
    print(banner)

def print_help():
    """打印帮助信息"""
    print("可用命令:")
    print("  train    - 训练贪吃蛇智能体")
    print("  evaluate - 评估训练好的模型")
    print("  ui       - 启动游戏界面")
    print("  analyze  - 分析训练日志")
    print("  quantum  - 量子超参数搜索")
    print("  help     - 显示此帮助信息")
    print("\n使用示例:")
    print("  python main.py train --grid_size 10 --mode feature")
    print("  python main.py ui --model_path models/ppo_snake/ppo_snake_final")
    print("  python main.py evaluate --model_path models/ppo_snake/ppo_snake_final --n_episodes 20")
    print("  python main.py quantum --algorithm ppo --grid_size 10 --n_iterations 10")

def run_training(args):
    """运行训练模块"""
    print("启动训练模块...")
    cmd = [
        sys.executable, "training/train.py",
        "--algorithm", args.algorithm
    ]
    
    # 传递参数
    if args.grid_size:
        cmd.extend(["--grid_size", str(args.grid_size)])
    if args.mode:
        cmd.extend(["--mode", args.mode])
    if args.n_envs:
        cmd.extend(["--n_envs", str(args.n_envs)])
    if args.total_timesteps:
        cmd.extend(["--total_timesteps", str(args.total_timesteps)])
    if args.save_freq:
        cmd.extend(["--save_freq", str(args.save_freq)])
    if args.eval_freq:
        cmd.extend(["--eval_freq", str(args.eval_freq)])
    if args.learning_rate:
        cmd.extend(["--learning_rate", str(args.learning_rate)])
    if args.num_snakes:
        cmd.extend(["--num_snakes", str(args.num_snakes)])
    if args.multi_agent:
        cmd.append("--multi_agent")
    
    print(f"执行命令: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def run_evaluation(args):
    """运行评估模块"""
    print("启动评估模块...")
    
    if not args.model_path:
        print("错误: 必须提供模型路径")
        sys.exit(1)
    
    cmd = [
        sys.executable, "evaluation/evaluate_model.py",
        "--model_path", args.model_path
    ]
    
    # 传递参数
    if args.grid_size:
        cmd.extend(["--grid_size", str(args.grid_size)])
    if args.mode:
        cmd.extend(["--mode", args.mode])
    if args.n_episodes:
        cmd.extend(["--n_episodes", str(args.n_episodes)])
    if args.render:
        cmd.append("--render")
    if args.generalization_test:
        cmd.append("--generalization_test")
    if args.num_snakes:
        cmd.extend(["--num_snakes", str(args.num_snakes)])
    if args.multi_agent:
        cmd.append("--multi_agent")
    
    print(f"执行命令: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def run_ui(args):
    """运行用户界面"""
    print("启动游戏界面...")
    cmd = [
        sys.executable, "ui/snake_ui.py"
    ]
    
    # 传递参数
    if args.grid_size:
        cmd.extend(["--grid_size", str(args.grid_size)])
    if args.mode:
        cmd.extend(["--mode", args.mode])
    if args.model_path:
        cmd.extend(["--model_path", args.model_path])
    if hasattr(args, 'num_snakes') and args.num_snakes:
        cmd.extend(["--num_snakes", str(args.num_snakes)])
    if hasattr(args, 'multi_agent') and args.multi_agent:
        cmd.append("--multi_agent")
    
    print(f"执行命令: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def run_analysis(args):
    """运行分析模块"""
    print("启动分析模块...")
    
    # 这里可以直接导入并运行分析功能
    from analysis.data_analyzer import DataAnalyzer
    
    analyzer = DataAnalyzer()
    
    try:
        analyzer.load_training_logs()
        analyzer.load_eval_logs()
        
        analyzer.analyze_performance()
        analyzer.analyze_eval_performance()
        
        analyzer.plot_training_curve(window=100)
        analyzer.plot_eval_results()
        
    except Exception as e:
        print(f"分析失败: {e}")
        sys.exit(1)

def run_quantum_search(args):
    """运行量子超参数搜索"""
    print("启动量子超参数搜索...")
    cmd = [
        sys.executable, "training/quantum_hyperparameter_search.py",
        "--algorithm", args.algorithm
    ]
    
    # 传递参数
    if args.grid_size:
        cmd.extend(["--grid_size", str(args.grid_size)])
    if args.mode:
        cmd.extend(["--mode", args.mode])
    if args.n_envs:
        cmd.extend(["--n_envs", str(args.n_envs)])
    if args.total_timesteps:
        cmd.extend(["--total_timesteps", str(args.total_timesteps)])
    if args.n_iterations:
        cmd.extend(["--n_iterations", str(args.n_iterations)])
    if args.num_snakes:
        cmd.extend(["--num_snakes", str(args.num_snakes)])
    if args.multi_agent:
        cmd.append("--multi_agent")
    
    print(f"执行命令: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    """主函数"""
    print_banner()
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='深度强化学习贪吃蛇项目')
    
    # 子命令解析器
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练贪吃蛇智能体')
    train_parser.add_argument('--algorithm', type=str, default='ppo', 
                          choices=['ppo', 'dqn', 'a2c', 'ddpg', 'sac', 'td3'],
                          help='选择强化学习算法')
    train_parser.add_argument('--grid_size', type=int, default=10, help='游戏网格大小')
    train_parser.add_argument('--mode', type=str, default='feature', choices=['feature', 'pixel', 'grid'], help='观察模式')
    train_parser.add_argument('--n_envs', type=int, default=4, help='并行环境数量')
    train_parser.add_argument('--total_timesteps', type=int, default=1000000, help='总训练步数')
    train_parser.add_argument('--learning_rate', type=float, default=3e-4, help='学习率')
    train_parser.add_argument('--save_freq', type=int, default=10000, help='模型保存频率')
    train_parser.add_argument('--eval_freq', type=int, default=5000, help='评估频率')
    train_parser.add_argument('--num_snakes', type=int, default=1, help='蛇的数量')
    train_parser.add_argument('--multi_agent', action='store_true', help='是否启用多智能体模式')
    
    # 评估命令
    eval_parser = subparsers.add_parser('evaluate', help='评估训练好的模型')
    eval_parser.add_argument('--model_path', type=str, required=True, help='训练好的模型路径')
    eval_parser.add_argument('--grid_size', type=int, default=10, help='游戏网格大小')
    eval_parser.add_argument('--mode', type=str, default='feature', choices=['feature', 'pixel', 'grid'], help='观察模式')
    eval_parser.add_argument('--n_episodes', type=int, default=20, help='评估回合数')
    eval_parser.add_argument('--render', action='store_true', help='是否渲染游戏')
    eval_parser.add_argument('--generalization_test', action='store_true', help='是否进行泛化能力测试')
    eval_parser.add_argument('--num_snakes', type=int, default=1, help='蛇的数量')
    eval_parser.add_argument('--multi_agent', action='store_true', help='是否启用多智能体模式')
    
    # UI命令
    ui_parser = subparsers.add_parser('ui', help='启动游戏界面')
    ui_parser.add_argument('--grid_size', type=int, default=10, help='游戏网格大小')
    ui_parser.add_argument('--mode', type=str, default='feature', choices=['feature', 'pixel', 'grid'], help='观察模式')
    ui_parser.add_argument('--model_path', type=str, help='AI模型路径')
    ui_parser.add_argument('--num_snakes', type=int, default=1, help='蛇的数量')
    ui_parser.add_argument('--multi_agent', action='store_true', help='是否启用多智能体模式')
    
    # 分析命令
    analyze_parser = subparsers.add_parser('analyze', help='分析训练日志')
    
    # 量子超参数搜索命令
    quantum_parser = subparsers.add_parser('quantum', help='量子超参数搜索')
    quantum_parser.add_argument('--algorithm', type=str, default='ppo', 
                          choices=['ppo', 'dqn', 'a2c', 'ddpg', 'sac', 'td3'],
                          help='选择强化学习算法')
    quantum_parser.add_argument('--grid_size', type=int, default=10, help='游戏网格大小')
    quantum_parser.add_argument('--mode', type=str, default='feature', choices=['feature', 'pixel', 'grid'], help='观察模式')
    quantum_parser.add_argument('--n_envs', type=int, default=4, help='并行环境数量')
    quantum_parser.add_argument('--total_timesteps', type=int, default=100000, help='每轮训练的总步数（用于超参数搜索）')
    quantum_parser.add_argument('--n_iterations', type=int, default=20, help='搜索迭代次数')
    quantum_parser.add_argument('--num_snakes', type=int, default=1, help='蛇的数量')
    quantum_parser.add_argument('--multi_agent', action='store_true', help='是否启用多智能体模式')
    
    # 解析参数
    args = parser.parse_args()
    
    # 根据命令执行相应功能
    if args.command == 'train':
        run_training(args)
    elif args.command == 'evaluate':
        run_evaluation(args)
    elif args.command == 'ui':
        run_ui(args)
    elif args.command == 'analyze':
        run_analysis(args)
    elif args.command == 'quantum':
        run_quantum_search(args)
    else:
        print_help()


if __name__ == '__main__':
    main()
