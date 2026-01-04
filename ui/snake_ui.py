import pygame
import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# 确保能导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controller.snake_ai import SnakeAI
from environment.snake_env import SnakeEnv
from environment.multiagent_wrapper import ParallelMultiAgentWrapper

class SnakeUI:
    def __init__(self, grid_size=10, mode='feature', num_snakes=1, multi_agent=False):
        """
        贪吃蛇游戏用户界面
        
        Args:
            grid_size: 网格大小
            mode: 观察模式
            num_snakes: 蛇的数量
            multi_agent: 是否启用多智能体模式
        """
        self.grid_size = grid_size
        self.cell_size = 40
        self.screen_size = grid_size * self.cell_size
        self.mode = mode
        self.num_snakes = num_snakes
        self.multi_agent = multi_agent
        
        # Pygame初始化
        pygame.init()
        pygame.display.set_caption("贪吃蛇游戏 - SnakeAI")
        
        # 主屏幕
        self.screen = pygame.display.set_mode((self.screen_size + 200, self.screen_size))
        self.clock = pygame.time.Clock()
        
        # 颜色
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GRAY = (30, 30, 30)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.ORANGE = (255, 165, 0)
        self.PURPLE = (128, 0, 128)
        self.CYAN = (0, 255, 255)
        self.PINK = (255, 192, 203)
        
        # 蛇的颜色列表
        self.snake_colors = [
            (0, 255, 0),      # 绿色
            (255, 0, 0),      # 红色
            (0, 0, 255),      # 蓝色
            (255, 255, 0),    # 黄色
            (255, 165, 0),    # 橙色
            (128, 0, 128),    # 紫色
            (0, 255, 255),    # 青色
            (255, 192, 203)   # 粉色
        ]
        
        # AI控制器
        self.ai = None
        self.ai_mode = False
        
        # 游戏状态
        self.env = None
        self.reset_game()
        
        # 字体
        self.font = pygame.font.SysFont('Arial', 18)
        self.large_font = pygame.font.SysFont('Arial', 24)
    
    def reset_game(self):
        """重置游戏"""
        if self.env:
            self.env.close()
        
        # 创建环境（UI中不使用包装器，直接使用原始环境）
        self.env = SnakeEnv(
            render_mode=None,
            grid_size=self.grid_size,
            mode=self.mode,
            num_snakes=self.num_snakes,
            multi_agent=self.multi_agent
        )
        
        self.observation, self.info = self.env.reset()
        self.done = False
        self.score = [0] * self.num_snakes
        self.step_count = 0
        self.episode_reward = [0.0] * self.num_snakes
        self.last_action = [None] * self.num_snakes
    
    def load_ai(self, model_path):
        """加载AI模型"""
        try:
            self.ai = SnakeAI(model_path, mode=self.mode)
            self.ai_mode = True
            print(f"AI模型已加载: {model_path}")
        except Exception as e:
            print(f"加载AI模型失败: {e}")
            self.ai = None
            self.ai_mode = False
    
    def toggle_ai_mode(self):
        """切换AI模式"""
        self.ai_mode = not self.ai_mode
        print(f"AI模式: {'开启' if self.ai_mode else '关闭'}")
    
    def handle_events(self):
        """处理用户输入事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_r:
                    # 重置游戏
                    self.reset_game()
                elif event.key == pygame.K_SPACE:
                    # 暂停/继续
                    pass
                elif event.key == pygame.K_a:
                    # 切换AI模式
                    self.toggle_ai_mode()
                
                # 人工控制（只控制第一条蛇，多蛇情况下其他蛇使用默认动作）
                if not self.ai_mode:
                    if event.key == pygame.K_UP:
                        self.last_action[0] = 0
                    elif event.key == pygame.K_RIGHT:
                        self.last_action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        self.last_action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        self.last_action[0] = 3
        return True
    
    def update(self):
        """更新游戏状态"""
        # 检查游戏是否结束（处理多智能体模式的tuple done）
        if isinstance(self.done, tuple):
            all_done = all(self.done)
        else:
            all_done = self.done
        
        if all_done:
            if self.step_count == 0:
                print(f"Game ended immediately. done={self.done}, truncated={getattr(self, 'truncated', 'N/A')}")
            return
        
        print(f"Step {self.step_count}: Updating...")
        
        # 获取动作
        if self.ai_mode and self.ai:
            # 多智能体模式下需要为每条蛇预测动作
            if self.multi_agent:
                action = [self.ai.predict(obs) for obs in self.observation]
            else:
                action = self.ai.predict(self.observation)
            print(f"AI predicted action: {action}")
        elif any(a is not None for a in self.last_action):
            # 人工控制第一条蛇，其他蛇使用默认动作（0）
            action = []
            for i, a in enumerate(self.last_action):
                if i == 0 and a is not None:
                    action.append(a)
                else:
                    action.append(0)  # 默认动作
            action = action[0] if len(action) == 1 else action
        else:
            print("No action available")
            return  # 无动作，不更新
        
        # 执行动作
        if self.multi_agent:
            self.observation, reward, self.done, truncated, self.info = self.env.step(action)
        else:
            self.observation, reward, self.done, truncated, self.info = self.env.step(action)
        
        self.truncated = truncated
        
        # 更新状态
        self.step_count += 1
        
        # 更新每条蛇的得分和奖励
        scores = self.info.get('scores', [0] * self.num_snakes)
        for i in range(self.num_snakes):
            self.score[i] = scores[i] if i < len(scores) else 0
            if isinstance(reward, (list, tuple)):
                self.episode_reward[i] += reward[i]
            else:
                self.episode_reward[i] += reward
        
        print(f"Step {self.step_count} completed. done={self.done}, truncated={truncated}, reward={reward}")
    
    def render(self):
        """渲染游戏画面"""
        # 填充背景
        self.screen.fill(self.BLACK)
        
        # 绘制游戏区域
        self._render_game()
        
        # 绘制信息面板
        self._render_panel()
        
        # 刷新屏幕
        pygame.display.flip()
        self.clock.tick(15)  # 控制帧率
    
    def _render_game(self):
        """渲染游戏区域"""
        # 绘制网格
        for x in range(0, self.screen_size, self.cell_size):
            pygame.draw.line(self.screen, self.GRAY, (x, 0), (x, self.screen_size))
        for y in range(0, self.screen_size, self.cell_size):
            pygame.draw.line(self.screen, self.GRAY, (0, y), (self.screen_size, y))
        
        # 绘制食物
        food_rect = pygame.Rect(
            self.env.food[0] * self.cell_size,
            self.env.food[1] * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(self.screen, self.RED, food_rect)
        
        # 绘制多条蛇
        for snake_idx in range(self.num_snakes):
            if self.env.alive[snake_idx]:  # 只绘制活着的蛇
                snake = self.env.snakes[snake_idx]
                color = self.snake_colors[snake_idx % len(self.snake_colors)]
                darker_color = tuple(max(0, c - 50) for c in color)
                direction = self.env.directions[snake_idx]
                
                for i, (x, y) in enumerate(snake):
                    rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                    if i == 0:
                        # 蛇头
                        pygame.draw.rect(self.screen, color, rect)
                        # 绘制眼睛
                        self._draw_eyes(x, y, direction, color)
                    else:
                        # 蛇身
                        pygame.draw.rect(self.screen, darker_color, rect)
    
    def _draw_eyes(self, x, y, direction, color):
        """绘制蛇头眼睛"""
        eye_size = self.cell_size // 8
        eye_color = (255, 255, 255) if sum(color) > 380 else (0, 0, 0)  # 根据蛇头颜色选择眼睛颜色
        
        if direction == 0:  # 上
            eye1 = (x * self.cell_size + self.cell_size // 4, y * self.cell_size + self.cell_size // 4)
            eye2 = (x * self.cell_size + 3 * self.cell_size // 4 - eye_size, y * self.cell_size + self.cell_size // 4)
        elif direction == 1:  # 右
            eye1 = (x * self.cell_size + 3 * self.cell_size // 4 - eye_size, y * self.cell_size + self.cell_size // 4)
            eye2 = (x * self.cell_size + 3 * self.cell_size // 4 - eye_size, y * self.cell_size + 3 * self.cell_size // 4 - eye_size)
        elif direction == 2:  # 下
            eye1 = (x * self.cell_size + self.cell_size // 4, y * self.cell_size + 3 * self.cell_size // 4 - eye_size)
            eye2 = (x * self.cell_size + 3 * self.cell_size // 4 - eye_size, y * self.cell_size + 3 * self.cell_size // 4 - eye_size)
        else:  # 左
            eye1 = (x * self.cell_size + self.cell_size // 4, y * self.cell_size + self.cell_size // 4)
            eye2 = (x * self.cell_size + self.cell_size // 4, y * self.cell_size + 3 * self.cell_size // 4 - eye_size)
        
        pygame.draw.rect(self.screen, eye_color, (*eye1, eye_size, eye_size))
        pygame.draw.rect(self.screen, eye_color, (*eye2, eye_size, eye_size))
    
    def _render_panel(self):
        """渲染信息面板"""
        panel_x = self.screen_size
        panel_width = self.screen.get_width() - panel_x
        
        # 绘制面板背景
        pygame.draw.rect(self.screen, self.GRAY, (panel_x, 0, panel_width, self.screen_size))
        
        # 标题
        title_text = self.large_font.render("游戏信息", True, self.WHITE)
        self.screen.blit(title_text, (panel_x + 10, 10))
        
        # 得分（每条蛇的得分）
        for i in range(self.num_snakes):
            snake_color = self.snake_colors[i % len(self.snake_colors)]
            score_text = self.font.render(f"蛇{i+1}得分: {self.score[i]}", True, snake_color)
            self.screen.blit(score_text, (panel_x + 10, 50 + i * 30))
        
        # 步数
        step_text = self.font.render(f"步数: {self.step_count}", True, self.WHITE)
        self.screen.blit(step_text, (panel_x + 10, 50 + self.num_snakes * 30))
        
        # 奖励（每条蛇的奖励）
        for i in range(self.num_snakes):
            snake_color = self.snake_colors[i % len(self.snake_colors)]
            reward_text = self.font.render(f"蛇{i+1}奖励: {self.episode_reward[i]:.2f}", True, snake_color)
            self.screen.blit(reward_text, (panel_x + 10, 80 + self.num_snakes * 30 + i * 30))
        
        # AI模式
        ai_mode_text = self.font.render(f"AI模式: {'开启' if self.ai_mode else '关闭'}", True, self.GREEN if self.ai_mode else self.RED)
        self.screen.blit(ai_mode_text, (panel_x + 10, 80 + self.num_snakes * 60))
        
        # 最后动作
        if any(a is not None for a in self.last_action):
            action_names = ['上', '右', '下', '左']
            for i, a in enumerate(self.last_action):
                if a is not None:
                    snake_color = self.snake_colors[i % len(self.snake_colors)]
                    action_text = self.font.render(f"蛇{i+1}动作: {action_names[a]}", True, snake_color)
                    self.screen.blit(action_text, (panel_x + 10, 110 + self.num_snakes * 60 + i * 30))
        
        # 游戏状态
        if isinstance(self.done, tuple):
            done_status = all(self.done)
        else:
            done_status = self.done
        state_text = self.font.render(f"状态: {'结束' if done_status else '进行中'}", True, self.RED if done_status else self.GREEN)
        self.screen.blit(state_text, (panel_x + 10, 110 + self.num_snakes * 60 + self.num_snakes * 30))
        
        # 控制说明
        pygame.draw.line(self.screen, self.WHITE, (panel_x + 10, 230), (panel_x + panel_width - 10, 230), 1)
        
        controls = [
            "控制说明:",
            "↑↓←→: 方向控制",
            "R: 重置游戏",
            "A: 切换AI模式",
            "ESC: 退出游戏"
        ]
        
        for i, control in enumerate(controls):
            control_text = self.font.render(control, True, self.WHITE)
            self.screen.blit(control_text, (panel_x + 10, 240 + i * 25))
    
    def run(self):
        """运行游戏主循环"""
        running = True
        
        while running:
            # 处理事件
            running = self.handle_events()
            
            # 更新游戏状态
            self.update()
            
            # 渲染画面
            self.render()
        
        # 清理
        if self.env:
            self.env.close()
        if self.ai:
            self.ai.close()
        
        pygame.quit()
        print("游戏已退出")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='贪吃蛇游戏界面')
    parser.add_argument('--grid_size', type=int, default=10, help='游戏网格大小')
    parser.add_argument('--mode', type=str, default='feature', choices=['feature', 'pixel', 'grid'], help='观察模式')
    parser.add_argument('--model_path', type=str, default=None, help='AI模型路径')
    parser.add_argument('--num_snakes', type=int, default=1, help='蛇的数量')
    parser.add_argument('--multi_agent', action='store_true', help='是否启用多智能体模式')
    
    args = parser.parse_args()
    
    # 创建并运行游戏界面
    ui = SnakeUI(
        grid_size=args.grid_size, 
        mode=args.mode, 
        num_snakes=args.num_snakes,
        multi_agent=args.multi_agent
    )
    
    # 加载AI模型（如果提供）
    if args.model_path:
        ui.load_ai(args.model_path)
        ui.ai_mode = True
    
    ui.run()


if __name__ == '__main__':
    main()
