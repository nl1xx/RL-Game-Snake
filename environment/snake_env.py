import pygame
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random

class SnakeEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 15}
    
    def __init__(self, render_mode=None, grid_size=10, mode='feature', num_snakes=1, multi_agent=False):
        super().__init__()
        
        # 游戏配置
        self.grid_size = grid_size
        self.cell_size = 40
        self.screen_size = grid_size * self.cell_size
        self.mode = mode  # 'feature', 'pixel', 'grid'
        self.num_snakes = num_snakes
        self.multi_agent = multi_agent
        
        # 蛇的颜色
        self.snake_colors = [
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 红色
            (0, 0, 255),    # 蓝色
            (255, 255, 0),  # 黄色
            (255, 0, 255),  # 紫色
            (0, 255, 255)   # 青色
        ]
        
        # 动作空间：0=上, 1=右, 2=下, 3=左
        if self.multi_agent:
            self.action_space = tuple([spaces.Discrete(4) for _ in range(num_snakes)])
        else:
            self.action_space = spaces.Discrete(4)
        
        # 观察空间
        if mode == 'feature':
            # 特征向量：蛇头位置相对食物位置(4)、蛇身体方向(4)、障碍物检测(4) + 其他蛇信息
            if self.multi_agent:
                # 每个蛇独立的观察空间
                self.observation_space = tuple([spaces.Box(low=-1, high=1, shape=(12 + 4 * (num_snakes - 1),), dtype=np.float32) for _ in range(num_snakes)])
            else:
                self.observation_space = spaces.Box(low=-1, high=1, shape=(12 + 4 * (num_snakes - 1),), dtype=np.float32)
        elif mode == 'pixel':
            # 像素观察：RGB图像
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_size, self.screen_size, 3), dtype=np.uint8)
        elif mode == 'grid':
            # 网格编码：0=空, 1-2*num_snakes=蛇身和蛇头(每个蛇用两个连续数字), 2*num_snakes+1=食物
            max_value = 2 * num_snakes + 1
            self.observation_space = spaces.Box(low=0, high=max_value, shape=(grid_size, grid_size), dtype=np.int8)
        
        # 渲染模式
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        # 游戏状态
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 初始化多条蛇
        self.snakes = []
        self.directions = []
        self.next_directions = []
        self.scores = []
        self.steps_since_food = []
        self.alive = []
        
        # 为每条蛇选择不同的起始位置
        start_positions = self._get_start_positions(self.num_snakes)
        start_directions = [1, 3, 0, 2]  # 右、左、上、下，循环使用
        
        for i in range(self.num_snakes):
            # 初始化蛇：长度为2
            start_pos = start_positions[i]
            if start_directions[i % 4] == 1:  # 右
                snake = [start_pos, (start_pos[0] - 1, start_pos[1])]
            elif start_directions[i % 4] == 3:  # 左
                snake = [start_pos, (start_pos[0] + 1, start_pos[1])]
            elif start_directions[i % 4] == 0:  # 上
                snake = [start_pos, (start_pos[0], start_pos[1] + 1)]
            else:  # 下
                snake = [start_pos, (start_pos[0], start_pos[1] - 1)]
            
            self.snakes.append(snake)
            self.directions.append(start_directions[i % 4])
            self.next_directions.append(start_directions[i % 4])
            self.scores.append(0)
            self.steps_since_food.append(0)
            self.alive.append(True)
        
        # 生成初始食物
        self.food = self._generate_food()
        
        # 游戏统计
        self.max_steps_per_snake = [100 * len(snake) for snake in self.snakes]
        
        # 渲染
        if self.render_mode == 'human':
            self._render_frame()
        
        return self._get_observation(), {}
    
    def _get_start_positions(self, num_snakes):
        """为多条蛇生成不同的起始位置"""
        positions = []
        grid_quarters = {
            0: (self.grid_size // 4, self.grid_size // 4),      # 左上角
            1: (3 * self.grid_size // 4, self.grid_size // 4),    # 右上角
            2: (self.grid_size // 4, 3 * self.grid_size // 4),    # 左下角
            3: (3 * self.grid_size // 4, 3 * self.grid_size // 4)  # 右下角
        }
        
        for i in range(num_snakes):
            base_pos = grid_quarters[i % 4]
            # 添加一点随机偏移，避免蛇过于靠近角落
            offset_x = random.randint(-1, 1)
            offset_y = random.randint(-1, 1)
            pos = (max(1, min(self.grid_size - 2, base_pos[0] + offset_x)), 
                  max(1, min(self.grid_size - 2, base_pos[1] + offset_y)))
            positions.append(pos)
        
        return positions
    
    def _generate_food(self):
        while True:
            food = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            # 检查食物是否在任何一条蛇身上
            food_in_snake = False
            for snake in self.snakes:
                if food in snake:
                    food_in_snake = True
                    break
            if not food_in_snake:
                return food
    
    def _get_observation(self):
        if self.mode == 'feature':
            observations = []
            for snake_idx in range(self.num_snakes):
                if not self.alive[snake_idx]:
                    # 如果蛇已经死亡，返回全零向量
                    observations.append(np.zeros(12 + 4 * (self.num_snakes - 1), dtype=np.float32))
                    continue
                
                snake = self.snakes[snake_idx]
                head = snake[0]
                direction = self.directions[snake_idx]
                
                # 相对食物位置
                food_rel_x = (self.food[0] - head[0]) / self.grid_size
                food_rel_y = (self.food[1] - head[1]) / self.grid_size
                
                # 蛇头在网格中的位置比例
                head_x = head[0] / self.grid_size
                head_y = head[1] / self.grid_size
                
                # 身体方向的one-hot编码
                direction_vec = [0] * 4
                direction_vec[direction] = 1
                
                # 障碍物检测（1步以内，包括墙壁和其他蛇）
                obstacles = [0] * 4
                
                # 上
                up_pos = (head[0], head[1] - 1)
                if head[1] == 0 or self._is_position_in_any_snake(up_pos, exclude=snake_idx):
                    obstacles[0] = 1
                # 右
                right_pos = (head[0] + 1, head[1])
                if head[0] == self.grid_size - 1 or self._is_position_in_any_snake(right_pos, exclude=snake_idx):
                    obstacles[1] = 1
                # 下
                down_pos = (head[0], head[1] + 1)
                if head[1] == self.grid_size - 1 or self._is_position_in_any_snake(down_pos, exclude=snake_idx):
                    obstacles[2] = 1
                # 左
                left_pos = (head[0] - 1, head[1])
                if head[0] == 0 or self._is_position_in_any_snake(left_pos, exclude=snake_idx):
                    obstacles[3] = 1
                
                # 其他蛇的信息（相对位置）
                other_snakes_info = []
                for other_idx in range(self.num_snakes):
                    if other_idx != snake_idx and self.alive[other_idx]:
                        other_head = self.snakes[other_idx][0]
                        rel_x = (other_head[0] - head[0]) / self.grid_size
                        rel_y = (other_head[1] - head[1]) / self.grid_size
                        other_snakes_info.extend([rel_x, rel_y, len(self.snakes[other_idx]) / 20.0, 1.0])  # 相对位置、长度比例、是否存活
                    elif other_idx != snake_idx:
                        other_snakes_info.extend([0.0, 0.0, 0.0, 0.0])  # 其他蛇已死亡
                
                # 合并所有特征
                obs = np.array([food_rel_x, food_rel_y, head_x, head_y] + direction_vec + obstacles + other_snakes_info, dtype=np.float32)
                observations.append(obs)
            
            if self.multi_agent:
                return tuple(observations)
            else:
                return observations[0]  # 单智能体模式下只返回第一条蛇的观察
        
        elif self.mode == 'pixel':
            # 渲染到像素数组
            if self.screen is None:
                pygame.init()
                self.screen = pygame.Surface((self.screen_size, self.screen_size))
            self._render_frame()
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))
        
        elif self.mode == 'grid':
            grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
            
            # 绘制每条蛇
            for snake_idx in range(self.num_snakes):
                if not self.alive[snake_idx]:
                    continue
                    
                snake = self.snakes[snake_idx]
                head_val = 2 * snake_idx + 2  # 蛇头值：2, 4, 6, ...
                body_val = 2 * snake_idx + 1   # 蛇身值：1, 3, 5, ...
                
                for i, pos in enumerate(snake):
                    if i == 0:
                        grid[pos] = head_val  # 蛇头
                    else:
                        grid[pos] = body_val  # 蛇身
            
            # 绘制食物
            grid[self.food] = 2 * self.num_snakes + 1  # 食物值
            
            return grid
    
    def _is_position_in_any_snake(self, pos, exclude=None):
        """检查位置是否在任何一条蛇身上，可选排除特定蛇"""
        for snake_idx in range(self.num_snakes):
            if snake_idx == exclude:
                continue
            if self.alive[snake_idx] and pos in self.snakes[snake_idx]:
                return True
        return False
    
    def step(self, action):
        # 处理动作：如果是单智能体模式，转换为多智能体格式
        if not self.multi_agent:
            actions = [action] * self.num_snakes
        else:
            actions = action
        
        # 防止反向移动并更新方向
        opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        for snake_idx in range(self.num_snakes):
            if not self.alive[snake_idx]:
                continue
                
            action = actions[snake_idx]
            current_direction = self.directions[snake_idx]
            if action != opposites.get(current_direction, -1):
                self.next_directions[snake_idx] = action
        
        self.directions = self.next_directions.copy()
        
        # 奖励初始化
        rewards = [0.0] * self.num_snakes
        dones = [False] * self.num_snakes
        all_done = False
        
        # 计算新头部位置并检查碰撞
        for snake_idx in range(self.num_snakes):
            if not self.alive[snake_idx]:
                rewards[snake_idx] = 0.0
                dones[snake_idx] = True
                continue
                
            # 设置基础奖励
            rewards[snake_idx] = -0.01  # 每步微小惩罚，鼓励高效
            
            snake = self.snakes[snake_idx]
            direction = self.directions[snake_idx]
            head = snake[0]
            
            # 计算新头部位置
            if direction == 0:  # 上
                new_head = (head[0], head[1] - 1)
            elif direction == 1:  # 右
                new_head = (head[0] + 1, head[1])
            elif direction == 2:  # 下
                new_head = (head[0], head[1] + 1)
            else:  # 左
                new_head = (head[0] - 1, head[1])
            
            # 检查碰撞（墙壁、自身、其他蛇）
            collision = False
            
            # 检查墙壁
            if (new_head[0] < 0 or new_head[0] >= self.grid_size or 
                new_head[1] < 0 or new_head[1] >= self.grid_size):
                collision = True
            
            # 检查自身
            if not collision and new_head in snake:
                collision = True
            
            # 检查其他蛇
            if not collision and self._is_position_in_any_snake(new_head, exclude=snake_idx):
                collision = True
            
            if collision:
                rewards[snake_idx] = -10  # 碰撞惩罚
                dones[snake_idx] = True
                self.alive[snake_idx] = False
                continue
            
            # 更新蛇身
            self.snakes[snake_idx].insert(0, new_head)
            
            # 检查是否吃到食物
            if new_head == self.food:
                rewards[snake_idx] = 10  # 吃到食物奖励
                self.scores[snake_idx] += 1
                self.steps_since_food[snake_idx] = 0
                self.max_steps_per_snake[snake_idx] = 100 * len(self.snakes[snake_idx])
                self.food = self._generate_food()
            else:
                self.snakes[snake_idx].pop()
                self.steps_since_food[snake_idx] += 1
            
            # 检查是否超时
            if self.steps_since_food[snake_idx] >= self.max_steps_per_snake[snake_idx]:
                rewards[snake_idx] = -5  # 超时惩罚
                dones[snake_idx] = True
                self.alive[snake_idx] = False
        
        # 检查是否所有蛇都死亡
        all_done = all(dones)
        
        # 渲染
        if self.render_mode == 'human':
            self._render_frame()
        
        # 处理返回值
        observations = self._get_observation()
        if self.multi_agent:
            return observations, tuple(rewards), tuple(dones), all_done, {'scores': self.scores}
        else:
            # 单智能体模式下只返回第一条蛇的信息
            return observations, rewards[0], dones[0], all_done, {'scores': self.scores}
    
    def _render_frame(self):
        if self.screen is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # 更新窗口标题
        if self.num_snakes > 1:
            scores_text = ", ".join([f"Snake {i+1}: {self.scores[i]}" for i in range(self.num_snakes)])
            pygame.display.set_caption(f"Snake Game - {scores_text}")
        else:
            pygame.display.set_caption(f"Snake Game - Score: {self.scores[0]}")
        
        # 填充背景
        self.screen.fill((0, 0, 0))
        
        # 绘制网格
        for x in range(0, self.screen_size, self.cell_size):
            pygame.draw.line(self.screen, (30, 30, 30), (x, 0), (x, self.screen_size))
        for y in range(0, self.screen_size, self.cell_size):
            pygame.draw.line(self.screen, (30, 30, 30), (0, y), (self.screen_size, y))
        
        # 绘制食物
        food_rect = pygame.Rect(self.food[0] * self.cell_size, self.food[1] * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (255, 0, 0), food_rect)
        
        # 绘制每条蛇
        for snake_idx in range(self.num_snakes):
            if not self.alive[snake_idx]:
                continue
                
            snake = self.snakes[snake_idx]
            direction = self.directions[snake_idx]
            head_color = self.snake_colors[snake_idx % len(self.snake_colors)]
            body_color = tuple(int(c * 0.8) for c in head_color)  # 蛇身颜色稍暗
            
            for i, (x, y) in enumerate(snake):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                if i == 0:
                    # 蛇头
                    pygame.draw.rect(self.screen, head_color, rect)
                    # 绘制眼睛
                    eye_size = self.cell_size // 8
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
                    pygame.draw.rect(self.screen, (0, 0, 0), (*eye1, eye_size, eye_size))
                    pygame.draw.rect(self.screen, (0, 0, 0), (*eye2, eye_size, eye_size))
                else:
                    # 蛇身
                    pygame.draw.rect(self.screen, body_color, rect)
        
        # 更新显示
        if self.render_mode == 'human':
            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])
            
            # 处理退出事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
    
    def render(self):
        if self.render_mode == 'rgb_array':
            return self._get_observation()
        
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

# 注册环境
from gymnasium.envs.registration import register
register(
    id='SnakeEnv-v1',
    entry_point='environment.snake_env:SnakeEnv',
    max_episode_steps=1000
)
