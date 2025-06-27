# -*- coding:utf-8 -*-
import pygame
from pygame.sprite import Sprite
import math
from neural_network import NeuralNetwork
import random
import time
import settings
import numpy as np

ai_settings1 = settings.Settings()


class Robot(Sprite):
    def __init__(self, screen):
        # initialize robot and its location
        self.screen = screen
        # load image and get rectangle
        self.image = pygame.image.load('images/robot.png').convert_alpha()
        self.rect = self.image.get_rect()
        self.screen_rect = screen.get_rect()
        # put sweeper on the center of window
        self.rect.center = self.screen_rect.center
        # 初始角度
        self.angle = 0
        self.moving_speed = [1, 1]
        self.moving_pos = [None, None]
        # 使用浮点数存储精确位置
        self.x = float(self.rect.centerx)
        self.y = float(self.rect.centery)

        self.moving_right = False
        self.moving_left = False

        # 添加神经网络
        self.nn = NeuralNetwork(8, 16, 1)  # 输入层3个节点，隐藏层5个节点，输出层1个节点

        # 添加自动控制标志
        self.automatic_mode = False

        # 边界反弹计数器
        self.boundary_bounce_count = 0
        self.boundary_bounce_threshold = 5  # 连续反弹5次后触发随机转向

        # 添加平滑转向参数
        self.target_angle = self.angle  # 目标角度
        self.turn_speed = 1  # 每秒转向速度（度/帧）
        self.dust_tracking = False  # 是否正在追踪灰尘

        # 添加区域探索相关属性
        self.regions = {}  # 存储区域访问状态的字典
        self.current_region = None  # 当前所在区域
        self.last_dust_time = time.time()  # 上次处理灰尘的时间
        self.region_change_timer = 0  # 区域切换计时器
        self.region_width = self.screen_rect.width // 3  # 三列
        self.region_height = self.screen_rect.height // 2  # 两行
        self.region_entry_time = time.time()  # 记录进入当前区域的时间
        self.region_stay_duration = 5.0  # 每个区域停留5秒

        self.idle_timer_started = False  # 空闲计时器是否已启动
        self.idle_turn_done = False  # 空闲转向是否已完成+

        # 添加边框检查相关属性
        self.border_check_interval = 30  # 每30秒检查一次
        self.last_border_check = time.time()
        self.border_points = []  # 存储边框检查点
        self.current_border_point = 0  # 当前目标点索引

        # 自动训练计数器
        self.train_counter = 0
        # 模型保存计数器
        self.train_model_counter = 0

        self.exploration_map = []  # 探索地图
        self.map_resolution = 20  # 地图分辨率（像素/格）
        self.last_known_dust = None  # 最后已知的灰尘位置
        self.high_traffic_areas = set()  # 记录访问超过4次的区域
        self.escape_target = None  # 逃离目标位置
        self.escape_distance = 240  # 两个检测器的距离（240像素）
        self.escape_mode = False
        self.exploration_decay = 0.99  # 探索度衰减
        self.last_explore_update = 0
        self.grid_visits = {}

        # 初始化神经网络后尝试加载模型
        self.nn = NeuralNetwork(8, 16, 1)
        self.nn.load_model()  # 尝试加载已有模型

        # 添加旋转后的图像缓存
        self.rotated_image = self.image

    def blitme(self):
        # 绘制旋转后的图像
        self.screen.blit(self.rotated_image, self.rect)

    def handle_boundary_bounce(self):
        """简化的边界反弹处理"""
        screen_rect = self.screen_rect
        boundary_bounced = False
        half_width = self.rect.width // 2
        half_height = self.rect.height // 2

        # 左边界
        if self.x - half_width <= 0:
            if self.angle == 0:
                self.angle = 180  # 0度时设为180度
            else:
                self.angle = 180 - self.angle
                boundary_bounced = True
                self.x = half_width + 10  # 防止卡在边界

        # 右边界
        if self.x + half_width >= screen_rect.width:
            self.angle = 180 - self.angle
            boundary_bounced = True
            self.x = screen_rect.width - half_width - 10

        # 上边界
        if self.y - half_height <= 0:
            self.angle = - self.angle
            boundary_bounced = True
            self.y = half_height + 10

        # 下边界
        if self.y + half_height >= screen_rect.height:
            self.angle = - self.angle
            boundary_bounced = True
            self.y = screen_rect.height - half_height - 10

        # 角度规范化
        if boundary_bounced:
            if self.angle < -180:
                self.angle += 360
            elif self.angle > 180:
                self.angle -= 360
        return boundary_bounced

    def start_border_check(self):
        """开始边框检查（绕屏幕边缘一圈）"""
        if self.border_points:
            return  # 如果已在检查中，则跳过

        screen_width = self.screen_rect.width
        screen_height = self.screen_rect.height
        margin = 50  # 离边界的距离

        # 顺时针创建边框点：左上 -> 右上 -> 右下 -> 左下
        self.border_points = [
            (margin, margin),  # 左上
            (screen_width - margin, margin),  # 右上
            (screen_width - margin, screen_height - margin),  # 右下
            (margin, screen_height - margin)  # 左下
        ]
        self.current_border_point = 0
        self.escape_target = self.border_points[0]
        self.escape_mode = True
        print("开始边框检查...")

    def update(self, robot, ai_settings):
        # 如果是第一次更新，初始化位置
        if self.moving_pos[0] is None:
            self.moving_pos[0] = self.x
            self.moving_pos[1] = self.y

        # 更新探索地图
        self.update_exploration_map()

        # 保存旧位置用于检测边界反弹
        old_pos = (self.moving_pos[0], self.moving_pos[1])

        # 保存当前位置
        current_x = self.x
        current_y = self.y

        # 计算移动增量
        angle_rad = math.radians(self.angle)
        dx = math.cos(angle_rad) * self.moving_speed[0]
        dy = -math.sin(angle_rad) * self.moving_speed[1]

        # 更新位置（浮点数）
        self.x = current_x + dx
        self.y = current_y + dy

        # 更新rect位置（取整）
        self.rect.centerx = int(self.x)
        self.rect.centery = int(self.y)

        # 更新移动位置记录
        self.moving_pos[0] = self.x
        self.moving_pos[1] = self.y

        # 检测边界反弹并处理
        boundary_bounced = self.handle_boundary_bounce()

        # 更新边界反弹计数器
        if boundary_bounced:
            self.boundary_bounce_count += 1
        else:
            self.boundary_bounce_count = 0

        # 如果连续反弹次数达到阈值，则随机转向
        if self.boundary_bounce_count >= self.boundary_bounce_threshold:
            self.random_turn()
            self.boundary_bounce_count = 0  # 重置计数器

            # 更新区域信息
            self.update_regions()

        # 更新区域计时器
        current_time = time.time()
        if current_time - self.last_dust_time > 5:  # 5秒没有处理灰尘
            self.region_change_timer += 1

            # 每10帧尝试切换到新区域（避免过于频繁）
            if self.region_change_timer >= 10:
                self.move_to_new_region()

        # 重构转向处理开始
        turn_amount = 0

        # 平滑转向逻辑
        if self.dust_tracking:
            # 计算角度差（考虑360度环绕）
            angle_diff = (self.target_angle - self.angle + 180) % 360 - 180

            # 计算转向方向
            turn_direction = 1 if angle_diff > 0 else -1

            # 计算转向幅度（不超过最大转向速度）
            turn_amount = min(abs(angle_diff), self.turn_speed) * turn_direction

            # 如果角度差很小，停止追踪
            if abs(angle_diff) < self.turn_speed:
                self.dust_tracking = False
        elif not self.automatic_mode:
            # 普通转向处理
            if self.moving_right:
                self.angle -= 1
            if self.moving_left:
                self.angle += 1
        # 应用转向
        self.angle += turn_amount

        # 角度规范化
        if self.angle < -180:
            self.angle += 360
        if self.angle > 180:
            self.angle -= 360

        # 计算移动增量（使用最新的角度）
        dx = -math.sin(self.angle / 180 * math.pi) * self.moving_speed[0]
        dy = -math.cos(self.angle / 180 * math.pi) * self.moving_speed[1]

        # 更新位置（基于当前位置）
        self.rect.centerx = current_x + dx
        self.rect.centery = current_y + dy

        # 更新移动位置记录
        self.moving_pos[0] = self.rect.centerx
        self.moving_pos[1] = self.rect.centery

        # 旋转图片并缓存
        self.rotated_image = pygame.transform.rotate(self.image, self.angle)
        # 校正旋转图片的中心点
        self.rect = self.rotated_image.get_rect(center=self.rect.center)

        # 在自动模式下优先处理灰尘检测
        if self.automatic_mode and hasattr(self, 'detector') and hasattr(self, 'dusts') and self.dusts:
            # 检测整个屏幕的灰尘，而不仅仅是当前区域
            nearest_dust = self.detector.detect_dust(self.dusts)
            if nearest_dust:
                # 转向并移向灰尘
                self.turn_towards_dust((nearest_dust.rect.centerx, nearest_dust.rect.centery))
                self.move_towards_dust((nearest_dust.rect.centerx, nearest_dust.rect.centery))
                # 更新神经网络经验
                self.update_neural_network_experience(nearest_dust)
                return  # 检测到灰尘后立即返回，不再执行后续逻辑

            # # 更新当前区域信息
            # self.update_regions()
            #
            # # 获取当前区域内的灰尘
            # current_region_dusts = [dust for dust in self.dusts if self.is_dust_in_current_region(dust)]
            #
            # if current_region_dusts:
            #     nearest_dust = self.detector.detect_dust(current_region_dusts)
            #     if nearest_dust:
            #         # 转向并移向灰尘
            #         self.turn_towards_dust((nearest_dust.rect.centerx, nearest_dust.rect.centery))
            #         self.move_towards_dust((nearest_dust.rect.centerx, nearest_dust.rect.centery))
            #         # 更新神经网络经验
            #         self.update_neural_network_experience(nearest_dust)

        # 每1000帧自动保存一次模型
        if self.train_model_counter % 1000 == 0 and self.train_model_counter > 0:
            self.nn.save_model()

        # 手动模式数据收集
        if not self.automatic_mode and hasattr(self, 'detector') and hasattr(self, 'dusts') and self.dusts:
            nearest_dust = self.detector.detect_dust(self.dusts)
            if nearest_dust:
                dust_pos = (nearest_dust.rect.centerx, nearest_dust.rect.centery)
                robot_pos = (self.rect.centerx, self.rect.centery, self.angle)
                grid_x = int(self.rect.centerx / self.map_resolution)
                grid_y = int(self.rect.centery / self.map_resolution)

                # 获取状态向量
                state = self.nn.get_state_vector(
                    robot_pos,
                    dust_pos,
                    self.exploration_map,
                    grid_x,
                    grid_y,
                )

                # 记录玩家操作（转向动作）
                # 左转为正，右转为负
                player_action = 0
                if self.moving_left:
                    player_action = 1.0
                elif self.moving_right:
                    player_action = -1.0

                # 存入经验回放
                self.nn.add_experience(state, [player_action])

        if self.automatic_mode:
            # 没有灰尘时的探索行为
            self.automatic_control(ai_settings)
            current_time = time.time()
            # 每10帧训练一次神经网络
            self.train_counter += 1
            if self.train_counter >= 10:
                self.nn.replay()
                self.train_counter = 0
            # 检查是否在当前区域停留超过设定时间（3秒）
            if current_time - self.region_entry_time > self.region_stay_duration:
                self.move_to_new_region()
                # 重置进入时间
                self.region_entry_time = current_time

            # 每10帧训练一次神经网络
            self.train_counter += 1
            if self.train_counter >= 10:
                self.nn.replay()
                self.train_counter = 0

        # 每30秒触发一次边框检查
        current_time = time.time()
        if (current_time - self.last_border_check > self.border_check_interval and
                not self.escape_mode and
                not self.dust_tracking and
                not self.border_points):
            self.start_border_check()
            self.last_border_check = current_time

        # 处理边框检查逻辑
        if self.border_points and self.escape_mode:
            # 检查是否到达当前目标点
            target_x, target_y = self.border_points[self.current_border_point]
            dx = target_x - self.rect.centerx
            dy = target_y - self.rect.centery
            distance = math.sqrt(dx ** 2 + dy ** 2)

            if distance < 20:  # 到达目标点
                # 移动到下一个点
                self.current_border_point = (self.current_border_point + 1) % len(self.border_points)

                if self.current_border_point == 0:  # 完成一圈
                    print("边框检查完成")
                    self.border_points = []
                    self.escape_mode = False
                    self.escape_target = None
                else:  # 继续下一个点
                    self.escape_target = self.border_points[self.current_border_point]

            # 转向并移向下一个目标点
            self.turn_towards_dust(self.escape_target)
            self.move_towards_dust(self.escape_target)

            return  # 边框检查期间不执行其他行为

    def is_dust_in_current_region(self, dust):
        """检查灰尘是否在当前区域内"""
        if self.current_region is None:
            return False

        region_x, region_y = self.current_region
        left = region_x * self.region_width
        right = left + self.region_width
        top = region_y * self.region_height
        bottom = top + self.region_height

        dust_x, dust_y = dust.rect.centerx, dust.rect.centery
        return left <= dust_x < right and top <= dust_y < bottom

    # 移动方法
    def move_towards_dust(self, dust_pos):
        """向灰尘方向移动"""
        # 计算灰尘方向向量
        dx = dust_pos[0] - self.rect.centerx
        dy = dust_pos[1] - self.rect.centery

        # 计算移动距离（限制最大速度）
        distance = math.sqrt(dx ** 2 + dy ** 2)
        if distance > 0:
            # 限制移动速度
            max_speed = 1.0  # 最大速度

            # 计算移动向量
            move_x = (dx / distance) * max_speed
            move_y = (dy / distance) * max_speed

            # 更新位置
            self.rect.centerx += move_x
            self.rect.centery += move_y

            # 更新浮点位置，确保位置一致
            self.x = self.rect.centerx
            self.y = self.rect.centery

    def reset_idle_timer(self):
        """重置空闲计时器（当吸收灰尘时调用）"""
        self.last_dust_time = time.time()
        self.idle_timer_started = True
        self.idle_turn_done = False

    def handle_collision(self):
        """处理灰尘碰撞"""
        self.last_dust_time = time.time()  # 重置计时器
        self.region_change_timer = 0  # 重置区域切换计时器

    def automatic_exploration(self, ai_settings):
        """没有灰尘时的探索行为"""
        # 获取探索度最低的方向
        target_angle = self.get_least_explored_direction()

        # 计算角度差并平滑转向
        angle_diff = (target_angle - self.angle + 180) % 360 - 180
        turn_amount = min(abs(angle_diff), self.turn_speed) * (1 if angle_diff > 0 else -1)
        self.angle += turn_amount

        # 角度规范化
        self.angle %= 360
        if self.angle > 180:
            self.angle -= 360

    def update_neural_network_experience(self, dust):
        """更新神经网络经验"""
        dust_pos = (dust.rect.centerx, dust.rect.centery)
        robot_pos = (self.rect.centerx, self.rect.centery, self.angle)
        grid_x = int(self.rect.centerx / self.map_resolution)
        grid_y = int(self.rect.centery / self.map_resolution)

        state = self.nn.get_state_vector(
            robot_pos,
            dust_pos,
            self.exploration_map,
            grid_x,
            grid_y,
        )

        # 计算最优动作（直接面向灰尘）
        dx = dust_pos[0] - self.rect.centerx
        dy = dust_pos[1] - self.rect.centery
        target_angle = math.degrees(math.atan2(-dy, dx)) - 90
        angle_diff = (target_angle - self.angle + 180) % 360 - 180
        optimal_action = angle_diff / 180.0

        # 存入经验回放
        self.nn.add_experience(state, [optimal_action])

    def automatic_control(self, ai_settings):
        """自动控制逻辑"""
        # 优先处理逃离模式
        if self.escape_mode:
            # 检查是否到达目标
            dx = self.escape_target[0] - self.rect.centerx
            dy = self.escape_target[1] - self.rect.centery
            distance = math.sqrt(dx ** 2 + dy ** 2)

            if distance < 50:  # 到达目标附近
                self.escape_mode = False
                self.escape_target = None
            else:
                # 转向并移向目标
                self.turn_towards_dust(self.escape_target)
                self.move_towards_dust(self.escape_target)
                return

        # 如果没有灰尘，进行探索
        if not self.dusts or len(self.dusts) == 0:
            self.automatic_exploration(ai_settings)
            return
        # 获取当前区域内的灰尘（修改部分）
        current_region_dusts = [dust for dust in self.dusts if self.is_dust_in_current_region(dust)]

        if current_region_dusts:
            # 只处理当前区域内的灰尘
            nearest_dust = self.detector.detect_dust(current_region_dusts)
            if nearest_dust:
                # 转向并移向灰尘
                self.turn_towards_dust((nearest_dust.rect.centerx, nearest_dust.rect.centery))
                self.move_towards_dust((nearest_dust.rect.centerx, nearest_dust.rect.centery))
                # 更新神经网络经验
                self.update_neural_network_experience(nearest_dust)
                return
        else:
            # 当前区域内没有灰尘，随机选择新区域进行探索
            if not self.escape_mode:
                self.move_to_new_region()

    def init_exploration_map(self, screen_width, screen_height):
        """初始化探索地图"""
        cols = screen_width // self.map_resolution
        rows = screen_height // self.map_resolution
        self.exploration_map = [[0] * cols for _ in range(rows)]

    def update_exploration_map(self):
        """改进探索地图更新"""
        if not self.exploration_map:
            return

        current_time = time.time()
        # 每0.5秒更新一次探索地图
        if current_time - self.last_explore_update < 0.5:
            return

        self.last_explore_update = current_time

        # 衰减所有区域的探索度
        for y in range(len(self.exploration_map)):
            for x in range(len(self.exploration_map[0])):
                self.exploration_map[y][x] *= self.exploration_decay

        # 更新当前位置
        x = int(self.rect.centerx / self.map_resolution)
        y = int(self.rect.centery / self.map_resolution)

        if 0 <= x < len(self.exploration_map[0]) and 0 <= y < len(self.exploration_map):
            # 增加当前区域探索度
            self.exploration_map[y][x] += 1.0

            # 增加周围区域探索度
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < len(self.exploration_map[0]) and 0 <= ny < len(self.exploration_map):
                        distance_factor = 1.0 / (abs(dx) + abs(dy) + 1)
                        self.exploration_map[ny][nx] += 0.3 * distance_factor

    def get_least_explored_direction(self):
        """获取探索度最低的方向"""
        if not self.exploration_map:
            return random.randint(0, 360)

        # 获取当前位置的网格坐标
        x = int(self.rect.centerx // self.map_resolution)
        y = int(self.rect.centery // self.map_resolution)

        # 计算当前网格访问次数
        current_grid = (x, y)
        self.grid_visits[current_grid] = self.grid_visits.get(current_grid, 0) + 1

        # 如果当前网格访问超过4次，触发逃离模式
        if self.grid_visits[current_grid] >= 4 and not self.escape_mode:
            self.create_escape_target()
            return self.target_angle

        # 计算八个方向的探索度（增加对角线方向）
        directions = {
            0: 0,  # 北
            45: 0,  # 东北
            90: 0,  # 东
            135: 0,  # 东南
            180: 0,  # 南
            225: 0,  # 西南
            270: 0,  # 西
            315: 0  # 西北
        }

        # 检查每个方向（深度增加到8格）
        for angle in directions:
            # 计算方向向量
            rad_angle = math.radians(angle)
            dx = math.cos(rad_angle)
            dy = math.sin(rad_angle)

            # 沿方向检查探索度
            for distance in range(1, 9):  # 1到8格距离
                nx = int(x + dx * distance)
                ny = int(y + dy * distance)

                # 检查是否超出边界
                if nx < 0 or nx >= len(self.exploration_map[0]) or ny < 0 or ny >= len(self.exploration_map):
                    directions[angle] += 1  # 边界外视为未探索
                    continue

                # 累加探索度
                directions[angle] += (10 - min(self.exploration_map[ny][nx], 10))

        # 选择探索度最高的方向
        best_angle = max(directions, key=directions.get)

        return best_angle

    def create_escape_target(self):
        """创建逃离高访问区域的目标位置"""
        # 将屏幕分成6个区域
        cols = 3
        rows = 2
        col_width = self.screen_rect.width // cols
        row_height = self.screen_rect.height // rows

        # 选择未访问或访问次数最少的区域
        min_visits = float('inf')
        target_region = None

        for row in range(rows):
            for col in range(cols):
                region = (col, row)
                visits = self.grid_visits.get(region, 0)

                # 选择访问次数最少的区域
                if visits < min_visits:
                    min_visits = visits
                    target_region = region

        # 在目标区域内随机选择一个点
        if target_region:
            col, row = target_region
            left = col * col_width
            top = row * row_height
            right = left + col_width
            bottom = top + row_height

            # 确保目标在屏幕内
            target_x = random.randint(max(50, left), min(right - 50, ai_settings1.screen_width - 50))
            target_y = random.randint(max(50, top), min(bottom - 50, ai_settings1.screen_height - 50))

            self.escape_target = (target_x, target_y)
            self.escape_mode = True

            # 计算目标角度
            dx = target_x - self.rect.centerx
            dy = target_y - self.rect.centery
            self.target_angle = math.degrees(math.atan2(-dy, dx)) - 90

            # 角度规范化
            if self.target_angle < -180:
                self.target_angle += 360
            if self.target_angle > 180:
                self.target_angle -= 360

            return self.target_angle

        return random.randint(0, 360)

    def update_regions(self):
        """更新区域信息"""
        # 计算当前所在区域
        region_x = int(self.x // self.region_width)
        region_y = int(self.y // self.region_height)
        region_key = (region_x, region_y)

        # 如果进入新区域
        if self.current_region != region_key:
            self.current_region = region_key
            self.region_entry_time = time.time()  # 重置区域进入时间

            # 标记区域为已访问
            if region_key not in self.regions:
                self.regions[region_key] = True
                print(f"进入新区域: {region_key}")

    def get_unexplored_region(self):
        """获取未探索的区域 - 基于整个屏幕的六等分"""
        # 计算总区域数
        total_regions_x = 3
        total_regions_y = 2

        # 找出所有未探索的区域
        unexplored = []
        for x in range(total_regions_x):
            for y in range(total_regions_y):
                if (x, y) not in self.regions:
                    unexplored.append((x, y))

        # 如果有未探索区域，随机返回一个
        if unexplored:
            return random.choice(unexplored)

        # 如果所有区域都已探索，随机返回一个区域
        return (random.randint(0, total_regions_x - 1),
                random.randint(0, total_regions_y - 1))

    def move_to_new_region(self):
        """移动到新的区域 - 基于整个屏幕的六等分"""
        # 获取未探索区域
        new_region = self.get_unexplored_region()

        # 计算目标位置（区域中心）
        target_x = (new_region[0] * self.region_width) + (self.region_width / 2)
        target_y = (new_region[1] * self.region_height) + (self.region_height / 2)

        # 确保目标在屏幕范围内 - 使用ai_settings中的参数
        target_x = max(50, min(target_x, ai_settings1.screen_width - 50))
        target_y = max(50, min(target_y, ai_settings1.screen_height - 50))

        # 设置目标位置
        self.escape_target = (target_x, target_y)
        self.escape_mode = True

        # 计算目标角度
        dx = target_x - self.rect.centerx
        dy = target_y - self.rect.centery
        self.target_angle = math.degrees(math.atan2(-dy, dx)) - 90

        # 角度规范化
        if self.target_angle < -180:
            self.target_angle += 360
        if self.target_angle > 180:
            self.target_angle -= 360

        print(f"移动到新区域: {new_region}, 目标位置: ({target_x}, {target_y})")

        # 重置计时器
        self.last_dust_time = time.time()

    def turn_towards_dust(self, dust_pos):
        """立即转向灰尘方向"""
        dx = dust_pos[0] - self.rect.centerx
        dy = dust_pos[1] - self.rect.centery

        # 计算灰尘相对于机器人的角度
        target_angle = math.degrees(math.atan2(-dy, dx)) - 90

        # 确保角度在-180到180范围内
        if target_angle < -180:
            target_angle += 360
        if target_angle > 180:
            target_angle -= 360

        # 直接设置机器人角度
        self.angle = target_angle

    def random_turn(self):
        """随机转向，避免边界反弹循环"""
        # 随机选择转向方向和幅度
        turn_direction = random.choice([-1, 1])
        turn_amount = random.randint(30, 90)  # 30-120度转向

        # 设置目标角度并开始追踪
        self.target_angle = (self.angle + turn_direction * turn_amount) % 360
        if self.target_angle > 180:
            self.target_angle -= 360
        self.dust_tracking = True

        # 边界检查
        if self.angle < -180:
            self.angle = 360 + self.angle
        if self.angle > 180:
            self.angle = self.angle - 360

        # 重置边界反弹计数器
        self.boundary_bounce_count = 0

    def set_dusts(self, dusts):
        """设置灰尘组"""
        self.dusts = dusts

    def set_detector(self, detector):
        """设置触觉器"""
        self.detector = detector

    def set_automatic_mode(self, mode):
        """设置自动控制模式"""
        self.automatic_mode = mode
