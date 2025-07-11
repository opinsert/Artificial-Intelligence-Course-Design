# -*- coding:utf-8 -*-
import pygame
from pygame.sprite import Sprite
import math
from neural_network import NeuralNetwork
import random
import time
from settings import Settings

ai_set = Settings()


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
        self.moving_pos = [self.rect.centerx, self.rect.centery]

        self.moving_right = False
        self.moving_left = False

        # 添加神经网络
        self.nn = NeuralNetwork(ai_set.nn_input_size, ai_set.nn_hidden_size, ai_set.nn_output_size)  # 输入层5个节点，隐藏层8个节点，输出层1个节点

        # 添加这两行设置屏幕尺寸
        self.nn.screen_width = self.screen_rect.width
        self.nn.screen_height = self.screen_rect.height

        # 添加自动控制标志
        self.automatic_mode = False

        # 边界处理参数
        self.boundary_threshold = 50  # 离边界50像素时触发避让
        self.last_boundary_time = 0  # 上次处理边界的时间

        # 添加平滑转向参数
        self.target_angle = self.angle  # 目标角度
        self.turn_speed = 2  # 每秒转向速度（度/帧）
        self.dust_tracking = False  # 是否正在追踪灰尘

        # 添加灰尘吸收计时器
        self.last_dust_time = 0  # 上次吸收灰尘的时间
        self.idle_timer_started = False  # 空闲计时器是否已启动
        self.idle_turn_done = False  # 空闲转向是否已完成

        # 添加训练计数器
        self.train_counter = 0
        self.train_interval = 10  # 每10帧训练一次

        # 添加状态跟踪
        self.last_state = None
        self.last_action = None

        # 添加预测时间跟踪
        self.last_prediction_time = time.time()  # 记录上次预测时间

        # 尝试加载之前的经验
        self.nn.load_experience('experience.pkl')

    def blitme(self):
        # buld the sweeper at the specific location
        self.screen.blit(self.image, self.rect)

    def update(self, new_robot):
        # 更新空闲计时器
        self.update_idle_timer()

        # 在自动模式下禁用边界物理反弹
        if not self.automatic_mode:
            # 普通模式下的边界处理
            if self.rect.top <= 0 and -90 < self.angle < 90:
                self.angle = 180 - self.angle
            # 下边界反弹的处理
            if self.rect.bottom >= self.screen_rect.height and (self.angle > 90 or self.angle < -90):
                self.angle = 180 - self.angle
            # 左边界反弹
            if self.rect.left <= 0 and (0 < self.angle <= 180):
                self.angle = -self.angle
            # 右边界反弹
            if self.rect.right >= self.screen_rect.width and (-180 < self.angle < 0):
                self.angle = -self.angle

        self.moving_pos[0] -= math.sin(self.angle / 180 * math.pi) * self.moving_speed[0]
        self.moving_pos[1] -= math.cos(self.angle / 180 * math.pi) * self.moving_speed[1]
        self.rect.centerx = self.moving_pos[0]
        self.rect.centery = self.moving_pos[1]

        # 确保机器人不会移出屏幕范围
        self.moving_pos[0] = max(self.rect.width / 2,
                                 min(self.screen_rect.width - self.rect.width / 2, self.moving_pos[0]))
        self.moving_pos[1] = max(self.rect.height / 2,
                                 min(self.screen_rect.height - self.rect.height / 2, self.moving_pos[1]))

        # 更新位置
        self.rect.centerx = self.moving_pos[0]
        self.rect.centery = self.moving_pos[1]

        # 平滑转向逻辑
        if self.dust_tracking:
            # 计算角度差（考虑360度环绕）
            angle_diff = (self.target_angle - self.angle + 180) % 360 - 180

            # 计算转向方向
            turn_direction = 1 if angle_diff > 0 else -1

            # 计算转向幅度（不超过最大转向速度）
            turn_amount = min(abs(angle_diff), self.turn_speed) * turn_direction

            # 应用转向
            self.angle += turn_amount

            # 如果角度差很小，停止追踪
            if abs(angle_diff) < self.turn_speed:
                self.dust_tracking = False
        else:
            # 普通转向处理
            if self.moving_right:
                self.angle -= 1
            if self.moving_left:
                self.angle += 1

        # 角度边界检查
        if self.angle < -180:
            self.angle = 360 + self.angle
        if self.angle > 180:
            self.angle = self.angle - 360

        # 旋转图片(注意：这里要搞一个新变量，存储旋转后的图片
        new_robot.image = pygame.transform.rotate(self.image, self.angle)
        # 校正旋转图片的中心点
        new_robot.rect = new_robot.image.get_rect(center=self.rect.center)

        if self.automatic_mode:
            self.automatic_control()

    def update_idle_timer(self):
        """更新空闲计时器并处理空闲转向"""
        current_time = time.time()
        # 如果空闲计时器已启动且超过5秒，且尚未执行转向
        if self.idle_timer_started and current_time - self.last_dust_time > 10 and not self.idle_turn_done:
            # 向右偏转60度
            self.angle -= 60
            self.idle_turn_done = True

            # 角度边界检查
            if self.angle < -180:
                self.angle = 360 + self.angle
            if self.angle > 180:
                self.angle = self.angle - 360

            # 开始平滑转向
            self.dust_tracking = True
            self.idle_turn_done = True

    def reset_idle_timer(self):
        """重置空闲计时器（当吸收灰尘时调用）"""
        self.last_dust_time = time.time()
        self.idle_timer_started = True
        self.idle_turn_done = False

    def automatic_control(self):
        """自动控制逻辑"""
        # 获取当前位置和角度
        robot_pos = (self.rect.centerx, self.rect.centery, self.angle)

        # 计算边界距离
        left_dist = self.rect.left
        right_dist = self.screen_rect.width - self.rect.right
        top_dist = self.rect.top
        bottom_dist = self.screen_rect.height - self.rect.bottom
        min_dist = min(left_dist, right_dist, top_dist, bottom_dist)

        # 边界接近检测
        boundary_close = min_dist < self.boundary_threshold
        current_time = time.time()

        # 如果有灰尘，获取最近灰尘位置
        if hasattr(self, 'detector') and hasattr(self, 'dusts'):
            nearest_dust = self.detector.detect_dust(self.dusts)
            if nearest_dust:
                dust_pos = (nearest_dust.rect.centerx, nearest_dust.rect.centery)

                # 使用神经网络预测转向
                turn_strength, state = self.nn.predict(robot_pos, dust_pos,
                                                       (left_dist, right_dist, top_dist, bottom_dist))

                # 保存当前状态和动作
                current_state = state

                # 应用转向
                self.angle -= turn_strength * 2

                # 如果有上一次的状态，添加到经验回放
                if self.last_state is not None:
                    # 计算奖励（假设没有立即收集到灰尘）
                    reward = self.nn.calculate_reward(self.last_state, current_state, False, boundary_close)
                    self.nn.add_experience(self.last_state, self.last_action, reward, current_state)

                # 更新状态跟踪
                self.last_state = current_state
                self.last_action = turn_strength

                # 设置目标角度并开始追踪
                self.target_angle = math.degrees(math.atan2(
                    -(dust_pos[1] - self.rect.centery),
                    dust_pos[0] - self.rect.centerx
                )) - 90

                # 转换为-180到180范围
                if self.target_angle < -180:
                    self.target_angle += 360
                if self.target_angle > 180:
                    self.target_angle -= 360

                self.dust_tracking = True
                self.last_prediction_time = current_time
            else:
                # 如果没有检测到灰尘，且距离上次预测超过一定时间
                current_time = time.time()
                if current_time - self.last_prediction_time > 1.0:  # 每1秒预测一次
                    # 设置一个在机器人前方的目标位置
                    look_ahead_distance = 200  # 向前看200像素
                    target_x = self.rect.centerx - look_ahead_distance * math.sin(self.angle / 180 * math.pi)
                    target_y = self.rect.centery - look_ahead_distance * math.cos(self.angle / 180 * math.pi)

                    # 确保目标位置在屏幕范围内
                    target_x = max(50, min(self.screen_rect.width - 50, target_x))
                    target_y = max(50, min(self.screen_rect.height - 50, target_y))

                    # 使用神经网络预测转向
                    turn_strength, state = self.nn.predict(robot_pos, (target_x, target_y),
                                                           (left_dist, right_dist, top_dist, bottom_dist))

                    # 保存当前状态和动作
                    current_state = state

                    # 应用转向
                    self.angle -= turn_strength * 2

                    # 设置目标角度并开始追踪
                    self.target_angle = math.degrees(math.atan2(
                        -(target_y - self.rect.centery),
                        target_x - self.rect.centerx
                    )) - 90

                    # 转换为-180到180范围
                    if self.target_angle < -180:
                        self.target_angle += 360
                    if self.target_angle > 180:
                        self.target_angle -= 360

                    self.dust_tracking = True
                    self.last_prediction_time = current_time  # 更新预测时间

        # 边界接近时的特殊处理 指向屏幕中心
        if boundary_close and current_time - self.last_boundary_time > 1.0:
            # 计算屏幕中心的方向
            center_x = self.screen_rect.centerx
            center_y = self.screen_rect.centery
            # 计算指向屏幕中心的向量
            dx = center_x - self.rect.centerx
            dy = center_y - self.rect.centery
            # 计算角度（注意：我们的坐标系中，0度向上，所以需要转换）
            target_angle = math.degrees(math.atan2(-dy, dx)) - 90

            # 转换为-180到180范围
            if target_angle < -180:
                target_angle += 360
            if target_angle > 180:
                target_angle -= 360
            self.target_angle = target_angle

            self.dust_tracking = True
            self.last_boundary_time = current_time

        # 如果持续接近边界，进行随机转向
        if boundary_close and min_dist < 20 and current_time - self.last_boundary_time > 5.0:
            self.random_turn()
            self.last_boundary_time = current_time

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

    def reset_after_dust_collected(self):
        """在收集灰尘后重置状态跟踪"""
        # 如果有上一次的状态，添加收集灰尘的奖励
        if self.last_state is not None:
            # 创建一个虚拟的"下一个状态"
            next_state = self.last_state.copy()
            next_state[0][2] *= 0.9  # 假设距离减少了10%

            # 计算奖励（收集到灰尘）
            reward = self.nn.calculate_reward(self.last_state, next_state, True, False)
            self.nn.add_experience(self.last_state, self.last_action, reward, next_state)

        # 重置状态跟踪
        self.last_state = None
        self.last_action = None

    def set_dusts(self, dusts):
        """设置灰尘组"""
        self.dusts = dusts

    def set_detector(self, detector):
        """设置触觉器"""
        self.detector = detector

    def set_automatic_mode(self, mode):
        """设置自动控制模式"""
        self.automatic_mode = mode
