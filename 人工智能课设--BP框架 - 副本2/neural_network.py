# -*- coding:utf-8 -*-
# -*- coding:utf-8 -*-
import numpy as np
import math
import random


class NeuralNetwork:
    """三层BP神经网络"""

    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重矩阵
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.1
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.1

        # 初始化偏置向量
        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, output_size))

        # 添加经验回放缓存
        self.experience_buffer = []
        self.buffer_size = 1000
        self.batch_size = 32

        # 学习参数
        self.learning_rate = 0.1
        self.gamma = 0.9  # 折扣因子

    def sigmoid(self, x):
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Sigmoid导数"""
        return x * (1 - x)

    def forward(self, x):
        """前向传播"""
        # 输入层到隐藏层
        self.hidden = self.sigmoid(np.dot(x, self.weights1) + self.bias1)
        # 隐藏层到输出层
        output = self.sigmoid(np.dot(self.hidden, self.weights2) + self.bias2)
        return output

    def predict(self, robot_pos, dust_pos, boundary_dists):
        """根据机器人和灰尘位置预测转向"""
        # 输入数据归一化
        dx = dust_pos[0] - robot_pos[0]
        dy = dust_pos[1] - robot_pos[1]

        # 计算机器人当前朝向向量
        current_dx = -math.sin(robot_pos[2] / 180 * math.pi)
        current_dy = -math.cos(robot_pos[2] / 180 * math.pi)

        # 计算目标方向向量
        target_angle = math.atan2(dy, dx)
        target_dx = math.cos(target_angle)
        target_dy = math.sin(target_angle)

        # 计算当前方向与目标方向的点积（cosθ）
        dot_product = current_dx * target_dx + current_dy * target_dy

        # 计算当前方向与目标方向的叉积（sinθ）
        cross_product = current_dx * target_dy - current_dy * target_dx

        # 输入向量：点积、叉积和距离（归一化） 确保距离不会为负
        distance = max(0.001, math.sqrt(dx ** 2 + dy ** 2)) / 500.0

        # 归一化边界距离
        max_dim = max(self.screen_width, self.screen_height) if hasattr(self, 'screen_width') else 1000
        left_norm = boundary_dists[0] / max_dim
        right_norm = boundary_dists[1] / max_dim
        top_norm = boundary_dists[2] / max_dim
        bottom_norm = boundary_dists[3] / max_dim

        x = np.array([[dot_product, cross_product, distance, left_norm, right_norm]])

        # 前向传播
        output = self.forward(x)

        # 解析输出 - 更智能的决策逻辑
        turn_strength = output[0][0] * 2 - 1  # 将[0,1]映射到[-1,1]

        # 根据转向强度调整角度
        return turn_strength, x  # 返回状态用于训练

    def add_experience(self, state, action, reward, next_state):
        """添加经验到回放缓存"""
        self.experience_buffer.append((state, action, reward, next_state))

        # 保持缓存大小
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer.pop(0)

    def calculate_reward(self, state, next_state, dust_collected, boundary_close):
        """计算奖励值"""
        # 从状态中提取特征
        dot_product, cross_product, distance, left_dist, right_dist = state[0]
        next_dot, next_cross, next_distance, next_left_dist, next_right_dist = next_state[0]

        # 基本奖励：距离减少
        reward = (distance - next_distance) * 10

        # 收集灰尘的奖励
        if dust_collected:
            reward += 100

        # 边界接近惩罚 (新增)
        if boundary_close:
            reward -= 20

        # 边界距离增加的奖励 (新增)
        boundary_improvement = (min(left_dist, right_dist) - min(next_left_dist, next_right_dist))
        reward += boundary_improvement * 5

        # 鼓励探索中央区域（使用左右边界距离估算中心距离）
        center_x = (left_dist + right_dist) / 2
        # 使用左右边界距离的差值作为中心偏离程度的代理
        center_deviation = abs(left_dist - right_dist)
        # 中心偏离减少则奖励
        next_center_deviation = abs(next_left_dist - next_right_dist)

        if next_center_deviation < center_deviation:
            reward += 5  # 向中心移动奖励

        return reward

    def train(self):
        """使用经验回放训练神经网络"""
        if len(self.experience_buffer) < self.batch_size:
            return

        # 随机采样一批经验
        batch = random.sample(self.experience_buffer, self.batch_size)

        for state, action, reward, next_state in batch:
            # 前向传播
            current_q = self.forward(state)
            next_q = self.forward(next_state)

            # 计算目标Q值
            target = current_q.copy()
            target[0][0] = reward + self.gamma * np.max(next_q)

            # 反向传播
            # 输出层误差
            output_error = target - current_q
            output_delta = output_error * self.sigmoid_derivative(current_q)

            # 隐藏层误差
            hidden_error = output_delta.dot(self.weights2.T)
            hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden)

            # 更新权重
            self.weights2 += self.hidden.T.dot(output_delta) * self.learning_rate
            self.bias2 += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate

            self.weights1 += state.T.dot(hidden_delta) * self.learning_rate
            self.bias1 += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

    def periodic_training(self):
        """定期训练神经网络"""
        self.train()
