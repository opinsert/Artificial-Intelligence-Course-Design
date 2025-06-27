# neural_network.py
import numpy as np
import math
import random
import os
import json
import settings

ai_settings = settings.Settings()


class NeuralNetwork:
    """三层BP神经网络，支持经验回放"""

    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重矩阵
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01

        # 添加动量参数
        self.momentum = 0.9
        self.delta_weights1 = np.zeros_like(self.weights1)
        self.delta_weights2 = np.zeros_like(self.weights2)

        # 经验回放缓冲区
        self.buffer_size = 5000
        self.batch_size = 64
        self.replay_buffer = []  # 缓冲区初始化

        # 初始化偏置向量
        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, output_size))

        # 学习率
        self.learning_rate = 0.1
        self.min_learning_rate = 0.01

        # 添加模型标识
        self.model_id = "sweeping_robot_nn"

    def sigmoid(self, x):
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Sigmoid导数"""
        return x * (1 - x)

    def forward(self, X):
        """前向传播"""
        # 输入层到隐藏层
        self.hidden = self.sigmoid(np.dot(X, self.weights1) + self.bias1)
        # 隐藏层到输出层
        output = self.sigmoid(np.dot(self.hidden, self.weights2) + self.bias2)
        return output

    def add_experience(self, state, action):
        """添加经验到回放缓冲区"""
        if len(self.replay_buffer) >= self.buffer_size:
            self.replay_buffer.pop(0)  # 移除最旧的经验
        self.replay_buffer.append((state, action))

    def replay(self):
        """从经验回放中学习"""
        if len(self.replay_buffer) < self.batch_size:
            return  # 缓冲区中样本不足

        # 动态调整学习率
        if len(self.replay_buffer) > self.buffer_size * 0.8:
            self.learning_rate = max(self.min_learning_rate, self.learning_rate * 0.99)

        # 随机抽取一批经验
        batch = random.sample(self.replay_buffer, self.batch_size)
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])

        # 前向传播
        hidden = self.sigmoid(np.dot(states, self.weights1) + self.bias1)
        outputs = self.sigmoid(np.dot(hidden, self.weights2) + self.bias2)

        # 计算误差
        errors = actions - outputs

        # 反向传播
        delta_output = errors * self.sigmoid_derivative(outputs)

        errors_hidden = delta_output.dot(self.weights2.T)
        delta_hidden = errors_hidden * self.sigmoid_derivative(hidden)

        # 使用动量更新权重
        self.delta_weights1 = self.momentum * self.delta_weights1 + (1 - self.momentum) * states.T.dot(delta_hidden)
        self.delta_weights2 = self.momentum * self.delta_weights2 + (1 - self.momentum) * hidden.T.dot(delta_output)

        self.weights1 += self.delta_weights1 * self.learning_rate
        self.weights2 += self.delta_weights2 * self.learning_rate

    def get_state_vector(self, robot_pos, dust_pos, exploration_map, grid_x, grid_y):
        """获取增强的状态向量"""
        dx = dust_pos[0] - robot_pos[0]
        dy = dust_pos[1] - robot_pos[1]

        # 距离和方向
        distance = math.sqrt(dx ** 2 + dy ** 2) / 500.0

        # 相对角度
        robot_angle_rad = robot_pos[2] * math.pi / 180.0
        target_angle = math.atan2(dy, dx)
        angle_diff = (target_angle - robot_angle_rad + math.pi) % (2 * math.pi) - math.pi

        # 边界距离
        screen_width = ai_settings.screen_width  # 屏幕宽度
        screen_height = ai_settings.screen_height  # 屏幕高度
        left_dist = robot_pos[0] / screen_width
        right_dist = (screen_width - robot_pos[0]) / screen_width
        top_dist = robot_pos[1] / screen_height
        bottom_dist = (screen_height - robot_pos[1]) / screen_height

        # 探索度
        explore_value = 0
        if exploration_map and 0 <= grid_y < len(exploration_map) and 0 <= grid_x < len(exploration_map[0]):
            explore_value = min(exploration_map[grid_y][grid_x], 10) / 10.0

        return [
            math.cos(angle_diff),
            math.sin(angle_diff),
            distance,
            left_dist,
            right_dist,
            top_dist,
            bottom_dist,
            explore_value
        ]

    def save_model(self, directory="models"):
        """保存模型到指定目录"""
        # 创建目录
        if not os.path.exists(directory):
            os.makedirs(directory)

        # 构建模型路径
        model_path = os.path.join(directory, f"{self.model_id}.npz")

        # 保存权重和参数
        np.savez_compressed(
            model_path,
            weights1=self.weights1,
            weights2=self.weights2,
            bias1=self.bias1,
            bias2=self.bias2,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate
        )

        print(f"模型已保存到 {model_path}")

    def load_model(self, directory="models"):
        """从指定目录加载模型"""
        model_path = os.path.join(directory, f"{self.model_id}.npz")

        if os.path.exists(model_path):
            # 加载模型数据
            data = np.load(model_path)

            # 恢复权重和参数
            self.weights1 = data['weights1']
            self.weights2 = data['weights2']
            self.bias1 = data['bias1']
            self.bias2 = data['bias2']
            self.buffer_size = int(data['buffer_size'])
            self.batch_size = int(data['batch_size'])
            self.learning_rate = float(data['learning_rate'])

            print(f"已从 {model_path} 加载模型")
            return True
        else:
            print(f"找不到模型文件 {model_path}")
            return False
