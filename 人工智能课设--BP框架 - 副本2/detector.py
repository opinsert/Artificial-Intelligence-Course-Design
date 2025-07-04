# -*- coding:utf-8 -*-
import pygame
from pygame.sprite import Sprite
import math


class Detector(Sprite):
    """触觉器类，用于检测周围灰尘"""

    def __init__(self, screen, robot):
        super().__init__()
        self.screen = screen
        self.robot = robot

        # 增大检测半径并加粗边框
        self.radius = 120  # 增大检测半径
        size = self.radius * 2

        # 创建触觉器图像
        self.image = pygame.Surface((size, size), pygame.SRCALPHA)

        # 绘制更明显的圆形：外圈白色粗边框，内圈半透明填充
        pygame.draw.circle(self.image, (200, 200, 255, 80), (self.radius, self.radius), self.radius)
        pygame.draw.circle(self.image, (255, 255, 255, 255), (self.radius, self.radius), self.radius, 3)  # 加粗边框

        self.rect = self.image.get_rect()
        # 以机器人为中心
        self.rect.center = robot.rect.center

    def update(self):
        """更新触觉器位置，跟随机器人"""
        self.rect.center = self.robot.rect.center

    def detect_dust(self, dusts):
        """检测范围内的灰尘，返回最近的灰尘位置"""
        min_distance = float('inf')
        nearest_dust = None

        for dust in dusts:
            # 计算灰尘与机器人的距离
            dx = dust.rect.centerx - self.robot.rect.centerx
            dy = dust.rect.centery - self.robot.rect.centery
            distance = math.sqrt(dx ** 2 + dy ** 2)

            if distance < self.radius+40:
                # 返回所有在范围内的灰尘，而不仅仅是最近的
                return dust
                # 如果需要最近的灰尘，取消下面注释
                # if distance < min_distance:
                #     min_distance = distance
                #     nearest_dust = dust

        return nearest_dust

    def blitme(self):
        """绘制触觉器"""
        self.screen.blit(self.image, self.rect)