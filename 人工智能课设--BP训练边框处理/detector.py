# detector.py
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
        if not dusts:  # 如果灰尘组为空，直接返回None
            return None
        min_distance = float('inf')
        nearest_dust = None

        for dust in dusts:
            # 计算灰尘与机器人的距离
            dx = dust.rect.centerx - self.robot.rect.centerx
            dy = dust.rect.centery - self.robot.rect.centery
            distance = math.sqrt(dx ** 2 + dy ** 2)

            # 增强边界检测：对靠近边界的灰尘放宽检测条件
            boundary_margin = 30  # 边界额外检测范围

            # 检查灰尘是否在边界附近
            on_boundary = (
                    dust.rect.left <= boundary_margin or
                    dust.rect.right >= self.screen.get_width() - boundary_margin or
                    dust.rect.top <= boundary_margin or
                    dust.rect.bottom >= self.screen.get_height() - boundary_margin
            )

            # 如果在检测范围内或靠近边界，则考虑
            if distance < self.radius + dust.radius / 2 or on_boundary:
                # 只返回最近的灰尘
                if distance < min_distance:
                    min_distance = distance
                    nearest_dust = dust
        return nearest_dust

    def blitme(self):
        """绘制触觉器"""
        self.screen.blit(self.image, self.rect)