# -*- coding:utf-8 -*-
import pygame
from pygame.sprite import Sprite


class Dust(Sprite):
    """A class to represent single dust in the room."""

    def __init__(self, ai_settings, screen):
        """Initialize the dust, and set its starting position."""
        super().__init__()
        self.screen = screen
        self.ai_settings = ai_settings

        # Load the dust image, and set its rect attribute.
        self.image = pygame.image.load('images/dust.png').convert_alpha()
        self.rect = self.image.get_rect()
        self.radius = self.rect.width

        # Start each new dust near the top left of the screen.
        self.rect.x = self.rect.width
        self.rect.y = self.rect.height

    def blitme(self):
        """Draw the dust at its current location."""
        self.screen.blit(self.image, self.rect)
