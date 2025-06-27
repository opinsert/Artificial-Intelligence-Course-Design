# -*- coding:utf-8 -*-
import sys
import pygame
import random
from dust import Dust


def check_keydown_events(event, robot):
    if event.key == pygame.K_RIGHT:
        # move right
        robot.moving_right = True
    elif event.key == pygame.K_LEFT:
        # move left
        robot.moving_left = True


def check_keyup_events(event, robot):
    if event.key == pygame.K_RIGHT:
        robot.moving_right = False
    elif event.key == pygame.K_LEFT:
        robot.moving_left = False


def check_events(robot):
    """Respond to keyboard events only (no quit handling)."""
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            check_keydown_events(event, robot)
        elif event.type == pygame.KEYUP:
            check_keyup_events(event, robot)


# def check_events(robot):
#     # respond to  keyboard and mouse item
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             sys.exit()
#         elif event.type == pygame.KEYDOWN:
#             check_keydown_events(event, robot)
#         elif event.type == pygame.KEYUP:
#             check_keyup_events(event, robot)

def check_robot_dust_collisions(robot, dusts):
    """Respond to robot-dust collisions."""
    # Remove any robot and dusts that have collided.
    pygame.sprite.spritecollide(robot, dusts, True, None)


def update_screen(ai_settings, screen, dusts, robot, detector=None, mode_text=None):

    # fill color
    screen.fill(ai_settings.bg_color)

    # 绘制触觉器（如果存在）
    if detector:
        detector.blitme()
    # draw the dusts
    dusts.draw(screen)
    # draw the robot
    robot.blitme()

    # 绘制模式文本（如果存在）
    if mode_text:
        font = pygame.font.SysFont(None, 36)
        text = font.render(mode_text, True, (0, 0, 255))
        screen.blit(text, (10, 10))

    # 绘制探索地图
    if detector and ai_settings.debug_mode and robot.exploration_map:
        for y in range(len(robot.exploration_map)):
            for x in range(len(robot.exploration_map[0])):
                explore_value = min(robot.exploration_map[y][x], 10)
                color_value = min(255, int(explore_value * 25))
                pygame.draw.rect(screen, (color_value, 0, 0),
                                 (x * robot.map_resolution, y * robot.map_resolution,
                                  robot.map_resolution, robot.map_resolution), 1)

    # 显示网格访问次数
    if ai_settings.debug_mode and robot.grid_visits:
        font = pygame.font.SysFont(None, 24)
        for (x, y), visits in robot.grid_visits.items():
            text = font.render(f"{visits}", True, (0, 0, 255))
            screen_x = x * robot.map_resolution + 5
            screen_y = y * robot.map_resolution + 5
            screen.blit(text, (screen_x, screen_y))

    # 显示当前目标
    if robot.escape_target:
        pygame.draw.circle(screen, (255, 0, 0), robot.escape_target, 10, 2)
        pygame.draw.line(screen, (255, 0, 0),
                         (robot.rect.centerx, robot.rect.centery),
                         robot.escape_target, 2)

    # visualiaze the window
    pygame.display.flip()


def create_dust(ai_settings, screen, dusts):
    """Create dust, and place it in the room."""
    dust = Dust(ai_settings, screen)
    dust.rect.x = random.randint(50, ai_settings.screen_width - 50)
    dust.rect.y = random.randint(50, ai_settings.screen_height - 50)
    dusts.add(dust)


def create_room(ai_settings, screen, dusts):
    """Create a full room of dusts."""
    for dust_number in range(ai_settings.dust_number):
        create_dust(ai_settings, screen, dusts)


# 清理完成后任意键退出
def show_game_over_screen(screen, message):
    """Display a game over message and wait for a key press."""
    # Set up the font
    font = pygame.font.SysFont(None, 60)
    small_font = pygame.font.SysFont(None, 36)

    # Render the text
    text = font.render(message, True, (255, 0, 0))
    instruction = small_font.render("Press any key to exit", True, (200, 200, 200))

    # Get text rectangles
    text_rect = text.get_rect(center=(screen.get_rect().centerx, screen.get_rect().centery - 30))
    instr_rect = instruction.get_rect(center=(screen.get_rect().centerx, screen.get_rect().centery + 30))

    # Create a semi-transparent overlay  创建半透明覆盖层
    overlay = pygame.Surface(screen.get_size())
    overlay.set_alpha(180)
    overlay.fill((0, 0, 0))

    # Draw everything
    screen.blit(overlay, (0, 0))
    screen.blit(text, text_rect)
    screen.blit(instruction, instr_rect)
    pygame.display.flip()

    # # Wait for a key press
    # waiting = True
    # while waiting:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
    #             waiting = False
