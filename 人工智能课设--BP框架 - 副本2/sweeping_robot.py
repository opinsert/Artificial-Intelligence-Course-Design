import pygame
from pygame.sprite import Group
from settings import Settings
import game_functions as gf
from robot import Robot
from detector import Detector
import sys
import time


def run_game():
    # initialize game and create a dispaly object
    pygame.init()
    ai_settings = Settings()
    screen = pygame.display.set_mode((ai_settings.screen_width, ai_settings.screen_height))
    pygame.display.set_caption("Sweeping Robot")
    # initialize the group of dusts
    dusts = Group()
    # initialize the robot
    robot = Robot(screen)
    new_robot = Robot(screen)
    # Create the fleet of dusts.
    gf.create_room(ai_settings, screen, dusts)
    # 控制循环时间
    clock = pygame.time.Clock()
    # 添加游戏结束标志
    game_over = False
    # 添加弹窗显示标志
    game_over_shown = False
    # 初始化触觉器
    detector = Detector(screen, robot)
    robot.set_detector(detector)
    robot.set_dusts(dusts)

    # 模式选择标志
    mode_selected = False
    manual_mode = False

    # 创建提示信息字体
    font = pygame.font.SysFont(None, 48)
    small_font = pygame.font.SysFont(None, 36)

    # game loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                # 模式选择
                if not mode_selected:
                    if event.key == pygame.K_1:  # 手动模式
                        manual_mode = True
                        mode_selected = True
                    elif event.key == pygame.K_2:  # 自动模式
                        robot.set_automatic_mode(True)
                        mode_selected = True
                elif game_over:  # 游戏结束时按任意键退出
                    pygame.quit()
                    sys.exit()
                elif manual_mode:  # 手动模式下处理按键
                    gf.check_keydown_events(event, robot)
            elif event.type == pygame.KEYUP and manual_mode and not game_over:
                gf.check_keyup_events(event, robot)

        # 确定当前模式文本
        mode_str = None
        if mode_selected:
            mode_str = "Manual Mode (1)" if manual_mode else "Automatic Mode (2)"

        # 仅在游戏未结束时更新游戏状态
        if not game_over:
            # 如果模式未选择，显示提示信息
            if not mode_selected:
                # 填充背景
                screen.fill(ai_settings.bg_color)

                # 创建提示信息
                title = font.render("Sweeping Robot Game", True, (0, 0, 0))
                prompt = small_font.render("Select Control Mode:", True, (0, 0, 0))
                manual_text = small_font.render("Press 1 for Manual Control", True, (0, 0, 255))
                auto_text = small_font.render("Press 2 for Automatic Control", True, (0, 0, 255))

                # 获取文本位置
                title_rect = title.get_rect(center=(screen.get_rect().centerx, screen.get_rect().centery - 80))
                prompt_rect = prompt.get_rect(center=(screen.get_rect().centerx, screen.get_rect().centery - 20))
                manual_rect = manual_text.get_rect(center=(screen.get_rect().centerx, screen.get_rect().centery + 30))
                auto_rect = auto_text.get_rect(center=(screen.get_rect().centerx, screen.get_rect().centery + 80))

                # 绘制文本
                screen.blit(title, title_rect)
                screen.blit(prompt, prompt_rect)
                screen.blit(manual_text, manual_rect)
                screen.blit(auto_text, auto_rect)

                pygame.display.flip()
            else:
                # 更新触觉器位置
                detector.update()
                # 检测碰撞并通知机器人

                collisions = pygame.sprite.spritecollide(new_robot, dusts, True, None)
                if collisions:
                    robot.reset_idle_timer()  # 通知机器人重置空闲计时器
                    # 在自动模式下，通知神经网络灰尘被收集
                    if robot.automatic_mode:
                        robot.reset_after_dust_collected()

                # update robot
                robot.update(new_robot)

                # update screen
                gf.update_screen(ai_settings, screen, dusts, new_robot, detector, mode_str)

                # 检查是否所有灰尘都被清除
                if len(dusts) == 0:
                    game_over = True
        else:
            # 只在第一次进入游戏结束状态时显示弹窗
            if not game_over_shown:
                gf.show_game_over_screen(screen, "dusts have been cleared")
                game_over_shown = True
            # 保持显示结束画面
            pygame.display.flip()
        clock.tick(100)  # framerate = 100


if __name__ == '__main__':
    run_game()
