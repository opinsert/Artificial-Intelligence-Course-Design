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
    robot.init_exploration_map(ai_settings.screen_width, ai_settings.screen_height)
    # Create the fleet of dusts.
    gf.create_room(ai_settings, screen, dusts)
    # 控制循环时间
    clock = pygame.time.Clock()
    # 添加游戏结束标志
    game_over = False
    # 添加弹窗显示标志
    game_over_shown = False
    # 添加游戏状态标志
    MODE_SELECT = 0
    GAME_RUNNING = 1
    GAME_OVER = 2
    game_state = MODE_SELECT
    # 初始化触觉器
    detector = Detector(screen, robot)
    robot.set_detector(detector)
    robot.set_dusts(dusts)

    # 添加模型保存标
    model_saved = False

    # 训练状态跟踪
    training_steps = 0
    status_font = pygame.font.SysFont(None, 24)
    status_text = ""  # 存储当前状态文本
    status_render = None  # 存储渲染后的文本对象

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
                # 模式选择阶段
                if game_state == MODE_SELECT:
                    if event.key == pygame.K_1:  # 手动模式
                        manual_mode = True
                        game_state = GAME_RUNNING
                    elif event.key == pygame.K_2:  # 自动模式
                        robot.set_automatic_mode(True)
                        game_state = GAME_RUNNING

                # 游戏结束阶段
                elif game_state == GAME_OVER:
                    pygame.quit()
                    sys.exit()

                # 游戏运行阶段（手动模式）
                elif game_state == GAME_RUNNING and manual_mode:
                    gf.check_keydown_events(event, robot)
            elif event.type == pygame.KEYUP and game_state == GAME_RUNNING and manual_mode:
                gf.check_keyup_events(event, robot)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    robot.nn.save_model()
                    print("手动保存模型成功！")
        # 确定当前模式文本
        mode_str = None
        if mode_selected:
            mode_str = "Manual Mode (1)" if manual_mode else "Automatic Mode (2)"

        # 根据游戏状态更新屏幕
        if game_state == MODE_SELECT:
            # 绘制模式选择界面
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

        elif game_state == GAME_RUNNING:
            # 更新机器人
            robot.update(robot, ai_settings)

            # 更新游戏状态
            detector.update()

            # 显示边框检查状态
            if robot.border_points:
                border_text = status_font.render("Border Check in progress...", True, (255, 165, 0))
                screen.blit(border_text, (10, 170))

            # 确保探索地图已初始化
            if not robot.exploration_map:
                robot.init_exploration_map(ai_settings.screen_width, ai_settings.screen_height)

            # 添加实际碰撞检测逻辑
            collisions = pygame.sprite.spritecollide(robot, dusts, True)
            if collisions:
                robot.reset_idle_timer()
                robot.handle_collision()  # 调用碰撞处理方法

            # 加载模型
            model_status = "模型: " + ("已加载" if robot.nn.load_model else "新训练")
            model_text = status_font.render(model_status, True, (0, 100, 0))
            screen.blit(model_text, (10, 80))

            # 确定当前模式文本
            mode_str = "Manual Mode (1)" if manual_mode else "Automatic Mode (2)"

            #显示灰尘数量和机器人位置
            dust_count = len(dusts)
            dust_text = status_font.render(f"Dusts: {dust_count}", True, (255, 0, 0))
            screen.blit(dust_text, (10, 110))

            pos_text = status_font.render(f"Robot: ({int(robot.x)}, {int(robot.y)})", True, (0, 0, 255))
            screen.blit(pos_text, (10, 140))

            # 更新屏幕
            gf.update_screen(ai_settings, screen, dusts, robot, detector, mode_str)
            # 在自动模式下显示训练状态
            if not manual_mode:
                # 修改：无论手动/自动都进行训练
                # 每100帧更新一次训练状态显示
                if training_steps % 100 == 0:
                    # 获取神经网络状态
                    buffer_size = len(robot.nn.replay_buffer)
                    buffer_capacity = robot.nn.buffer_size
                    buffer_percent = int((buffer_size / buffer_capacity) * 100) if buffer_capacity > 0 else 0

                    # 创建训练状态文本
                    status_text = f"Training: Buffer {buffer_size}/{buffer_capacity} ({buffer_percent}%)"
                    status_render = status_font.render(status_text, True, (0, 0, 0))

                # 绘制训练状态（每帧都绘制）
                if status_render:
                    screen.blit(status_render, (10, 50))

                training_steps += 1

            # 检查游戏是否结束
            if len(dusts) == 0:
                # 保存模型（仅在第一次检测到游戏结束时）
                if not model_saved:
                    robot.nn.save_model()
                    print("游戏结束时自动保存模型成功！")
                    model_saved = True

                game_state = GAME_OVER
                gf.show_game_over_screen(screen, "All dusts have been cleared")

            # 检查是否所有灰尘都被清除
            if len(dusts) == 0:
                game_over = True


        elif game_state == GAME_OVER:
            # 保持显示结束画面
            pygame.display.flip()

        clock.tick(100)  # framerate = 100
    # 按间距中的绿色按钮以运行脚本。


if __name__ == '__main__':
    run_game()