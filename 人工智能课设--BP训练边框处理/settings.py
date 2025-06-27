# -*- coding:utf-8 -*-
class Settings(object):
    # """docstring for Settings"""
    def __init__(self):
        # initialize setting of game
        # screen setting
        self.screen_width = 1000
        self.screen_height = 600
        self.bg_color = (230, 230, 230)
        self.dust_number = 10
        self.nn_input_size = 3
        self.nn_hidden_size = 5
        self.nn_output_size = 1
        self.debug_mode = False
        self.debug_mode = False  # 设置为True以显示调试信息
