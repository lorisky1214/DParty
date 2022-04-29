"""
! Name: classlib.py
! Author: lolisky
! Date: 2022-04-22

功能：为智能分析提供类库支持
"""
import time
import cv2

class DScene():
    def __init__(self):
        self.UserID = 0
        self.DCameraID = 1
        self.DSceneID = 1
        self.DSceneNickname = "233"
        self.AreaPointsNumber = 0  # 画框区域点数
        self.AreaPointsPositions = []  # 画框区域点位置字典
        self.MotionEnable = 0  # 动作条件是否开启
        self.DHumanMotion = 1  # 人体动作
        self.time_set = time.strftime("%Y-%m-%d %H:%M", time.gmtime())  # 0时区（格林威治）时间
        self.TimeEnable = 0  # 时间条件是否开启
        self.ValidTimeStart = time.strftime("%H:%M", time.strptime("00:00", "%H:%M"))  # 有效时间开始
        self.ValidTimeEnd = time.strftime("%H:%M", time.strptime("00:00", "%H:%M"))  # 有效时间结束
        self.DItem = [] # 物体识别列表
        self.DWeather = -1  # 气象限制

# 算法控制主类
class main_control():
    def __init__(self):
        # self.addres = "rtsp://admin:DengXuyao@192.168.31.254:554/h264/ch1/main/av_stream"      # 流地址
        self.addres = ""
        self.scenes = []

class Weather():
    def __init__(self):
        self.DWeather = -1  # DWeather: -1：不设限制  0：晴天  1：雨天  2：雪天  3：雨雪  4：阴天  5：雾天  6：雾霾  7：沙尘
        self.temp = None

class hc_control():
    def __init__(self):
        self.hcs = []
