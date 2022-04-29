"""
! Name: IntellgentStart.py
! Author: lolisky
! Date: 2022-04-22

功能：基于切换连接进行的循环通信
     接收网关发来的智能场景信息
     分析视频画面物体
     分析视频画面人物动作
     发回给网关 场景触发信息
"""
import socket
import requests
import json
import time
import multiprocessing
import cv2
import re
import mediapipe as mp
import numpy as np

from multiprocessing import Process
import multiprocessing
from classlib import DScene, main_control, Weather
from yolov5_onnx import mult_test

# 返回对应两序号节点的向量（数对）
def backvector(list,n,m):
    return np.array([list[n][0]-list[m][0],list[n][1]-list[m][1],list[n][2]-list[m][2]])

# 计算输入向量的模
def calculdist_sq(vec):
    return round(vec[0]**2+vec[1]**2+vec[2]**2,2)

# 监听网关发来的命令进程
def listening_deal(antenna, mylist):
    while 1:
        try:
            mc = mylist[0]
            mc.scenes.clear()
            print("listening...")
            conn, adds = antenna.accept()
            received_commands = conn.recv(10240)  # 接收命令请求,设置所能接收字节数
            # 字符串类型命令处理
            received_commands = received_commands.decode('utf-8')  # 对网络传输的字节进行解码
            if received_commands.upper() == 'Q':  # 网关发来结束字符'Q'---> 正常结束
                order = 'Q'
            # Json类型命令处理
            recvData = eval(received_commands)
            print(f'来自网关的消息：{recvData}')
            mc.addres = recvData['videoStreamaddress']
            sccount = len(recvData['scenes'])
            print("共" + str(sccount) + "个场景")
            for i in range(sccount):                # 遍历所有场景，获取场景信息
                value = recvData['scenes'][i]
                s = DScene()
                s.UserID = value['UserID']
                s.DCameraID = value['DCameraID']
                s.DSceneID = value['DSceneID']
                s.AreaPointsNumber = value['AreaPointsNumber']
                receive_points = value['AreaPointsPosition']
                x1 = -1
                x2 = 100000
                y1 = -1
                y2 = 100000
                for i in range(s.AreaPointsNumber):     # 规格化多个坐标点围成的区域为矩形
                    if (x1 < int(receive_points[i]['x'])):
                        x1 = int(receive_points[i]['x'])
                    if (x2 > int(receive_points[i]['x'])):
                        x2 = int(receive_points[i]['x'])
                for j in range(s.AreaPointsNumber):
                    if (y1 < int(receive_points[j]['y'])):
                        y1 = int(receive_points[j]['y'])
                    if (y2 > int(receive_points[j]['y'])):
                        y2 = int(receive_points[j]['y'])
                s.AreaPointsPositions = [x2, x1, y2, y1]
                s.ValidTimeStart = value['ValidTimeStart'][11:16]
                s.ValidTimeEnd = value['ValidTimeEnd'][11:16]
                s.DWeather = value['DWeather']
                items = []
                motion = int(value['DHumanMotion'])
                if motion == 3:
                    items.append(0)  # 区域入侵
                    s.DHumanMotion = -1
                elif motion == 4:
                    items.append(100)  # 区域离开
                    s.DHumanMotion = -1
                else:
                    s.DHumanMotion = motion
                # s.DItem = value['DItem']
                item = int(value['DItem'])
                if item != -1:
                    items.append(item)
                # items.append(int(value['DItem']))
                s.DItem = items
                mc.scenes.append(s)
            mylist[0] = mc
        except:
            print("Listening Wrong")

# 智能分析进程
def processing_deal(mylist,wthlist):
    wth = wthlist[0]  # 天气状态
    # 默认地址（后续根据网关指令更改）
    # cap = cv2.VideoCapture("rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4")
    # old = "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4"
    old = ""
    while 1:
        antenna_send = socket.socket()
        try:
            antenna_send.connect(('127.0.0.1', 6667))
        except:
            print("初始化连接失败，再次发送……")
            continue
        initjson = {"messageType": "IntelligentStartM",
                  "messageTime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
        antenna_send.send(repr(initjson).encode('utf-8'))
        antenna_send.close()
        break
    while 1:
        print("begin dealing")
        mc = mylist[0]
        scenecount = len(mc.scenes)
        print(scenecount)
        if old != mc.addres:
            print("修改流地址了")
            cap = cv2.VideoCapture(mc.addres)
            old = mc.addres
        flag_valid_time = 0     # 时间有效位标志，看看该时间段是否符合各个场景的限制，符合则进入（置为1），不符合则不进行算法识别
        flag_valid_weather = 0  # 天气有效位标志，看看该时间天气是否满足用户设定，符合则置1，不符合则不进行算法识别
        valid_time_set = []
        valid_weather_set = []
        result = []
        try:
            for i in range(scenecount):
                print(str(i) + " Motion:" + str(mc.scenes[i].DHumanMotion))
                s = mc.scenes[i]
                starttime = int(s.ValidTimeStart.split(':')[0] + s.ValidTimeStart.split(':')[1])  # 取整型值用于比较
                endtime = int(s.ValidTimeEnd.split(':')[0] + s.ValidTimeEnd.split(':')[1])
                now = time.strftime("%H:%M", time.localtime())  # 取系统当前时间，进行判断
                now = int(now.split(':')[0] + now.split(':')[1])
                if (starttime <= now and now <= endtime):  # 判断是否符合设定的时间条件
                    flag_valid_time = 1
                    valid_time_set.append(1)
                else:
                    valid_time_set.append(0)
                if (wth.DWeather == s.DWeather):
                    flag_valid_weather = 1
                    valid_weather_set.append(1)
                else:
                    valid_weather_set.append(0)
            if (flag_valid_time == 1 and flag_valid_weather == 1):
                # if  old != mc.addres:
                #     print("修改流地址了")
                #     cap = cv2.VideoCapture(mc.addres)
                #     old = mc.addres
                print("符合时间及天气条件，开始智能分析")
                # 导入画面
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing_styles = mp.solutions.drawing_styles
                mp_pose = mp.solutions.pose
                yoloresult = [[], [], [], [], [], [], [], []]
                pose_state = []  # 默认无动作状态

                n = 0
                # address = mc.addres
                # cap = cv2.VideoCapture(address)
                # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                # print("宽： " + str(width))
                # print("高: " + str(height))

                with mp_pose.Pose(
                        static_image_mode=False,
                        # 静态图像模式=否：将尝试检测第一张图像中最突出的人，并在成功检测后进一步定位姿势标志。在随后的图像中，它只是简单地跟踪这些地标而不调用另一个检测，直到它失去跟踪，以减少计算和延迟。
                        model_complexity=1,  # 模型复杂度[0,1,2]
                        smooth_landmarks=True,  # 平滑地标 设置为True则过滤不同输入图像中的地标来减少抖动
                        min_detection_confidence=0.5,  # 人体检测模型最小置信度
                        min_tracking_confidence=0.5  # 跟踪模型最小置信度
                ) as pose:
                    # 姿态状态码集
                    # pose_state = []  # 默认无动作状态
                    DScenes_count = len(mc.scenes)
                    for i in range(24):
                        success, image = cap.read()
                    success, image = cap.read()
                    image_yolo = image
                    # if image == None:
                    #     time.sleep(1)
                    # success, image = cap.read()
                    print("共" + str(DScenes_count) + "个场景")
                    object_detect = []  # 记录物体识别结果
                    flag_yolo = 0  # 标志是否有物品识别需求的场景
                    # yoloresult = [[], [], [], [], [], [], [], []]
                    pose_state = []
                    yolo_open = []  # 标志场景是否需要进行物体识别
                    for i in range(DScenes_count):
                        object_detect.append([])
                        flag_yxz = 1  # 默认检测画面中有下半身腿部
                        if (valid_time_set[i] == 0 or valid_weather_set[i] == 0):  # 如果该场景未到触发时间 或者 不符合天气条件设定
                            pose_state.append(-1)
                            # yoloresult.append([])
                            yolo_open.append(0)
                            continue
                        x1 = mc.scenes[i].AreaPointsPositions[0]
                        x2 = mc.scenes[i].AreaPointsPositions[1]
                        y1 = mc.scenes[i].AreaPointsPositions[2]
                        y2 = mc.scenes[i].AreaPointsPositions[3]
                        # boxloc.append([x1, x2, y1, y2])
                        # 进行画面裁剪
                        image = image_yolo
                        image = image[y1:y2, x1:x2]
                        print("裁剪后宽：" + str(x1) + "~" + str(x2) + " 高：" + str(y1) + "~" + str(y2))
                        # image = image[0:int(width/2),0:int(height)]
                        if not success:
                            print("Ignoring empty camera frame.")
                            continue
                        # 转换为RGB
                        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        # 当规定有动作时才进行Media Pipe 姿态识别
                        if mc.scenes[i].DHumanMotion >= 0:
                            # 异常处理
                            try:
                                # 使用Mediapipe模型进行处理
                                # 提高性能
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                image.flags.writeable = False
                                results = pose.process(image)
                                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            except:
                                print("Wrong in Processing !")
                                continue
                            pose_landmarks = results.pose_landmarks

                            # 在画面上画出识别出的各个人体节点
                            image.flags.writeable = True
                            # 存入各个姿态坐标点
                            landmark_list = []
                            # 规格化后的坐标点
                            pose_list = []

                            # 用来存储各姿态点的矩形坐标
                            pose_x_list = []
                            pose_y_list = []
                            pose_z_list = []

                            # 确保有检测点 防止遇到"检测为空"陷阱
                            if results.pose_landmarks:
                                for landmark_id, pose_axis in enumerate(
                                        results.pose_landmarks.landmark):
                                    landmark_list.append([
                                        landmark_id, pose_axis.x, pose_axis.y,
                                        pose_axis.z
                                    ])
                                    pose_list.append(
                                        [round(pose_axis.x * 10, 1), round(pose_axis.y * 10, 1),
                                         round(pose_axis.z * 10, 1)])
                                    pose_x_list.append(round(pose_axis.x * 10, 2))
                                    pose_y_list.append(round(pose_axis.y * 10, 2))
                                    pose_z_list.append(round(pose_axis.z * 10, 2))

                                # 如果腿部没有检测出来，则不做进一步处理
                                if pose_y_list[25] >= 10 or pose_y_list[26] >= 10:
                                    pose_state.append(-1)
                                    flag_yxz = 0
                                    # continue

                                # 在检测出画面中有腿存在时，进行进一步处理
                                # 向量储备
                                mid_xiaofu = np.array(
                                    [(pose_x_list[24] + pose_x_list[23]) / 2, (pose_y_list[24] + pose_y_list[23]) / 2,
                                     (pose_z_list[24] + pose_z_list[23]) / 2])
                                mid_xiong = np.array(
                                    [(pose_x_list[12] + pose_x_list[11]) / 2, (pose_y_list[12] + pose_y_list[11]) / 2,
                                     (pose_z_list[12] + pose_z_list[11]) / 2])
                                qugan = mid_xiaofu - mid_xiong
                                # xiaobi_r = np.array([pose_x_list[16]-pose_x_list[14],pose_y_list[16]-pose_y_list[14],pose_z_list[16]-pose_z_list[14]])
                                # xiaobi_l = np.array([pose_x_list[15]-pose_x_list[13],pose_y_list[15]-pose_y_list[13],pose_z_list[15]-pose_z_list[13]])
                                xiaobi_r = backvector(pose_list, 16, 14)
                                xiaobi_l = backvector(pose_list, 15, 13)
                                datui_r = backvector(pose_list, 26, 24)
                                datui_l = backvector(pose_list, 25, 23)
                                xiaotui_r = backvector(pose_list, 28, 26)
                                xiaotui_l = backvector(pose_list, 27, 25)
                                thleg_r = backvector(pose_list, 28, 24)
                                thleg_l = backvector(pose_list, 27, 23)

                                lenxiaotui_r = calculdist_sq(xiaotui_r)
                                lendatui_r = calculdist_sq(datui_r)
                                lenthleg_r = calculdist_sq(thleg_r)
                                lenxiaotui_l = calculdist_sq(xiaotui_l)
                                lendatui_l = calculdist_sq(datui_l)
                                lenthleg_l = calculdist_sq(thleg_l)

                                print("右小腿长： " + str(lenxiaotui_r))
                                print("右大腿长： " + str(lendatui_r))
                                print("右腿第三边长： " + str(lenthleg_r))
                                print("左小腿长： " + str(lenxiaotui_l))
                                print("左大腿长： " + str(lendatui_l))
                                print("左腿第三边长： " + str(lenthleg_l))

                                # 姿态识别模块
                                if flag_yxz:
                                    if qugan[1] > 0.1 and lenthleg_r < (
                                            lendatui_r + lenxiaotui_r) * 0.9 and lenthleg_l < (
                                            lendatui_l + lenxiaotui_l) * 0.9:
                                        # map_sit['sit'] += 1
                                        pose_state.append(1)
                                    elif pose_x_list[25] < pose_x_list[23] and pose_x_list[25] < pose_x_list[
                                        26] * 0.8 and pose_x_list[23] * 1.1 < pose_x_list[24]:
                                        pose_state.append(2)
                                    elif pose_x_list[26] > pose_x_list[24] and pose_x_list[26] > pose_x_list[
                                        24] * 1.2 and pose_x_list[23] * 1.1 < pose_x_list[24]:
                                        pose_state.append(2)
                                    elif abs(pose_x_list[26] - pose_x_list[25]) > abs(
                                            pose_x_list[24] - pose_x_list[23]) * 1.4:
                                        pose_state.append(-1)
                                        # map_sit['stand'] += 1
                                    else:
                                        pose_state.append(0)
                            else:
                                pose_state.append(-1)
                        else:
                            pose_state.append(-1)  # 未设定人物动作时，默认值为-1
                        # 当规定有物体时，才进行Yolo识别
                        if len(mc.scenes[i].DItem) > 0:
                            print("要识别的物体：")
                            print(mc.scenes[i].DItem)
                            yolo_open.append(1)
                            flag_yolo = 1
                        else:
                            yolo_open.append(0)  # 未有规定，则不识别
                        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    if flag_yolo == 1:      # 开启物体识别
                        cv2.imwrite("cache.jpg", image_yolo)
                        time.sleep(0.00001)
                        yoloresult_origi = mult_test(r'yolov5s.onnx', "cache.jpg")
                        print(yoloresult_origi)
                        for i in range(DScenes_count):
                            if yolo_open[i] == 1:
                                for n in range(len(mc.scenes[i].DItem)):
                                    itemindex = mc.scenes[i].DItem[n]
                                    x1 = mc.scenes[i].AreaPointsPositions[0]
                                    x2 = mc.scenes[i].AreaPointsPositions[1]
                                    y1 = mc.scenes[i].AreaPointsPositions[2]
                                    y2 = mc.scenes[i].AreaPointsPositions[3]
                                    if itemindex >= 100:
                                        itemindex = itemindex - 100
                                    detected = len(yoloresult_origi[itemindex])
                                    if detected > 0:
                                        for j in range(detected):
                                            x_cen = (yoloresult_origi[itemindex][j][0] + yoloresult_origi[itemindex][j][2]) / 2
                                            y_cen = (yoloresult_origi[itemindex][j][1] + yoloresult_origi[itemindex][j][3]) / 2
                                            if x1 < x_cen < x2 and y1 < y_cen < y2:
                                                object_detect[i].append(itemindex)
                                                break  # 找到了一个，就不再继续，以免重复
                    yoloresult = object_detect
                    result = pose_state

                print("PoseResult:")
                print(result)
                print("yoloresult: ")
                print(yoloresult)
                # 发送数组
                # conn.send(repr(result).encode('utf-8'))
                # 发送json
                rejson = {"messageType": "ScenarioTrigger",
                          "messageTime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
                UserID = []
                DCameraID = []
                DSceneID = []
                flag_appear = 0     # 判断有无触发成功，成功则发回执信息
                for i in range(len(mc.scenes)):
                    flag_seted_yolo = 1
                    flag_seted_pose = 1
                    flag_pose = 0  # 判断姿态有无触发成功，成功则发回执信息
                    flag_yolo = 0
                    if len(mc.scenes[i].DItem) >= 0:
                        if len(mc.scenes[i].DItem) == 0:
                            flag_seted_yolo = 0
                        flag_item = 1
                        for j in range(len(mc.scenes[i].DItem)):
                            item = mc.scenes[i].DItem[j]
                            if item >= 100:
                                reitem = item - 100
                                if (yoloresult[i].count(reitem) != 0):
                                    flag_item = 0
                                    break
                            else:
                                if (yoloresult[i].count(item) <= 0):
                                    flag_item = 0
                                    break
                        if flag_item == 1:
                            flag_yolo = 1
                    else:
                        flag_seted_yolo = 0
                        flag_yolo = 1

                    if mc.scenes[i].DHumanMotion >= 0:
                        if (result[i] == mc.scenes[i].DHumanMotion):
                            flag_pose = 1  # 成功则置该标志位为 1
                    else:
                        flag_seted_pose = 0
                        flag_pose = 1
                    print(str(i))
                    print("flag_seted_pose:" + str(flag_seted_pose))
                    print("flag_seted_yolo:" + str(flag_seted_yolo))
                    if flag_yolo and flag_pose:
                        if flag_seted_yolo or flag_seted_pose:
                            flag_appear = 1
                            UserID.append(mc.scenes[i].UserID)
                            DCameraID.append(mc.scenes[i].DCameraID)
                            DSceneID.append(mc.scenes[i].DSceneID)
                            rejson['UserID'] = UserID
                            rejson['DCameraID'] = DCameraID
                            rejson['DSceneID'] = DSceneID

                if (flag_appear == 1):
                    antenna_send = socket.socket()
                    # antenna_send.connect(('192.168.31.52', 6667))
                    try:
                        antenna_send.connect(('127.0.0.1', 6667))
                        antenna_send.send(repr(rejson).encode('utf-8'))
                        antenna_send.close()
                    except:
                        print("Connect False...")
                        antenna_send.close()
            else:
                print("不进行分析")
            flag_valid_time = 0  # 重置标志位
        except TypeError as ee:
            print("sleeping.........")
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(old)
        except Exception as e:
            print("Othering.........")
            time.sleep(1)
            print('错误类型是', e.__class__.__name__)
            print('错误明细是', e)
            print("Wrong in Processing !")
            antenna_send = socket.socket()
            # antenna_send.connect(('192.168.31.52', 6667))
            try:
                antenna_send.connect(('127.0.0.1', 6667))
                antenna_send.send("Wrong in Processing !".encode('utf-8'))
                antenna_send.close()
            except:
                print("Connect False...")
                antenna_send.close()


def getweather():
    # 获取城市名
    city_url = "http://api.map.baidu.com/location/ip?ak=ypxIEHth7hQj9A2qsMVFEe0mGSazTk2a"  # 获取城市名的网址
    respon_city = requests.get(city_url)  # 获取返回的值
    loc_detail = json.loads(respon_city.text)  # 取其中某个字段的值
    loc_city = loc_detail['content']['address_detail']['city']
    print(loc_detail)
    print(loc_city)

    # https://api.map.baidu.com/weather/v1/?district_id=222405&data_type=all&ak=ypxIEHth7hQj9A2qsMVFEe0mGSazTk2a
    # http://api.map.baidu.com/location/ip?ak=ypxIEHth7hQj9A2qsMVFEe0mGSazTk2a

    # 获取经纬度
    latlon_url = "https://geoapi.qweather.com/v2/city/lookup?key=bb3e9850157f4a7f901e9de4bad2170f&location=" + \
                 loc_city  # 获取城市经纬度的网址
    respon_loc = requests.get(latlon_url)  # 获取返回的值
    loc = json.loads(respon_loc.text)  # 取其中某个字段的值
    lat = round(float(loc['location'][0]['lat']), 2)  # 纬度
    lon = round(float(loc['location'][0]['lon']), 2)  # 经度
    print("纬度：" + str(lat) + "  经度：" + str(lon))

    location = str(lon) + "," + str(lat)
    weather_url = "https://devapi.qweather.com/v7/weather/now?key=bb3e9850157f4a7f901e9de4bad2170f&location=" + location  # 获取天气的网址
    r = requests.get(weather_url)  # 获取返回的值
    weather = json.loads(r.text)  # 取其中某个字段的值
    text = weather['now']['text']       # 天气信息
    temp = weather['now']['temp']       # 温度信息
    print(weather)
    print("Weather:" + text)
    print("temp:" + temp)
    pattern = re.compile(r"['雨','雪','晴','云','雾','霾','沙']")
    result = pattern.findall(text)  # 0：晴天  1：雨天  2：雪天  3：雨雪  4：阴天  5：雾天  6：雾霾  7：沙尘
    text = -1
    if len(result) == 0:
        text = -1
    elif len(result) == 1:
        if result[0] == '晴':
            text = 0
        elif result[0] == '雨':
            text = 1
        elif result[0] == '雪':
            text = 2
        elif result[0] == '阴' or result[0] == '云':
            text = 4
        elif result[0] == '雾':
            text = 5
        elif result[0] == '霾':
            text = 6
        elif result[0] == '沙':
            text = 7
    elif len(result) == 2:
        if result[0] == '雨' and result[1] == '雪':
            text = 3
        elif result[1] == '雨' and result[0] == '雪':
            text = 3
        elif result[0] == '霾' or result[1] == '霾':
            text = 6
    print('DWeather: ' + str(text))
    return text, temp

def weather_deal(wthlist):
    while 1:
        try:
            time.sleep(3600)
            text, temp = getweather()
            wth = Weather()
            wth.DWeather = text
            wth.temp = temp
            wthlist[0] = wth
        except:
            print("获取天气失败，等待重新获取……")
            text, temp = getweather()
            wth = Weather()
            wth.DWeather = text
            wth.temp = temp
            wthlist[0] = wth

if __name__ == '__main__':
    multiprocessing.freeze_support()
    print("开始程序...")
    mylist = multiprocessing.Manager().list()
    wthlist = multiprocessing.Manager().list()
    mc = main_control()
    mylist.append(mc)
    wth = Weather()
    firstDWeather, firsttemp = getweather()
    wth.DWeather = firstDWeather
    wth.temp = firsttemp
    wthlist.append(wth)
    antenna = socket.socket()  # 实例化套接字通讯对象

    antenna.bind(('127.0.0.1', 6666))  # 存放套接字
    antenna.listen(5)  # 设置连接的个数
    p_recv = Process(target=listening_deal, args=([antenna, mylist]))
    p_proc = Process(target=processing_deal, args=([mylist, wthlist]))
    p_weath = Process(target=weather_deal, args=([wthlist]))
    # p_proc = Process(target=processing_deal, args=([antenna_send, mylist]))
    p_recv.start()
    p_proc.start()
    p_weath.start()
    p_recv.join()
    p_proc.join()
    p_weath.join()

