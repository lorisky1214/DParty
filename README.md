![image](https://github.com/lorisky1214/DParty/blob/main/source/logo.png)

## Introduction

A smart home SDK uses the intranet loopback address socket to input the scene. Through the yolov5 and Media Pose network analysis contained in the SDK, it returns whether the scene is effectively triggered.

## Getting Started

Python 3.8 or later with all [requirements.txt](https://github.com/lorisky1214/DParty/blob/main/requirements.txt) dependencies installed. To install run:

```bash
$ pip install -r requirements.txt
```

## SDK Usage

1. Set up your own program.
2. Start the Intelligent analysis using the following example.

```python
from DParty import IntelligentStart

if __name__ == '__main__':
    bind_port = 6666	# SDK used to bind your scene message
    send_port = 6667	# SDK used to send the triggered to you
    IntelligentStart.Start(bind_port, send_port)	# Start
```

## Tutorials

Your program should be listening for a socket, and the SDK will be start after you receive a message json like this:

```python
initjson = {
    "messageType": "IntelligentStartM",
    "messageTime": "", 			# e.g. "2022-05-02T09:00:00"
}
```

Next, your program can send your scenes information in the following format to start analysis:

```python
sendData = {
    "videoStreamaddress":"",		# your video stream address
    "scenes": [
        {
            "UserID": "", 			# e.g. "DUSER2"
            "DCameraID": "", 		# e.g. "00-1A-2B-3C-4D-56"
            "DSceneID": "", 		# e.g. "SC001"
            "DHumanMotion": ,		# e.g. 1
            "DItem": , 				# e.g. 2
            "ValidTimeStart": "",  	# e.g. "2022-05-02T09:00:00"
            "AreaPointsNumber": ,	# e.g. 4
            "ValidTimeEnd": "",		# e.g. "2022-05-02T24:00:00 "
            'DWeather': ,			# e.g. 4				    							
            "AreaPointsPosition":[{"x":, "y": }, {"x":, "y": }, {"x":, "y": },{"x":, "y":}]},
        # e.g. [{"x": 10, "y": 10}, {"x": 22, "y": 22}, {"x": 77, "y": 77}, {"x": 664, "y": 297}]}
    ],
    "messageType":"",				# e.g. "AIProcessScenesConfigM"
    "messageTime": "", 				# e.g. "2022-005-02T07:55:00.0083193+08:00"
}
```

There are some references for the properties:

| DHumanMotion |         Refers to         |
| :----------: | :-----------------------: |
|      -1      | No Human Motion Detection |
|      0       |         Standing          |
|      1       |          Sitting          |
|      2       |         Climbing          |
|      3       |          Area-in          |
|      4       |         Area-out          |

| DItem |              Refers to               |
| :---: | :----------------------------------: |
|  -1   |          No Item Detection           |
|   0   |                People                |
|   1   |               Handbag                |
|   2   |                Bottle                |
|   3   |                Phone                 |
|   4   |                 Cat                  |
|   5   |                 Dog                  |
|   6   |               Umbrella               |
|   7   | Fruit(include banana, apple, orange) |



| DWeather |       Refers to       |
| :------: | :-------------------: |
|    -1    | No Weather Limitation |
|    0     |         Sunny         |
|    1     |         Rainy         |
|    2     |         Snowy         |
|    3     |    Rainy and Snowy    |
|    4     |        Cloudy         |
|    5     |         Foggy         |
|    6     |        Smoggy         |
|    7     |       Sand-dust       |

If the scene you send is triggered, you will get the triggered message in the following format, you can use it to do your downstream work:

```python
rejson = {
    "messageType": "ScenarioTrigger",
    "messageTime": "",				# e.g. 2022-05-02T12:00:00
    "UserID": "",					# e.g. DUSER2
    "DCameraID": "",				# e.g. 00-1A-2B-3C-4D-56
    "DSceneID": "",					# e.g. SC001
}
```



