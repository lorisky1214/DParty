![image](https://github.com/lorisky1214/DParty/blob/main/source/logo.png)

## Intro

A smart home SDK uses the intranet loopback address socket to input the scene. Through the yolov5 and Media Pose network analysis contained in the SDK, it returns whether the scene is effectively triggered.

## Requirements

Python 3.8 or later with all [requirements.txt](https://github.com/lorisky1214/DParty/blob/main/requirements.txt) dependencies installed. To install run:

```bash
$ pip install -r requirements.txt
```

## SDK Usage

```python
from DParty import IntelligentStart

if __name__ == '__main__':
    bind_port = 6666	# SDK used to bind your scene message
    send_port = 6667	# SDK used to send the triggered to you
    IntelligentStart.Start()	# Start
```



