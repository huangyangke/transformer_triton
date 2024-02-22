# -*- encoding: utf-8 -*-
'''
@File    : utils.py
@Time    : 2022/05/26 18:29:05
@Author  : hw
@Contact : zhangbiao@mgtv.com
@Version : 1.0
@Desc    : None
'''



import numpy as np
import cv2
import base64
import requests

def parse_item_image(item):
    if item[:4] == "http":
        image = UrlImageDecode(item)
    elif item[:4] == "path":
        image_path = item.strip("path:")
        image = cv2.imread(image_path)
    elif item[:4] == "data":
        b64code = item.strip("data:image/jpeg;base64,")
        image = B64ImageDecode(b64code)
    else:
        raise ValueError(
            "invalid input item, must be one of path, url, base64 code")
    return image

def BufferImageDecode(image_buffer):
    nparr = np.frombuffer(image_buffer, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    return image

def B64ImageDecode(image_b64):
    image_buffer = base64.b64decode(image_b64)
    return BufferImageDecode(image_buffer)

def UrlImageDecode(image_url):
    r = requests.get(image_url,stream=True)
    image_buffer=r.content
    return BufferImageDecode(image_buffer)

def B64ImageEncode(image_array):
    rect,image_buffer=cv2.imencode(".jpg", image_array)
    image_b64=base64.b64encode(image_buffer)
    return image_b64


