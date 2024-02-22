# -*- encoding: utf-8 -*-
'''
@File    : kserver_client.py
@Time    : 2022/05/26 18:26:44
@Author  : zhangbiao
@Contact : zhangbiao@mgtv.com
@Version : 1.0
@Desc    : kserver transformer接口测试
'''


import time
import requests
import cv2
import base64
import json
from common_data_type.request import Request,MultiRequest
from common_data_type.response import Response, ResponseItem, ExtraInfo, MultiResponse

# 本地测试
url = "http://127.0.0.1:8080/v1/models/facequality:predict"
# url = "http://127.0.0.1:8080/v2/models/facequality/infer"

# 远程测试
url="http://10.100.196.70/v1/models/facequality:predict"

headers = {
    # 'Host': 'facequality-transformer-transformer-default.mgtv-dev.api.aihub.imgo.tv',
    'Host': 'face-quality-new-transformer-default.mgtv.api.aihub.imgo.tv',
}

print("url:", url)
img = cv2.imread("data/1-Harry_Belafonte_1.jpg")
_, buffer = cv2.imencode(".png", img)
b64code = base64.b64encode(buffer)

# 是否启动批量数据发送
mutlti_request = False
mutlti_request = True

count = 2
batch = 16
if mutlti_request:
    b64code = f"data:image/jpeg;base64,{b64code}"
    
    multi_req = []
    for i in range(batch):
        request = Request(**{"data": {"item": b64code}})
        multi_req.append(request)
    multirequest = MultiRequest(multi_data=multi_req)
    
    for i in range(count):
        if i==1:
            t1 = time.time()
        result = requests.post(headers=headers, url=url, json=multirequest.dict())
        if i%10 ==0:
            print(i, result)
    t2 = time.time() 
    
else:
    b64code = f"data:image/jpeg;base64,{b64code}"
    request = Request(**{"data": {"item": b64code},
                    "extra_info": {"type": "subtitle-only"}})
                    
    for i in range(count):
        if i==1:
            t1 = time.time()
        result = requests.post(headers=headers, url=url, json=request.dict())
        print(i, result)
    t2 = time.time() 

count = count-1
print("total time cost:", t2-t1, "average time cost:",(t2-t1)/count)
# print("total time cost:", t2-t1, "average time cost:",(t2-t1)/count)
print("| ", batch, "|" ,t2-t1," | ",(t2-t1)/count,"s")
response = MultiResponse(**json.loads(result.text))

print("-"*20)
print(response.multi_data)

# 压力测试
from locust import HttpUser, task
# locust -f locust_stress_test.py
class TransformerTestCase(HttpUser):
    
    @task
    def stress_test(self):
        self.client.headers["Host"] = headers["Host"]
        self.client.post(url=url, json=request.dict())

