# -*- encoding: utf-8 -*-
'''
@File    : transformer.py
@Time    : 2022/05/25 11:26:11
@Author  : zhangbiao
@Contact : zhangbiao@mgtv.com
@Version : 1.0
@Desc    : None
'''

import time
import kserve
import numpy as np
from queue import Queue
from typing import Dict


from .utils import parse_item_image
from .common_data_type.request import Request, MultiRequest
from .common_data_type.response import Response, ResponseItem, ExtraInfo, MultiResponse

from .prediction import FaceQualityPrediction
from .triton_client import TritonHttpClient, TritonGrpcClient


class Transformer(kserve.Model):
    def __init__(self, name: str, predictor_host: str, use_grpc=True):
        super().__init__(name)

        self.predictor_host = predictor_host
        if use_grpc:
            self.tritonclient = TritonGrpcClient(
                predictor_host, concurrency=100)
        else:
            self.tritonclient = TritonHttpClient(
                predictor_host, concurrency=100)
        print("triton server predictor host: ", predictor_host)
        self.predictor = FaceQualityPrediction(self.tritonclient)

    def preprocess(self, request: Dict):
        t1 = time.time()
        request_info = {}
        images = []
        req_type = "Request"
        if "multi_data" in request:
            # batch
            multi_req = MultiRequest(**request)
            for req in multi_req.multi_data:
                if req.data.item is not None:
                    img = parse_item_image(req.data.item)
                    images.append(img)
            request_info["images"] = images
            req_type = "MultiRequest"
            if multi_req.extra_info is not None and multi_req.extra_info.others is not None:
                for key, value in multi_req.extra_info.others:
                    request_info[key[1]] = value[1]
        else:
            req = Request(**request)
            if req.data.item is not None:
                img = parse_item_image(req.data.item)
                images.append(img)
            request_info["images"] = images
            req_type = "Request"
            if req.extra_info is not None and req.extra_info.others is not None:
                for key, value in req.extra_info.others:
                    request_info[key[1]] = value[1]

        request_info['req_type'] = req_type
        print("transformer time cost: preprocessing", time.time() - t1)

        return request_info

    def predict(self, request: Dict):
        images = request["images"]
        batch = len(images)
        req_type = request['req_type']

        # 调用类型
        # # http同步
        # return self.http_infer(images)
        
        # # http异步
        # return self.http_async_infer(images)
        
        # # grpc同步
        # return self.grpc_infer(images)
        
        # # grpc异步
        return self.grpc_async_infer(images)

    def postprocess(self, request: Dict):
        t1 = time.time()
        content = []
        if "face_quality_score" in request:
            for score in request["face_quality_score"]:
                content.append(
                    Response(
                        data=[ResponseItem(score=score)],
                        extra_info=ExtraInfo(source="face_quality_score")
                    )
                )
        multi_response = MultiResponse(multi_data=content)

        print("transformer time cost: postprocessing", time.time() - t1)
        return multi_response.dict(exclude_unset=True)

    def http_infer(self, images):
        # synchronous
        t1 = time.time()
        print(">> http synchronous")
        scores = self.predictor.infer(images)
        result = {
            "face_quality_score": scores,
        }

        print("transformer time cost: predict", time.time() - t1)
        return result

    def http_async_infer(self, images):
        # asynchronous
        t1 = time.time()
        print(">> http asynchronous")
        async_requests = []
        sent_count = 500
        for i in range(sent_count):
            async_requests.append(self.predictor.async_infer(images))

        result = {}
        for j, async_request in enumerate(async_requests):
            res = self.predictor.postprocess(async_request.get_result())
            print("is_async", j, res)
            result["face_quality_score"] = res
        return result

    def grpc_infer(self, images):
        # synchronous
        t1 = time.time()
        print(">> grpc synchronous")
        scores = self.predictor.infer(images)
        result = {
            "face_quality_score": scores,
        }
        print("transformer time cost: predict", time.time() - t1)
        return result

    def grpc_async_infer(self, images):
        # 异步需要定义 timeout
        t1 = time.time()
        print(">> grpc asynchronous")
        result_queue = Queue()
        responses = []
        sent_count = 500

        for i in range(sent_count):
            self.predictor.grpc_async_infer(images, result_queue)

        processed_count = 0
        scores = []
        try:
            while processed_count < sent_count:
                (results, error) = result_queue.get(timeout=2) #单位s
                processed_count += 1
                if error is not None:
                    print("grpc asyn inference failed: " + str(error))
                else:
                    scores = results
                responses.append(results)
                    
                if processed_count % 100 == 0:
                    print("time cost:", processed_count, time.time() - t1, results)
        except Exception as e:
            print(e)
            
        result = {
            "face_quality_score": scores,
        }
        print("transformer time cost: predict", time.time() - t1)
        return result