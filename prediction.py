# -*- encoding: utf-8 -*-
import cv2
import numpy as np
from queue import Queue
from functools import partial


class FaceQualityPrediction():

    def __init__(self, tritonclient):
        self.tritonclient = tritonclient
        self.model_name = "face_quality_trt_fp16"
        self.input_names = ['input.1']
        self.output_names = ['1346']

    def preprocess(self, images):
        """图片数据前置处理流程

        Args:
            images (list): 图片数据列表

        Returns:
            numpy.ndarray: 4维Numpy数组
        """

        batch_img = []
        if len(images) > 0:
            for image in images:
                resized = cv2.resize(image, (112, 112))
                resized = resized[..., ::-1]    # BGR to RGB
                resized = resized.swapaxes(1, 2).swapaxes(0, 1)
                batch_img.append(resized)
        else:
            raise("Input data error, images.shape:", image.shape)

        batch_img = np.array(batch_img, dtype=np.float32)
        # normalization
        batch_img = (batch_img - 127.5) / 128.0

        return batch_img

    def infer(self, images):
        """人脸质量检测主流程

        Args:
            images (list): 图片数据列表

        Returns:
            list: 人脸质量分数列表
        """
        # 数据前置处理逻辑
        # t1 = time.time()
        input_data = self.preprocess(images)
        # t2 = time.time()

        input_dict = {"input.1": input_data}
        output_names = ['1346']

        results = self.tritonclient.infer(self.model_name, input_dict, output_names)
        # t3 = time.time()
        # print("predictor time cost total:{}, preprocess:{}, triton call:{}".format(
        #     t3-t1, t2-t1, t3-t2))

        return self.postprocess(results)

    def async_infer(self, images):
        """人脸质量检测主流程

        Args:
            images (list): 图片数据列表

        Returns:
            list: 人脸质量分数列表
        """
        # 数据前置处理逻辑
        # t1 = time.time()
        input_data = self.preprocess(images)
        # t2 = time.time()

        input_dict = {"input.1": input_data}
        output_names = ['1346']

        results = self.tritonclient.infer(self.model_name, input_dict, output_names, is_async=True)
        # t3 = time.time()

        # print("predictor time cost total:{}, preprocess:{}, triton call:{}".format(
        #     t3-t1, t2-t1, t3-t2))

        return results

    def grpc_async_infer(self, images:list, result_queue:Queue):
        """人脸质量检测主流程

        Args:
            images (list): 图片数据列表

        Returns:
            list: 人脸质量分数列表
        """
        # 数据前置处理逻辑
        # t1 = time.time()
        input_data = self.preprocess(images)
        # t2 = time.time()

        input_dict = {"input.1": input_data}
        output_names = ['1346']

        def asyn_callback(user_queue, result, error):
            """
                Define the callback function. Note the last two parameters should be
                result and error. InferenceServerClient would povide the results of an
                inference as grpcclient.InferResult in result. For successful
                inference, error will be None, otherwise it will be an object of
                tritonclientutils.InferenceServerException holding the error details
            """
            if error:
                print("asyn_callback error >> ", result, error)
                user_queue.put((None, error))
            else:
                scores = self.postprocess(result)
                user_queue.put((scores, error))
        
        self.tritonclient.async_infer(
            self.model_name, input_dict, output_names, callback=partial(asyn_callback, result_queue))
        # t3 = time.time()
        # print("predictor time cost total:{}, preprocess:{}, triton call:{}".format(
        #     t3-t1, t2-t1, t3-t2))


    def grpc_async_stream_infer():
        pass
    
    @staticmethod
    def postprocess(results):
        output0 = results.as_numpy('1346')
        quality_score = np.squeeze(output0, axis=-1)
        return quality_score.tolist()


if __name__ == "__main__":
    # 启动triton服务
    # tritonserver --model-repository={models} --http-port=8000 --grpc-port=8001
    # tritonserver --model-repository=triton-test --http-port=8000 --grpc-port=8001
    
    from triton_client import TritonHttpClient, TritonGrpcClient
    import time

    # echo "10.100.196.70 audioevent-predictor-default.mgtv" >> /etc/hosts
    # echo "10.100.196.70 face-quality-new-predictor-default.mgtv" >> /etc/hosts
    
    # predictor_host = "face-quality-new-predictor-default.mgtv"

    infer_type = 0
    
    if infer_type == 0:
        predictor_host = "localhost:8000"
        tritonclient = TritonHttpClient(predictor_host, concurrency=50)
    elif infer_type == 1:
        predictor_host = "localhost:8001"
        tritonclient = TritonGrpcClient(predictor_host, concurrency=50)
    elif infer_type == 2:
        # http直通triton
        predictor_host = "10.244.20.200:8080"
        tritonclient = TritonHttpClient(predictor_host, concurrency=50)
    elif infer_type == 3:
        # grpc直通triton
        predictor_host = "10.244.22.85:9000"
        predictor_host = "10.244.20.200:9000"
        tritonclient = TritonGrpcClient(predictor_host, concurrency=50)

    print(predictor_host)
    predictor = FaceQualityPrediction(tritonclient)

    images = []
    img = cv2.imread("data/1-Harry_Belafonte_1.jpg")
    batch_size = 1
    for i in range(batch_size):
        images.append(img)

    cnt = 11
    for i in range(cnt):
        if i==1:
            t1 = time.time()
        res = predictor.infer(images)
        if i % 10 == 0:
            print(">>", i, res)
    t2 = time.time()
    print("total time cost:", t2-t1, "average time cost:", (t2-t1)/cnt)
    print("| ", batch_size, "|", t2-t1, " | ", (t2-t1)/cnt, "s")

    # import concurrent
    # future_to_url = {}
    # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    #     for idx in range(cnt):
    #         future_to_url.update({executor.submit(predictor.infer, images): idx})    
    #     total_num = len(future_to_url.keys())    
    #     for future in concurrent.futures.as_completed(future_to_url):
    #         idx = future_to_url[future]
    #         res = future.result()   
    #         print(res)
