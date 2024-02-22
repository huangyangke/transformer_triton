[TOC]

# [Kserve Transformer Demo](<https://git.imgo.tv/zhangbiao/transformer_demo.git>)



## 简介

这是一个人脸质量检测的算法接口服务(能力)，主要用于人脸识别等一些场景，服务支持多batch调用。

此外，还是一个Kserve Transformer服务的demo, 底层调用的模型服务为Triton inference server。




## 规范
提供以下规范示例：

- [x] 文档结构及编码建议
- [x] 对外接口IO数据规范
- [x] Transformer规范
- [x] 基础镜像介绍
- [x] 部署交付标准

### 1) 文档结构及编码建议
- 文档结构简洁明了，说明文档需对服务功能及注意事项进行简要说明。
```
.
├── readme.md             # 说明文档
├── __main__.py           # 服务入口
└── utils.py              # 工具方法
├── transformer.py        # transformer主流程
├── prediction.py         # 算法前后处理及调用流程
├── triton_client.py      # tritonclient调用封装
├── kserver_client.py     # transformer服务测试脚本
├── data                  # 存放本地测试数据
├── common_data_type      # 数据对象解析封装库
```
- 编码建议：

  1）文件、重要类、属性、方法及重要逻辑段需添加必要注释说明；

  2）代码层次逻辑尽量简洁，避免复杂对象封装，异常及错误需返回给客户端；

  3）禁止使用多进程，建议少用多线程。


### 2) 对外接口IO数据规范
对外接口IO数据规范以项目 [common_data_type](<http://git.hunantv.com/huwei/common_data_type.git/>)为标准，
所有输入和输出的json或python dict需通过Request/MultiRequest和Response/MultiReponse来进行解析和封装。

### 3) Transformer规范
Transformer是一个位于客户端和模型服务之间的InferenceService组件, Transformer使用户可以在模型推理流程之前定义预处理和后期处理步骤。
Kserve提供了Transformer功能，可以通过triton client http/grpc方式来访问triton server。
此外, 对于较复杂的算法场景,如人脸识别, Transformer可以将多个模型组合起来, 同步或异步调用triton模型服务。

> 注意：这里的Transformer不是深度学习里的模型名字，而是Kseve的一种功能。


自定义Transformer注意事项
- 1）transformer主流程，见示例transformer.py，需要继承kfserving.KFModel的类，并重载predic、postprocess、preprocess方法；
  - preprocess：进行客户端数据接收处理
  - postprocess：进行客户端返回结果封装
  - predict：放置算法主逻辑流程，用来调用各个子算法推理接口，组合并拼装推理结果
- 2）入口文件__main__.py，统一文件名，见示例__main__.py
- 3）triton服务调用，见示例kserver_client.py， 包括http/grpc调用方式，支持异步调用
> 注意http/grpc调用不能使用多线程，对于并发任务需求可以开启异步调用is_async=True, 并结合Triton模型服务实际情况设置并发访问量concurrency。

本地Transformer服务运行命令
```shell
python -m <your_model_transformer_dir> --model_name <your_model_name> --predictor_host <kserve_host> --http_port <server port>
```

测试通过后即可将代码和环境打包成Transformer服务镜像
```shell
docker commit -m "<commit info>" -a "<author>" <container name> <image name>:<image tag>
```

### 4) 基础镜像介绍
这里指kserve的基础镜像，在kserve官方推荐镜像上安装了一些常用包，重新打包成一个新的镜像
```shell
docker pull ai-image.imgo.tv/serving/transformer_base_image:v1.5
```


### 5) 部署交付标准
需要交付给@朱彦的内容：
- Transformer服务镜像及服务启动脚本；
- Triton的模型文件、配置文件、插件及环境包等；

交付表单填写
```
Transformer：
服务名称(小写英文简称):
算力类型(CPU/GPU): 
服务镜像(请统一上传到ai-image.imgo.tv/serving/xx):
运行目录及启动命令：
资源需求：如2C2G(2个CPU核心以及2GB内存)
理论单位请求时间：如10ms
QPS预估：如10

Triton
Triton版本：2.15.0, NGC 21.10-py3
S3模型地址：
对外端口(HTTP/GRPC):
单容器模型全量load所需显存:
GPU驱动版本要求：
GPU型号要求：
常规资源需求：如2C2G(2个CPU核心以及2GB内存)
理论单位请求时间：如10ms
```

## QuickStart

### 1）本地Tritonsever启动及验证
```shell
# 启动本地tirtonserver模型服务
docker run --gpus=1 --name=triton_server_demo  --rm -p8000:8000 -p8001:8001 -p8002:8002 -v /home/ws/ws_benchmark/face_quality/FaceQuality/model_repo:/models nvcr.io/nvidia/tritonserver:21.10-py3 tritonserver --model-repository=/models

# 验证
curl -v localhost:8000/v2/health/ready
```

### 2）本地Transformer服务编码及部署验证
```shell
docker pull ai-image.imgo.tv/serving/transformer_base_image:v1.5
# 运行Transformer容器
docker run --gpus=all -it --rm --name=transformer_demo  --net=host -v /home/:/home/  ai-image.imgo.tv/serving/transformer_base_image:v1.5

# 拉取Transformer示例
git clone --recursive https://git.imgo.tv/zhangbiao/transformer_demo.git

# 运行Transformer服务: 服务名称为 facequality
# 方式1 http调用triton服务,具体同步/异步调用见transformer.py文件
python -m facequality_transformer --model_name facequality --predictor_host 0.0.0.0:8000 --http_port 8080
# 方式2 grpc调用triton服务,具体同步/异步调用见transformer.py文件
python -m facequality_transformer --model_name facequality --predictor_host 0.0.0.0:8001 --http_port 8080

# 测试
python kserver_client.py
```

### 3）打包Transformer服务镜像及测试
```shell
  # 打包
  docker commit -m "face quality transformer" -a "zhangbiao" transformer_demo cs-hub.imgo.tv/kubeflow/facequality_transformer:v1.2

  # 测试-启动镜像
  docker run --gpus=all -it --rm --name=facequality_transformer  --net=host -v /home/:/home/   cs-hub.imgo.tv/kubeflow/facequality_transformer:v1.2 

  # 测试-启动Transformer服务
  cd /workspace
  python -m facequality_transformer --model_name facequality --predictor_host 0.0.0.0:8000 --http_port 8080
  # 或者
  python -m facequality_transformer --model_name facequality --predictor_host 0.0.0.0:8001 --http_port 8080

  # 测试
  python kserver_client.py

  # 测试成功，push到仓库
  docker push cs-hub.imgo.tv/kubeflow/facequality_transformer:v1.2
```
## 相关环境

### 1) Tensorrt环境

- jupyter: http://10.200.16.100:8088/lab

- netron模型解析工具
```
  # 指令
  netron <your_model_file> --host 0.0.0.0
  # 查看结果: 
  http://10.200.16.100:18080
```

- docker镜像 
```shell
docker pull cs-hub.imgo.tv/kfserving/trts:8.0.3.4
```


### 2) TritonServer环境

  triton server定制镜像
  ```shell
  # 官方镜像
  docker pull nvcr.io/nvidia/tritonserver:21.10-py3
  ```


### 3) Kserve Transformer环境

  Kserve Transformer定制镜像
  ```shell
  docker pull ai-image.imgo.tv/serving/transformer_base_image:v1.5
  ```


## tritonclient http/grpc调用案例
### 1）tritonclient http同步调用案例 
```python
# TritonHttpClient同步调用方法
def infer(self, images):
  input_data = self.preprocess(images)
  input_dict = {"input.1": input_data}
  output_names = ['1346']

  results = self.tritonclient.infer(self.model_name, input_dict, output_names)
  return self.postprocess(results)

predictor_host="127.0.0.1:8000"
tritonclient = TritonHttpClient(predictor_host, concurrency=100)
predictor = FaceQualityPrediction(tritonclient)
scores = predictor.infer(images)
```

###  2）tritonclient http异步调用案例 
```python
# TritonHttpClient异步调用方法
def async_infer(self, images):
  input_data = self.preprocess(images)
  input_dict = {"input.1": input_data}
  output_names = ['1346']

  results = self.tritonclient.infer(self.model_name, input_dict, output_names, is_async=True)
  return results

predictor_host="127.0.0.1:8000"
tritonclient = TritonHttpClient(predictor_host, concurrency=100)
predictor = FaceQualityPrediction(tritonclient)

async_requests = []
sent_count = 500
for i in range(sent_count):
  async_requests.append(predictor.async_infer(images))

scores = []
for j, async_request in enumerate(async_requests):
  res = predictor.postprocess(async_request.get_result())
  scores.extend(res)

```

###  3）tritonclient grpc同步调用同http同步调用
###  4）tritonclient grpc异步调用
```python
# TritonHttpClient异步调用方法
def grpc_async_infer(self, images:list, result_queue:Queue):
  input_data = self.preprocess(images)
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
    
  self.tritonclient.async_infer(self.model_name, input_dict, output_names, 
                                callback=partial(asyn_callback, result_queue))


tritonclient = TritonGrpcClient(predictor_host)
predictor = FaceQualityPrediction(tritonclient)

result_queue = Queue()

sent_count = 500
for i in range(sent_count):
  predictor.grpc_async_infer(images, result_queue)

responses = []
processed_count = 0
try:
  while processed_count < sent_count:
    (results, error) = result_queue.get(timeout=2) #单位s
    processed_count += 1
    if error is not None:
      print("grpc asyn inference failed: " + str(error))
    else:
      responses.append(results)
except Exception as e:
  print(e)
    
```

