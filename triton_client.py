# -*- encoding: utf-8 -*-
'''
@File    : triton_client.py
@Time    : 2022/05/25 15:52:51
@Author  : zhangbiao
@Contact : zhangbiao@mgtv.com
@Version : 1.0
@Desc    : tritonserver服务调用相关接口封装
'''

from tritonclient.utils import np_to_triton_dtype, raise_error
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient


class TritonHttpClient():
    """ 
    Note:
    -----
    None of the methods are thread safe. The object is intended to be 
    used by a single thread and simultaneously calling different methods
    with different threads is not supported and will cause undefined behavior.
    """

    def __init__(self, predictor_host, verbose=False, concurrency=1):
        """
        Args:
            predictor_host (str): The triton predictor host url.
            concurrency (int, optional): Must set the concurrency when using asynchronous inference. Defaults to 1.
            concurrency用于指定在与推理服务器进行通信时的并发请求数量
        """
        self.predictor_host = predictor_host
        self.verbose = verbose
        self.concurrency = concurrency
        self.triton_client = None

        self.init()

    def init(self):
        try:
            if self.triton_client is not None:
                self.triton_client.close()

            self.triton_client = httpclient.InferenceServerClient(url=self.predictor_host, verbose=self.verbose, concurrency=self.concurrency)

            if not self.triton_client.is_server_ready():
                self.triton_client = None
                raise_error("FAILED : is_server_ready")
        except Exception as e:
            print("triton channel creation failed: " + str(e))

    def infer(self, model_name, input_dict, output_name_list, is_async=False):
        """Call tritonclient synchronous or asynchronous inference.

        Args:
            model_name (str): The name of the model to run inference.
            input_dict (dict): A dict of input objects, each describing data for a input numpy data required by the model.
            output_name_list (list): A list of output tensor name, each describing how the output data must be returned.
            async (bool, optional): Run tritonclient synchronous or asynchronous inference. Defaults to False.

        Returns:
            InferAsyncRequest or InferResult Object: Rreturn a InferAsyncRequest Object, if async is True ; otherwise return a InferResult Object.

            InferAsyncRequest: The handle to the asynchronous inference request.
            InferResult: The object holding the result of the inference.

        Raises:
            InferenceServerException: If server fails to issue inference.
        """

        if self.triton_client is None:
            self.init()

        # construct InferInput/InferRequestedOutput object list
        triton_inputs, triton_outputs = self._request_generator(input_dict, output_name_list)
        
        if is_async:
            # 一种异步的推理请求方法，客户端会发送推理请求但不会等待服务器返回结果，而是立即返回一个futrue对象。可以在后续代码中通过future对象来获取推理结果，而不会阻塞当前线程。
            return self.triton_client.async_infer(model_name, triton_inputs, outputs=triton_outputs)
        else:
            # 一种同步的推理请求方法，客户端会发送推理请求并等待服务器返回结果，然后才会继续执行后续代码。这意味着在收到推理结果之前，当前线程会被阻塞。
            return self.triton_client.infer(model_name, triton_inputs, outputs=triton_outputs)

    def _request_generator(self, input_dict, output_list):
        inputs, outputs = [], []

        for key in input_dict:
            value = input_dict[key]
            infer_input = httpclient.InferInput(key, value.shape, np_to_triton_dtype(value.dtype))
            infer_input.set_data_from_numpy(value)
            inputs.append(infer_input)

        for name in output_list:
            outputs.append(httpclient.InferRequestedOutput(name))

        return inputs, outputs

# 长连接
class TritonGrpcClient():
    def __init__(self, predictor_host, port="8001", verbose=False, concurrency=1, timeout=30):
        self.predictor_host = predictor_host
        self.verbose = verbose
        self.concurrency = concurrency
        self.triton_client = None
        self.client_timeout = timeout
        self.grpc_compression_algorithm = None

        self.init()

    def init(self):

        self.triton_client = grpcclient.InferenceServerClient(
            url=self.predictor_host,
            verbose=self.verbose)
        
        if not self.triton_client.is_server_ready():
            self.triton_client = None
            raise_error("FAILED : is_server_ready")

    def infer(self, model_name: str, input_dict: dict, output_name_list: list):
        if self.triton_client is None:
            self.init()
        # construct InferInput/InferRequestedOutput object list
        triton_inputs, triton_outputs = self._request_generator(input_dict, output_name_list)

        return self.triton_client.infer(
            model_name=model_name,
            inputs=triton_inputs,
            outputs=triton_outputs,
            client_timeout=self.client_timeout)

        
    def async_infer(self, model_name: str, input_dict: dict, output_name_list: list, callback):
        if self.triton_client is None:
            self.init()

        # construct InferInput/InferRequestedOutput object list
        triton_inputs, triton_outputs = self._request_generator(
            input_dict, output_name_list)

        # Inference call
        self.triton_client.async_infer(model_name=model_name,
                                       inputs=triton_inputs,
                                       callback=callback,
                                       outputs=triton_outputs,
                                       client_timeout=self.client_timeout)

    def async_stream_infer(self,
                           model_name: str,
                           input_dict: dict,
                           output_name_list: list,
                           req_cnt: str,
                           seq_id: int,
                           start: bool,
                           end: bool
                           ):
        if self.triton_client is None:
            self.init()

        # construct InferInput/InferRequestedOutput object list
        triton_inputs, triton_outputs = self._request_generator(
            input_dict, output_name_list)

        # Inference call
        self.triton_client.async_stream_infer(model_name=model_name,
                                              inputs=triton_inputs,
                                              request_id=req_cnt,
                                              sequence_id=seq_id,
                                              sequence_start=start,
                                              sequence_end=end,
                                              outputs=triton_outputs)

    def _request_generator(self, input_dict, output_list):
        inputs, outputs = [], []

        for key in input_dict:
            value = input_dict[key]
            infer_input = grpcclient.InferInput(
                key, value.shape, np_to_triton_dtype(value.dtype))
            infer_input.set_data_from_numpy(value)
            inputs.append(infer_input)

        for name in output_list:
            outputs.append(grpcclient.InferRequestedOutput(name))

        return inputs, outputs