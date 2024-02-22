# -*- encoding: utf-8 -*-
'''

@File    : __main__.py
@Time    : 2022/05/25 11:20:38
@Author  : zhangbiao
@Contact : zhangbiao@mgtv.com
@Version : 1.0
'''


import kserve
import argparse
from .transformer import Transformer

DEFAULT_MODEL_NAME = "model"

parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME,
                    help='The name that the model is served under.')
parser.add_argument('--predictor_host', help='The URL for the model predict function', required=True)

args, _ = parser.parse_known_args()

if __name__ == "__main__":
    transformer = Transformer(args.model_name, predictor_host=args.predictor_host)
    server = kserve.ModelServer()
    server.start(models=[transformer])
