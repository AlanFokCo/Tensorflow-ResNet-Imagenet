FROM registry.cn-beijing.aliyuncs.com/alan_fok/lab_distributed:horovod-0.20.0-tf2.3.0-torch1.6.0-mxnet1.6.0.post0-py3.7-cuda10.1

COPY ./load_data.py /examples/elastic/tensorflow2/load_data.py
COPY ./train_distributed_elastic_with_horovod.py /examples/elastic/tensorflow2/train_distributed_elastic_with_horovod.py

RUN pip install sklearn -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple

USER root
