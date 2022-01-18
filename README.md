# Tensorflow-ResNet-Imagenet

Single-machine and Horovod elastic distributed training of ImageNet using ResNet of Tensorflow 2.0.

Dataset Download Link:

Run Demo of Single-machine by:

```shell
python 	train_stand_alone.py
		--epochs=100
		--batch-size=128
		--learning-rate=0.001
		--train-dir=/dataset/ILSVRC2012_img_train/ 
		--val-dir=/dataset/ILSVRC2012_img_val/
```

Run Demo of Horovod elastic distributed training by:

```shell
kubectl apply -f elastic_training_job.yaml
```

or

```shell
./arena_submit.sh
```

You can install arena from https://github.com/kubeflow/arena, and the documentation link is https://arena-docs.readthedocs.io/en/latest.

We can submit horovod elastic training job by Elastic Training Operator(https://github.com/AliyunContainerService/et-operator).

Et-operator provides a set of Kubernetes Custom Resource Definition that makes it easy to run horovod or AIACC elastic training in kubernetes. After submit a training job, you can scaleIn and scaleOut workers during training on demand, which can make your training job more elasticity and efficient.
