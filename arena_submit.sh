arena submit etjob \
    --name=resnet-imagenet-elastic-lab \
    --gpus=1 \
    --workers=4 \
    --max-workers=128 \
    --min-workers=1 \
    --image=xxx \
    "horovodrun --log-level DEBUG --verbose -np 4 --min-np 1 --max-np 128
              --host-discovery-script /etc/edl/discover_hosts.sh python examples/elastic/tensorflow2/train_distributed_elastic_with_horovod.py
              --train-dir=/dataset/ILSVRC2012_img_train/ --val-dir=/dataset/ILSVRC2012_img_val/"