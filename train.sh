#!/usr/bin/env bash

GPUS=$1
while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
 python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
     train.py data/cifar100 \
     --dataset cifar100 \
     --config configs/cifar/cnn.yaml \
     --model resnet18 \
     --teacher swin_tiny_patch4_window7_224 \
     --teacher-pretrained pretrained/swin_tiny_patch4_window7_224_cifar100.pth \
     --num-classes 100 \
     --distiller rsd \

# python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     train.py data/cifar100 \
#     --dataset cifar100 \
#     --config configs/cifar/cnn.yaml \
#     --model resnet18 \
#     --teacher vit_small_patch16_224 \
#     --teacher-pretrained pretrained/vit_small_patch16_224_cifar100.pth \
#     --num-classes 100 \
#     --distiller rsd \

# python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     train.py data/cifar100 \
#     --dataset cifar100 \
#     --config configs/cifar/cnn.yaml \
#     --model resnet18 \
#     --teacher mixer_b16_224 \
#     --teacher-pretrained pretrained/mixer_b16_224_cifar100.pth \
#     --num-classes 100 \

# python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     train.py data/cifar100 \
#     --dataset cifar100 \
#     --config configs/cifar/cnn.yaml \
#     --model mobilenetv2_100 \
#     --teacher swin_tiny_patch4_window7_224 \
#     --teacher-pretrained pretrained/swin_tiny_patch4_window7_224_cifar100.pth \
#     --num-classes 100 \
#     --distiller rsd \

# python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     train.py data/cifar100 \
#     --dataset cifar100 \
#     --config configs/cifar/cnn.yaml \
#     --model mobilenetv2_100 \
#     --teacher vit_small_patch16_224 \
#     --teacher-pretrained pretrained/vit_small_patch16_224_cifar100.pth \
#     --num-classes 100 \
#     --distiller rsd \

# python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     train.py data/cifar100 \
#     --dataset cifar100 \
#     --config configs/cifar/cnn.yaml \
#     --model mobilenetv2_100 \
#     --teacher mixer_b16_224 \
#     --teacher-pretrained pretrained/mixer_b16_224_cifar100.pth \
#     --num-classes 100 \
#     --distiller rsd \


# python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     train.py data/cifar100 \
#     --dataset cifar100 \
#     --config configs/cifar/vit_mlp.yaml \
#     --model deit_tiny_patch16_224 \
#     --teacher convnext_tiny \
#     --teacher-pretrained pretrained/convnext_tiny_cifar100.pth \
#     --num-classes 100 \
#     --distiller rsd \

# python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     train.py data/cifar100 \
#     --dataset cifar100 \
#     --config configs/cifar/vit_mlp.yaml \
#     --model deit_tiny_patch16_224 \
#     --teacher mixer_b16_224 \
#     --teacher-pretrained pretrained/mixer_b16_224_cifar100.pth \
#     --num-classes 100 \
#     --distiller rsd \

# python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     train.py data/cifar100 \
#     --dataset cifar100 \
#     --config configs/cifar/vit_mlp.yaml \
#     --model swin_pico_patch4_window7_224 \
#     --teacher convnext_tiny \
#     --teacher-pretrained pretrained/convnext_tiny_cifar100.pth \
#     --num-classes 100 \
#     --distiller rsd \

# python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     train.py data/cifar100 \
#     --dataset cifar100 \
#     --config configs/cifar/vit_mlp.yaml \
#     --model swin_pico_patch4_window7_224 \
#     --teacher mixer_b16_224 \
#     --teacher-pretrained pretrained/mixer_b16_224_cifar100.pth \
#     --num-classes 100 \
#     --distiller rsd \


# python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     train.py data/cifar100 \
#     --dataset cifar100 \
#     --config configs/cifar/vit_mlp.yaml \
#     --model resmlp_12_224 \
#     --teacher convnext_tiny \
#     --teacher-pretrained pretrained/convnext_tiny_cifar100.pth \
#     --num-classes 100 \
#     --distiller rsd \

# python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     train.py data/cifar100 \
#     --dataset cifar100 \
#     --config configs/cifar/vit_mlp.yaml \
#     --model resmlp_12_224 \
#     --teacher swin_tiny_patch4_window7_224 \
#     --teacher-pretrained pretrained/swin_tiny_patch4_window7_224_cifar100.pth \
#     --num-classes 100 \
#     --distiller rsd \