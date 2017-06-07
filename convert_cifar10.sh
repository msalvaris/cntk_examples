#!/usr/bin/env bash

set -e
set -o pipefail

IMAGE_NAME=$1
CIFAR_DATA=$2

docker run --rm -v $CIFAR_DATA:$CIFAR_DATA -w $CIFAR_DATA $IMAGE_NAME /bin/bash -c "source /cntk/activate-cntk; python cifar_data_processing.py --datadir $CIFAR_DATA"