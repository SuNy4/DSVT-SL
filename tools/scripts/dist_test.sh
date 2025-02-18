#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

python3 test.py ${PY_ARGS} # --launcher pytorch  #-m torch.distributed.launch --nproc_per_node=${NGPUS} test.py --launcher pytorch ${PY_ARGS}

