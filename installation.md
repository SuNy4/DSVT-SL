CUMM 설치
1. cumm git clone하기
2. export CUMM_CUDA_ARCH_LIST="8.7"
3. pip install -e .
Spconv 설치
1. cumm 디렉토리에서 spconv 클론하기
2. v2.2.0으로 checkout
3. sudo vi pyproject.toml 로 cumm 요구사항 지우기
4. pip install -e .

Torch-scatter 설치

pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu114.html --no-cache-dir