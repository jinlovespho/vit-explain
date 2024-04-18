
DATA_ARGS="
    --dataset imagenet \

"

MODEL_ARGS="
    --model_name deit \

"

ETC_ARGS="
    --head_fusion mean \
    --discard_ratio 0.7 \
"

CUDA_VISIBLE_DEVICES=0 python vit_explain.py ${DATA_ARGS} ${MODEL_ARGS} ${ETC_ARGS}

