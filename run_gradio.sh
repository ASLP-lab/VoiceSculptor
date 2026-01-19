#!/bin/bash
# 替换对应的llasa_ckpt_path和xcodec_ckpt_path
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python gradio/webui.py \
    --llasa_ckpt_path "" \
    --xcodec_ckpt_path "" \
    --device "cuda"