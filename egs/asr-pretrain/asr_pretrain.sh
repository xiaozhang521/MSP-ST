
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup fairseq-hydra-train \
    --config-dir ./ \
    --config-name config  >log &
tail -f log

