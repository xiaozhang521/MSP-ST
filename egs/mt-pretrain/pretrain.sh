set -e
PRETRAIN= # the path to downloaded MBART model
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
output_dir= # path to save model
mkdir -p $output_dir
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup fairseq-train #path to data \
  --encoder-normalize-before --decoder-normalize-before \
  --arch mbart_large --layernorm-embedding \
  --task translation_from_pretrained_bart \
  --source-lang en_XX --target-lang de_DE \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt --lr 1e-04 --warmup-updates 10000 --max-update 140000 \
  --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
  --max-tokens 1024 --update-freq 2 \
  --reset-optimizer \
  --save-interval 1 --save-interval-updates 2000 --keep-interval-updates 5 --no-epoch-checkpoints \
  --seed 222 --log-format simple --log-interval 200 \
  --restore-file $PRETRAIN \
  --langs $langs \
  --fp16 \
  --log-interval 100 \
  --ddp-backend c10d \
  --save-dir $output_dir > $output_dir/train.log &
tail -f $output_dir/train.log
