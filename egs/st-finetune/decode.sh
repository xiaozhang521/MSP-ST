#! /bin/bash

gpu_num=1

data_dir=data_tgt_lcrm
#test_subset=(test_p500_update)
#test_subset=(test_p0.9_p500)
test_subset=(tst-COMMON)
#test_subset=(dev)

exp_name=st_sead

n_average=10
beam_size=8
len_penalty=1.2
max_tokens=10000
dec_model=checkpoint_best.pt
device=3
export CUDA_VISIBLE_DEVICES=${device}
cmd="./device_run.sh
    --stage 2
    --stop_stage 2
    --gpu_num ${gpu_num}
    --exp_name ${exp_name}
    --n_average ${n_average}
    --beam_size ${beam_size}
    --len_penalty ${len_penalty}
    --max_tokens ${max_tokens}
    --dec_model ${dec_model}
    "

if [[ -n ${data_dir} ]]; then
    cmd="$cmd --data_dir ${data_dir}"
fi
if [[ -n ${test_subset} ]]; then
    test_subset=`echo ${test_subset[*]} | sed 's/ /,/g'`
    cmd="$cmd --test_subset ${test_subset}"
fi

echo $cmd
eval $cmd
