#!/bin/bash

. ./path.sh || exit 1;

gpu=0

export NCCL_DEBUG=INFO
stage=1
stop_stage=2

num_nodes=1
node_rank=0

nj=16
dict=exp/20210601_u2++_conformer_exp/units.txt

data_type=raw

# data setting
train_set='aishell/train'
dstore_name='aishell'

test_set='aishell/test'
test_dataset_name='aishell'

# model setting
train_config=conf/train.yaml
cmvn=true
dir='exp/20210601_u2++_conformer_exp'
decode_checkpoint="$dir/final.pt"

# knn setting
decode_modes="knn_ctc"
dstore_dir="datastore/aishell_dstore"
use_null_mask=True # build datastore with skip-blank strategy
decode_skip_blank=True # decoding with skip-blank strategy
dstore_size=1798000 # dstore_size=13000001
lmbda=0.45 # interpolate weight, adjust to the dataset
thr=0.0 # threshold of CTC pseudo label, default = 0
knn_temp=1.0 # temperature, defalut = 1  
k=1024 # k neighbours

# no use
scale_lmbda=False
scale_lmbda_temp=1

. tools/parse_options.sh || exit 1;

# build index on the training set
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  decoding_chunk_size=
  ctc_weight=0.5
  reverse_weight=0.0
  for mode in ${decode_modes}; do
  {
    test_dir=$dir/test_${mode}_${dstore_size}_${dstore_name}_on_${test_dataset_name}
    mkdir -p $test_dir
    mkdir -p $dstore_dir
    python wenet_knn_ctc.py --gpu $gpu \
      --mode $mode \
      --config $dir/train.yaml \
      --data_type $data_type \
      --test_data data/$train_set/data.list \
      --checkpoint $decode_checkpoint \
      --beam_size 10 \
      --batch_size 1 \
      --penalty 0.0 \
      --dict $dict \
      --ctc_weight $ctc_weight \
      --reverse_weight $reverse_weight \
      --result_file $test_dir/text \
      ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size} \
      --build_index \
      --dstore_dir $dstore_dir\
      --dstore_size $dstore_size \
      --lmbda $lmbda \
      --thr $thr \
      --use_null_mask $use_null_mask
  } &
  done
  wait
fi
# knn_ctc decoding
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then

  # no use
  decoding_chunk_size=
  ctc_weight=0.5
  reverse_weight=0.0

  for mode in ${decode_modes}; do
  {
    test_dir=$dir/test_${mode}_${dstore_size}_${dstore_name}_on_${test_dataset_name}
    mkdir -p $test_dir
    python wenet_knn_ctc.py  --gpu $gpu \
      --mode $mode \
      --config $dir/train.yaml \
      --data_type $data_type \
      --test_data data/$test_set/data.list \
      --checkpoint $decode_checkpoint \
      --beam_size 10 \
      --batch_size 1 \
      --penalty 0.0 \
      --dict $dict \
      --ctc_weight $ctc_weight \
      --reverse_weight $reverse_weight \
      --result_file $test_dir/${test_dataset_name}_text \
      ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size} \
      --knn \
      --dstore_dir $dstore_dir\
      --dstore_size $dstore_size \
      --lmbda $lmbda \
      --knn_temp $knn_temp \
      --k $k \
      --decode_skip_blank $decode_skip_blank\
      --scale_lmbda $scale_lmbda\
      --scale_lmbda_temp $scale_lmbda_temp
    sed -i "s|▁| |g" $test_dir/${test_dataset_name}_text
    python tools/compute-wer.py --char=1 --v=1 \
      data/$test_set/text $test_dir/${test_dataset_name}_text > $test_dir/${test_dataset_name}_wer
    tail $test_dir/${test_dataset_name}_wer
  } &
  done
  wait
fi

# vanilla CTC decoding
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  decode_modes="ctc_greedy_search"

  # no use
  decoding_chunk_size=
  ctc_weight=0.5
  reverse_weight=0.0

  for mode in ${decode_modes}; do
  {
    test_dir=$dir/test_${mode}_${test_dataset_name}
    mkdir -p $test_dir
    python wenet_knn_ctc.py  --gpu $gpu \
      --mode $mode \
      --config $dir/train.yaml \
      --data_type $data_type \
      --test_data data/$test_set/data.list \
      --checkpoint $decode_checkpoint \
      --beam_size 10 \
      --batch_size 1 \
      --penalty 0.0 \
      --dict $dict \
      --ctc_weight $ctc_weight \
      --reverse_weight $reverse_weight \
      --result_file $test_dir/${test_dataset_name}_text \
      ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size} \
      --knn \
      --dstore_dir $dstore_dir\
      --dstore_size $dstore_size \
      --lmbda $lmbda

    sed -i "s|▁| |g" $test_dir/${test_dataset_name}_text
    python tools/compute-wer.py --char=1 --v=1 \
      data/$test_set/text $test_dir/${test_dataset_name}_text > $test_dir/${test_dataset_name}_wer
    tail $test_dir/${test_dataset_name}_wer
  } &
  done
  wait
fi