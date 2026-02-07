###########
nnodes=8
batch="-b 8 --eval-batch-size 8"
epochs="--epochs 50"
minibatches="--num_minibatches 1000 --num_eval_minibatches 25"
ngpus="8"
model="--module models.gptn.gpus=$ngpus --block_size 512 --n_embd 768 --n_head 12 --n_layer 8 --config_path models/gptn/gpus=$ngpus/mp_conf.json"
d="wikitext-103-v1"
outdir=""$d"_gptn_512_768_12_8_b8"
lr="--lr 3e-4 --lr_warmup --optimizer nadamw"
logtb="--log_tb --tb_dir ./runs"
cg="--clip_grad 10"
basecmdstr="python main_with_runtime.py $model $batch -d $d --master_addr localhost --distributed_backend gloo 
$lr $epochs $minibatches $cg $logtb --recompute --lr_policy cosine"

# Ours 
method="--momentum 0.99 --optimizer nadamw"   
expname="$outdir/gpus=$ngpus/ours/"
ckptdir="$d/$expname"    
mkdir -p $ckptdir   # create checkpoint directory if not exists
cmdstr="$basecmdstr $method --exp_name $expname --checkpoint_dir $ckptdir"
for rank in $(seq 0 $(($ngpus-1))); do
    cmd="$cmdstr --rank $rank --local_rank $(($rank % $nnodes)) &"
    echo $cmd
    eval $cmd
done
wait

# gpipe
basecmdstr="python sync_main.py $model $batch -d $d --master_addr localhost --distributed_backend nccl
$lr $epochs $minibatches $cg $logtb --recompute --lr_policy cosine --optimizer adamw"

# gpipe
method="--sync_schedule gpipe --num_microbatches 4"
expname="$outdir/gpus=$ngpus/gpipe/"
ckptdir="$d/$expname"    
mkdir -p $ckptdir   # create checkpoint directory if not exists
cmdstr="$basecmdstr $method --exp_name $expname --checkpoint_dir $ckptdir"
for rank in $(seq 0 $(($ngpus-1))); do
    cmd="$cmdstr --rank $rank --local_rank $(($rank % $nnodes)) &"
    echo $cmd
    eval $cmd
done
wait
