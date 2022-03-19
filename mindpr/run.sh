#!/bin/bash
# chmod 777 run.sh
# ./mindpr/run.sh 0,1,2,3,4,5,6,7
# ./mindpr/run.sh 0,1,2,3,4,5,6,7 --use_my_loader
# ./mindpr/run.sh 0,1,2,3,4,5,6,7 --seed=12345

OUTDIR="/data/local/DPR_runs/mindpr_runs/nq"
mkdir -p OUTDIR
model="${OUTDIR}/model"
pembs="${OUTDIR}/psgs_w100_shard*.pickle"
outfile="${OUTDIR}/out.json"

DATADIR="/common/home/jl2529/repositories/DPR/downloads/data/"
data_train="${DATADIR}/retriever/nq-train.json"
data_val="${DATADIR}/retriever/nq-dev.json"
data_test="${DATADIR}/retriever/qas/nq-test.csv"
data_wiki_whole="${DATADIR}/wikipedia_split/psgs_w100.tsv"
data_wiki_shards="${DATADIR}/wikipedia_split/psgs_w100_shard*.tsv"

gpus=$1
commas="${gpus//[^,]}"
num_commas="${#commas}"
num_gpus="$((num_commas+1))"

optional1=${2:-}
optional2=${3:-}
optional3=${4:-}

train="torchrun --standalone --nnodes=1 --nproc_per_node=${num_gpus} mindpr/train.py ${model} ${data_train} ${data_val} --num_warmup_steps 1237 --num_workers 2 --gpus ${gpus} ${optional1} ${optional2} ${optional3}"
encode="torchrun --standalone --nnodes=1 --nproc_per_node=${num_gpus} mindpr/encode_passages_sharded.py ${model} '${data_wiki_shards}' ${OUTDIR} --batch_size 2048 --num_workers 2 --gpus ${gpus}"
search="python mindpr/search.py ${model} ${data_test} '${pembs}' ${outfile} ${data_wiki_whole} --gpu 0"

echo $train
eval $train

echo $encode
eval $encode

echo $search
eval $search
