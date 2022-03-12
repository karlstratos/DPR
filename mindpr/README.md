## NQ
```
./mindpr/run.sh 0,1,2,3,4,5,6,7
```
or explicitly
```
torchrun --standalone --nnodes=1 --nproc_per_node=8 mindpr/train.py /data/local/DPR_runs/mindpr_runs/nq/model downloads/data/retriever/nq-train.json downloads/data/retriever/nq-dev.json --num_warmup_steps 1237 --num_workers 2 --gpus 0,1,2,3,4,5,6,7   # 2h6m
torchrun --standalone --nnodes=1 --nproc_per_node=8 mindpr/encode_passages_sharded.py /data/local/DPR_runs/mindpr_runs/nq/model '/common/home/jl2529/repositories/DPR/downloads/data/wikipedia_split/psgs_w100_shard*.tsv' /data/local/DPR_runs/mindpr_runs/nq --batch_size 2048 --num_workers 2 --gpus 0,1,2,3,4,5,6,7
python mindpr/search.py /data/local/DPR_runs/mindpr_runs/nq/model downloads/data/retriever/qas/nq-test.csv '/data/local/DPR_runs/mindpr_runs/nq/psgs_w100_shard*.pickle' /data/local/DPR_runs/mindpr_runs/nq/out.json downloads/data/wikipedia_split/psgs_w100.tsv --gpu 0
```

## Running DPR repo model
```
torchrun --standalone --nnodes=1 --nproc_per_node=5 mindpr/encode_passages_sharded.py /data/local/DPR_runs/nq/dpr_biencoder.34 '/common/home/jl2529/repositories/DPR/downloads/data/wikipedia_split/psgs_w100_shard*.tsv' /data/local/DPR_runs/mindpr_runs/dpr_embs --batch_size 2048 --num_workers 2 --gpus 0,1,2,3,4  # 2048 -> ~31G memory, 1h 35m
python mindpr/search.py /data/local/DPR_runs/nq/dpr_biencoder.34 downloads/data/retriever/qas/nq-test.csv '/data/local/DPR_runs/mindpr_runs/dpr_embs/psgs_w100_shard*.pickle' /data/local/DPR_runs/mindpr_runs/dpr_embs/out.json downloads/data/wikipedia_split/psgs_w100.tsv --gpu 0  # 20m
```

Results (matches)
```
k=1     k=5     k=20    k=100   filename        num_queries
46.0%   68.2%   79.1%   86.3%   nq-test.csv     3610
```

## Debugging validation (average rank)
```
python mindpr/validate.py --batch_size_val 10 --num_hard_negatives 3 --num_other_negatives 4 --subbatch_size 7 --gpus 0 --data_val downloads/data/retriever/nq-train10.json
torchrun --standalone --nnodes=1 --nproc_per_node=2 mindpr/validate.py --batch_size_val 5 --num_hard_negatives 3 --num_other_negatives 4 --subbatch_size 7 --gpus 0,1 --data_val downloads/data/retriever/nq-train10.json
torchrun --standalone --nnodes=1 --nproc_per_node=8 mindpr/validate.py --batch_size_val 512 --num_hard_negatives 30 --num_other_negatives 30 --subbatch_size 1024 --gpus 0,1,2,3,4,5,6,7 --num_workers 8  # avg rank ~15211/49584 per process, runtime 2m (setting num workers 2 vs 8 makes no difference, setting 0 makes it 2.5m)
```

## Debugging training
```
python mindpr/train.py /tmp/model downloads/data/retriever/nq-train10.json downloads/data/retriever/nq-train10.json --batch_size 10 --num_warmup_steps 1 --lr 1e-4 --epochs 5 --gpus 0 --start_epoch_val 0 --log_result_step 1
```
vs
```
CUDA_VISIBLE_DEVICES=0 python train_dense_encoder.py train=biencoder_nq train_datasets=[nq_train10] dev_datasets=[nq_train10] train.batch_size=10 train.dev_batch_size=10 train.num_train_epochs=5 output_dir=/data/local/DPR_runs/toy  train.learning_rate=1e-4 train.warmup_steps=1 train.log_batch_step=1 train.skip_saving=True val_av_rank_start_epoch=0
```

## Distributed
```
torchrun --standalone --nnodes=1 --nproc_per_node=5 mindpr/train.py /tmp/model downloads/data/retriever/nq-train10.json downloads/data/retriever/nq-train10.json --batch_size 1 --batch_size_val 1 --num_warmup_steps 1 --lr 1e-4 --epochs 5 --gpus 0,1,2,3,4 --start_epoch_val 0 --log_result_step 1
```
vs
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m torch.distributed.launch --nproc_per_node=5 train_dense_encoder.py train=biencoder_nq train_datasets=[nq_train10] dev_datasets=[nq_train10] train.batch_size=1 train.dev_batch_size=1 train.num_train_epochs=5 output_dir=/data/local/DPR_runs/toy  train.learning_rate=1e-4 train.warmup_steps=1 train.log_batch_step=1 train.skip_saving=True val_av_rank_start_epoch=0
```
Val different but loss same
