## Running DPR repo model
```
torchrun --standalone --nnodes=1 --nproc_per_node=5 mindpr/encode_passages_sharded.py /data/local/DPR_runs/nq/dpr_biencoder.34 '/common/home/jl2529/repositories/DPR/downloads/data/wikipedia_split/psgs_w100_shard*.tsv' /data/local/DPR_runs/mindpr_runs/dpr_embs --batch_size 2048 --num_workers 2 --gpus 0,1,2,3,4  # 2048 -> ~31G memory
```
