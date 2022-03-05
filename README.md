Do not fetch new commits, as they tend to break the code.

# Setup

```
conda create --name DPR python=3.6
conda activate DPR
pip install .
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html  # Need this for CUDA 11
python -m spacy download en_core_web_sm
python dpr/data/download_data.py --resource data.wikipedia_split.psgs_w100
python dpr/data/download_data.py --resource data.retriever.nq-train
python dpr/data/download_data.py --resource data.retriever.nq-dev
python dpr/data/download_data.py --resource data.retriever.qas.nq-test
python make_toy_data.py downloads/data/retriever/nq-train.json downloads/data/retriever/nq-train2.json --N 2
python make_toy_data.py downloads/data/retriever/nq-train.json downloads/data/retriever/nq-train10.json --N 10
```

Put the *absolute* path to the toy dataset in `conf/datasets/encoder_train_default.yaml`, like this (so if you change the machine, you'll have to change the path here):
```
nq_train2:
  _target_: dpr.data.biencoder_data.JsonQADataset
  file: /common/home/jl2529/repositories/DPR/downloads/data/retriever/nq-train2.json

nq_train10:
  _target_: dpr.data.biencoder_data.JsonQADataset
  file: /common/home/jl2529/repositories/DPR/downloads/data/retriever/nq-train10.json
```

Check if training works on the toy dataset. Annoyingly, the code loads a model from the output path, so you must make sure checkpoints are gone if you want to train from scratch.
```
rm -rf /data/local/DPR_runs/toy
CUDA_VISIBLE_DEVICES=0 python train_dense_encoder.py train=biencoder_nq train_datasets=[nq_train10] dev_datasets=[nq_train10] train.batch_size=10 train.dev_batch_size=10 train.num_train_epochs=3 output_dir=/data/local/DPR_runs/toy
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_dense_encoder.py train=biencoder_nq train_datasets=[nq_train10] dev_datasets=[nq_train10] train.batch_size=2 train.dev_batch_size=2 train.num_train_epochs=3 output_dir=/data/local/DPR_runs/toy
```

# Evaluating Pretrained Models

```
python dpr/data/download_data.py --resource checkpoint.retriever.single.nq.bert-base-encoder
mkdir /data/local/DPR_runs/pretrained_nq/
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=0 python generate_dense_embeddings.py model_file=/common/home/jl2529/repositories/DPR/downloads/checkpoint/retriever/single/nq/bert-base-encoder.cp ctx_src=dpr_wiki shard_id=${i} num_shards=16 out_file='/data/local/DPR_runs/pretrained_nq/embs' batch_size=1024; done;
CUDA_VISIBLE_DEVICES=0 python dense_retriever.py model_file=/common/home/jl2529/repositories/DPR/downloads/checkpoint/retriever/single/nq/bert-base-encoder.cp qa_dataset=nq_test ctx_datatsets=[dpr_wiki] encoded_ctx_files=[\"/data/local/DPR_runs/pretrained_nq/embs*\"] out_file=/data/local/DPR_runs/pretrained_nq/out.json
python evaluate.py /data/local/DPR_runs/pretrained_nq/out.json
```

This yields the following results:
```
k=1     k=5     k=20    k=100   filename        num_queries
46.4%   68.6%   80.1%   86.1%   out.json        3610
```

# Training

```
rm -rf /data/local/DPR_runs/nq
time CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train_dense_encoder.py train=biencoder_nq train_datasets=[nq_train] dev_datasets=[nq_dev] output_dir=/data/local/DPR_runs/nq > /data/local/DPR_runs/nq.log  # This takes up ~21G of memory on each GPU. The total training time is 3-4 hours using A100s.
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=0 python generate_dense_embeddings.py model_file=/data/local/DPR_runs/nq/dpr_biencoder.34 ctx_src=dpr_wiki shard_id=${i} num_shards=16 out_file='/data/local/DPR_runs/nq/embs' batch_size=1024; done;
CUDA_VISIBLE_DEVICES=0 python dense_retriever.py model_file=/data/local/DPR_runs/nq/dpr_biencoder.34 qa_dataset=nq_test ctx_datatsets=[dpr_wiki] encoded_ctx_files=[\"/data/local/DPR_runs/nq/embs*\"] out_file=/data/local/DPR_runs/nq/out.json
python evaluate.py /data/local/DPR_runs/nq/out.json
```

This yields the following result:
```
k=1     k=5     k=20    k=100   filename        num_queries
46.0%   68.2%   79.1%   86.3%   out.json        3610
```

# Debugging

Uncomment print functions in `train_dense_encoder.py`.

```
CUDA_VISIBLE_DEVICES= python train_dense_encoder.py train=biencoder_nq train_datasets=[nq_train2] dev_datasets=[nq_train2] train.batch_size=2 train.dev_batch_size=2 train.num_train_epochs=5 output_dir=/data/local/DPR_runs/toy train.shuffle=False encoder.dropout=0 train.learning_rate=1e-4 train.warmup_steps=2 train.log_batch_step=1 train.skip_saving=True
CUDA_VISIBLE_DEVICES=0 python train_dense_encoder.py train=biencoder_nq train_datasets=[nq_train3] dev_datasets=[nq_train3] train.batch_size=2 train.dev_batch_size=2 train.num_train_epochs=2 output_dir=/data/local/DPR_runs/toy  train.learning_rate=1e-4 train.warmup_steps=0 train.log_batch_step=1 train.skip_saving=True
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_dense_encoder.py train=biencoder_nq train_datasets=[nq_train3] dev_datasets=[nq_train3] train.batch_size=2 train.dev_batch_size=1 train.num_train_epochs=2 output_dir=/data/local/DPR_runs/toy  train.learning_rate=1e-4 train.warmup_steps=0 train.log_batch_step=1 train.skip_saving=True
```