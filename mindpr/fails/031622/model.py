import collections
import torch
import torch.distributed as dist
import torch.nn.functional as F

from torch.optim.lr_scheduler import LambdaLR
from torch.serialization import default_restore_location
from transformers import (BertTokenizer, AdamW, get_constant_schedule,
                          BertModel, get_linear_schedule_with_warmup)
from tqdm import tqdm


class BertEncoder(torch.nn.Module):

    def __init__(self, dropout=0.1):
        super().__init__()
        self.encoder = BertModel.from_pretrained(
            'bert-base-uncased', attention_probs_dropout_prob=dropout,
            hidden_dropout_prob=dropout) #add_pooling_layer=False)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        #from transformers import set_seed
        #set_seed(42)  # This controls dropout
        output = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)
        last_hidden_state = output[0]  # (B, L, d)
        embeddings = last_hidden_state[:, 0, :]  # (B, d)
        return embeddings


class DPRModel(torch.nn.Module):

    def __init__(self, tokenizer, query_encoder, passage_encoder):
        super().__init__()
        self.tokenizer = tokenizer
        self.query_encoder = query_encoder
        self.passage_encoder = passage_encoder

    def forward(self, Q=None, Q_mask=None, Q_type=None, P=None, P_mask=None,
                P_type=None):
        X = self.query_encoder(Q, Q_mask, Q_type) if Q is not None else None
        Y = self.passage_encoder(P, P_mask, P_type) if P is not None else None
        return X, Y

    def weight(self):
        return self.query_encoder.emb.weight[101, :]
    def grad(self):
        return self.query_encoder.emb.weight.grad[101, :]


def make_bert_model(tokenizer, dropout=0.1):
    query_encoder = BertEncoder(dropout)
    passage_encoder = BertEncoder(dropout)
    model = DPRModel(tokenizer, query_encoder, passage_encoder)
    return model


def run_forward_train(model, batch, rank=-1, world_size=-1, device=None):
    if device is None:
        device = torch.device('cpu')
    Q, Q_mask, Q_type, P, P_mask, P_type, labels  = [tensor.to(device) for
                                                     tensor in batch[:-1]]
    X, Y = model(Q, Q_mask, Q_type, P, P_mask, P_type)

    if world_size != -1:
        labels += rank * len(Y)
        X_list = [torch.zeros_like(X) for _ in range(world_size)]
        Y_list = [torch.zeros_like(Y) for _ in range(world_size)]
        labels_list = [torch.zeros_like(labels) for _ in
                       range(world_size)]
        dist.all_gather(tensor_list=X_list, tensor=X.contiguous())
        dist.all_gather(tensor_list=Y_list, tensor=Y.contiguous())
        dist.all_gather(tensor_list=labels_list,
                        tensor=labels.contiguous())

        # Since all_gather results do not have gradients, we replace
        # the current process's embeddings with originals.
        X_list[rank] = X
        Y_list[rank] = Y

        X = torch.cat(X_list, 0)
        Y = torch.cat(Y_list, 0)
        labels = torch.cat(labels_list, 0)

    return X, Y, labels


def get_loss(model, batch, rank=-1, world_size=-1, device=None):
    X, Y, labels = run_forward_train(model, batch, rank, world_size, device)
    scores = X @ Y.t()  # (B, 2B)

    # No need to all reduce loss/num_correct: each process already
    # has all examples across all processes.
    loss = F.cross_entropy(scores, labels, reduction='mean')
    num_correct = (torch.max(scores, 1)[1] == labels).sum()

    return loss, num_correct


def validate_by_rank(model, loader, rank=-1, world_size=-1,
                     device=None, subbatch_size=128, disable_tqdm=True):
    if device is None:
        device = torch.device('cpu')

    X_all = []
    Y_all = []
    labels_all = []
    buffer_size = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, disable=disable_tqdm):
            Q, Q_mask, Q_type, P, P_mask, P_type, labels, _ = batch

            X, _ = model(Q=Q.to(device), Q_mask=Q_mask.to(device),
                         Q_type=Q_type.to(device))
            X_all.append(X.cpu())

            for i in range(0, P.size(0), subbatch_size):
                subP = P[i: i + subbatch_size, :]
                subP_mask = P_mask[i: i + subbatch_size, :]
                subP_type = P_type[i: i + subbatch_size, :]
                _, subY = model(P=subP.to(device),
                                P_mask=subP_mask.to(device),
                                P_type=subP_type.to(device))
                Y_all.append(subY.cpu())

            labels_all.append((labels + buffer_size).cpu())
            buffer_size += P.size(0)

    X_all = torch.cat(X_all, 0)  # (N/P, d)
    Y_all = torch.cat(Y_all, 0)  # (MN/P, d)
    labels_all = torch.cat(labels_all, 0)  # (N/P,): each element in [MN/P]
    scores = X_all @ Y_all.t()  # (N/P, MN/P)
    _, indices = torch.sort(scores, dim=1, descending=True)  # (N/P, MN/P)
    ranks = (indices == labels_all.view(-1, 1)).nonzero()[:, 1]
    sum_ranks = ranks.sum().to(device)
    num_queries = torch.LongTensor([indices.size(0)]).to(device)
    num_cands = torch.LongTensor([indices.size(1)]).to(device)
    if world_size != -1:
        dist.all_reduce(sum_ranks, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_queries, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_cands, op=dist.ReduceOp.SUM)

    rank_average = (sum_ranks / num_queries).item()
    num_cands_avg = num_cands.item() / world_size \
                    if world_size != -1 else num_cands.item()

    return rank_average, num_cands_avg


def run_forward_encode(model, batch, world_size=-1, device=None):
    if device is None:
        device = torch.device('cpu')
    P, P_mask, P_type, I  = [tensor.to(device) for tensor in batch]
    _, Y = model(P=P, P_mask=P_mask, P_type=P_type)

    if world_size != -1:
        Y_list = [torch.zeros_like(Y) for _ in range(world_size)]
        I_list = [torch.zeros_like(I) for _ in range(world_size)]
        dist.all_gather(tensor_list=Y_list, tensor=Y.contiguous())
        dist.all_gather(tensor_list=I_list, tensor=I.contiguous())
        Y = torch.cat(Y_list, 0)
        I = torch.cat(I_list)

    return Y, I


def init_train_components(tokenizer, lr, num_warmup_steps, num_training_steps,
                          adam_eps=1e-8, dropout=0.1, device=None):
    if device is None:
        device = torch.device('cpu')
    model = make_bert_model(tokenizer, dropout=dropout).to(device)
    optimizer = get_bert_optimizer(model, learning_rate=lr,
                                   adam_eps=adam_eps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps)

    return model, optimizer, scheduler


def get_bert_optimizer(model, learning_rate=2e-5, adam_eps=1e-6):
    # https://github.com/google-research/bert/blob/master/optimization.py#L25
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate,
                      eps=adam_eps)
    return optimizer


def load_model(model_path, tokenizer, device):
    saved_pickle = torch.load(model_path, map_location=device)
    model, _, _ = init_train_components(tokenizer, 42, 42, 42, device=device)
    if 'sd' in saved_pickle:  # My model
        model.load_state_dict(saved_pickle['sd'])
        args_saved = saved_pickle['args']
    else:  # DPR model
        saved_state = load_states_from_checkpoint(model_path)
        question_state = {
            key[len("question_model."):]: value for (key, value) in
            saved_state.model_dict.items() if key.startswith("question_model.")
        }
        passage_state = {
            key[len("ctx_model."):]: value for (key, value) in
            saved_state.model_dict.items() if key.startswith("ctx_model.")
        }
        model.query_encoder.encoder.load_state_dict(question_state, strict=False)
        model.passage_encoder.encoder.load_state_dict(passage_state, strict=False)
        ArgsMock = collections.namedtuple('ArgsMock', ['max_length'])
        args_saved = ArgsMock(saved_state.encoder_params['sequence_length'])

    return model, args_saved


################## BEGIN: DPR code #############################################
CheckpointState = collections.namedtuple(
    "CheckpointState",
    [
        "model_dict",
        "optimizer_dict",
        "scheduler_dict",
        "offset",
        "epoch",
        "encoder_params",
    ],
)

def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, "cpu"))
    return CheckpointState(**state_dict)
################## END: DPR code ###############################################
