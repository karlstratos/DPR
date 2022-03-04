import torch
import unittest

from data import DPRDataset, tensorize
from model import make_bert_model, get_loss, init_train_components
from transformers import BertTokenizer, set_seed


class TestModel(unittest.TestCase):
    TRAIN2 = '/common/home/jl2529/repositories/DPR/downloads/data/retriever/nq-train2.json'

    def setUp(self):
        self.dataset = DPRDataset(self.TRAIN2)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = make_bert_model(self.tokenizer, dropout=0.)

    def test_train(self):
        model, optimizer, scheduler = init_train_components(
            self.tokenizer, 1e-4, 2, 5, dropout=0.)
        batch = tensorize(self.dataset.samples, self.tokenizer, 256)

        losses = []
        lrs = []
        model.train()
        for epoch in range(5):
            loss, _ = get_loss(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])

        # (DPR) jl2529@two:~/repositories/DPR$ CUDA_VISIBLE_DEVICES= python train_dense_encoder.py train=biencoder_nq train_datasets=[nq_train2] dev_datasets=[nq_train2] train.batch_size=2 train.dev_batch_size=2 train.num_train_epochs=5 output_dir=/data/local/DPR_runs/toy train.shuffle=False encoder.dropout=0 train.learning_rate=1e-4 train.warmup_steps=2 train.log_batch_step=1 train.skip_saving=True
        # Epoch: 0: Step: 1/1, loss=26.576244, lr=0.000050
        # Epoch: 1: Step: 1/1, loss=26.576244, lr=0.000100
        # Epoch: 2: Step: 1/1, loss=0.475171, lr=0.000067
        # Epoch: 3: Step: 1/1, loss=0.003750, lr=0.000033
        # Epoch: 4: Step: 1/1, loss=0.000301, lr=0.000000
        self.assertAlmostEqual(losses[0], 26.5762, delta=1e-4)
        self.assertAlmostEqual(losses[1], 26.5762, delta=1e-4)
        self.assertAlmostEqual(losses[2], 0.475171, delta=1e-4)
        self.assertAlmostEqual(losses[3], 0.003750, delta=1e-4)
        self.assertAlmostEqual(losses[4], 0.000301, delta=1e-4)

        self.assertAlmostEqual(lrs[0], 0.000050, delta=1e-4)
        self.assertAlmostEqual(lrs[1], 0.000100, delta=1e-4)
        self.assertAlmostEqual(lrs[2], 0.000067, delta=1e-4)
        self.assertAlmostEqual(lrs[3], 0.000033, delta=1e-4)
        self.assertAlmostEqual(lrs[4], 0.000000, delta=1e-4)

    def test_get_loss(self):
        batch = tensorize(self.dataset.samples, self.tokenizer, 256)
        loss, num_correct = get_loss(self.model, batch)
        self.assertAlmostEqual(loss.item(), 26.5762, delta=1e-4)
        self.assertEqual(num_correct.item(), 0)

    def test_model_forward(self):
        Q, Q_mask, Q_type, P, P_mask, P_type, labels = tensorize(
            self.dataset.samples, self.tokenizer, 256)
        X, Y = model(Q, Q_mask, Q_type, P, P_mask, P_type)

        #local_q_vector (dropout=0)
        #[[-0.2295, -0.4305,  0.0497,  ..., -0.3996,  0.2874,  0.3665],
        # [ 0.1620, -0.2417, -0.1147,  ..., -0.3998,  0.0562, -0.0686]]
        self.assertAlmostEqual(X[0, -1].item(), 0.3665, delta=1e-3)
        self.assertAlmostEqual(X[1, -1].item(), -0.0686, delta=1e-3)

        #local_ctx_vectors (dropout=0)
        #[[ 0.0490, -0.7454, -0.3477,  ...,  0.3973,  1.0177, -0.0231],
        # [-0.2084, -1.4637, -0.1923,  ...,  0.2691,  1.0294,  0.4863],
        # [-0.3259, -0.8479,  0.1591,  ...,  0.5030, -0.2953,  0.2830],
        # [-0.3363, -0.8176,  0.2325,  ..., -0.2906, -0.2854,  0.2375]],
        self.assertAlmostEqual(Y[0, -1].item(), -0.0231, delta=1e-3)
        self.assertAlmostEqual(Y[1, -1].item(), 0.4863, delta=1e-3)
        self.assertAlmostEqual(Y[2, -1].item(), 0.2830, delta=1e-3)
        self.assertAlmostEqual(Y[3, -1].item(), 0.2375, delta=1e-3)





if __name__ == '__main__':
    unittest.main()
