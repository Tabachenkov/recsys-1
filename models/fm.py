import numpy as np
from .utils.classes import BaseModel
import torch
from torch import nn
from tqdm.auto import tqdm


class FM(BaseModel):
    def __init__(self, dim, n_users, n_items,
                 num_epochs=5, batch_size=16, seed=42,
                 k=128, lr=1e-2, weight_decay=1e-5):
        self.dim = dim
        self.n_users = n_users
        self.n_items = n_items
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.k = k
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, indices, values):
        linear = (self.linear(indices).squeeze(-1) * values).sum(1)
        vectors = self.v(indices) * values.unsqueeze(-1)
        interactions = (vectors.sum(1).square() - vectors.square().sum(1)).sum(1) / 2
        return self.bias + linear + interactions

    def batch(self, rows):
        indices = nn.utils.rnn.pad_sequence(
            [torch.as_tensor(row[0], dtype=torch.long) for row in rows], batch_first=True,
        )
        values = nn.utils.rnn.pad_sequence(
            [torch.as_tensor(row[1], dtype=torch.float32) for row in rows], batch_first=True,
        )
        return indices, values

    def fit(self, matrix):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.bias = nn.Parameter(torch.zeros(()))
        self.linear = nn.Embedding(self.dim, 1, padding_idx=0)
        self.v = nn.Embedding(self.dim, self.k, padding_idx=0)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.v.weight, std=0.01)
        with torch.no_grad():
            self.v.weight[0] = 0 # for padding

        self.item_features, self.seen = {}, {}
        for indices, values in matrix:
            user = int(indices[0]) - 1
            item = int(indices[1]) - self.n_users - 1
            self.item_features[item] = (list(indices[1:]), list(values[1:]))
            self.seen.setdefault(user, set()).add(item)
        candidates = np.array(sorted(self.item_features))
        eligible = [i for i, (indices, _) in enumerate(matrix)
                    if len(self.seen[int(indices[0]) - 1]) < len(candidates)]
        if not eligible:
            raise ValueError("No unseen training items available for negative sampling")

        optimizer = torch.optim.Adam(
            [self.bias, self.linear.weight, self.v.weight],
            lr=self.lr, weight_decay=self.weight_decay,
        )
        self.losses = []
        for _ in tqdm(range(self.num_epochs), leave=False):
            order = rng.permutation(eligible)
            total_loss = 0
            for start in tqdm(range(0, len(order), self.batch_size), leave=False):
                batch = order[start:start + self.batch_size]
                positive = [matrix[i] for i in batch]
                negative = []
                for indices, _ in positive:
                    user = int(indices[0]) - 1
                    item = int(rng.choice(candidates))
                    while item in self.seen[user]:
                        item = int(rng.choice(candidates))
                    ids, vals = self.item_features[item]
                    negative.append(([user + 1, *ids], [1.0, *vals]))
                indices, values = self.batch(positive + negative)
                labels = torch.cat((torch.ones(len(batch)), torch.zeros(len(batch))))
                loss = nn.functional.binary_cross_entropy_with_logits(
                    self.forward(indices, values), labels,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch)
            self.losses.append(total_loss / len(order))
        return self

    def predict_pairs(self, matrix):
        scores = []
        with torch.no_grad():
            for start in range(0, len(matrix), self.batch_size):
                indices, values = self.batch(matrix[start:start + self.batch_size])
                scores.append(self.forward(indices, values).numpy())
        return np.concatenate(scores) if scores else np.array([])

    def predict_score(self, user):
        user = int(user)
        items = sorted(self.item_features)
        rows = [([user + 1, *self.item_features[item][0]],
                 [1.0, *self.item_features[item][1]]) for item in items]
        scores = np.full(self.n_items, -np.inf)
        scores[items] = self.predict_pairs(rows)
        return scores
