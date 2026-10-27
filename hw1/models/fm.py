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

    def fit(self, matrix, short=None, seen=None, short_weight=0.5, user_feature_count=0):
        self._item_indices = None
        self.user_feature_count = user_feature_count
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.bias = nn.Parameter(torch.zeros(()))
        self.linear = nn.Embedding(self.dim, 1, padding_idx=0)
        self.v = nn.Embedding(self.dim, self.k, padding_idx=0)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.v.weight, std=0.01)
        with torch.no_grad():
            self.v.weight[0] = 0 # for padding

        self.item_features, self.user_features, self.seen = {}, {}, {}
        offset = 2 + user_feature_count
        for indices, values in matrix:
            user = int(indices[0]) - 1
            item = int(indices[1]) - self.n_users - 1
            self.user_features[user] = (list(indices[2:offset]), list(values[2:offset]))
            self.item_features[item] = ([indices[1], *indices[offset:]],
                                        [values[1], *values[offset:]])
            self.seen.setdefault(user, set()).add(item)
        candidates = np.array(sorted(self.item_features))
        short_rows = []
        for indices, values in ([] if short is None else short):
            user = int(indices[0]) - 1
            item = int(indices[1]) - self.n_users - 1
            if user in self.seen and item not in self.seen[user]:
                short_rows.append((indices, values))
        for indices, _ in short_rows:
            user = int(indices[0]) - 1
            self.seen[user].add(int(indices[1]) - self.n_users - 1)
        if seen is not None:
            for user in self.seen:
                self.seen[user].update(seen[user].indices)
        candidate_set = set(candidates)
        can_sample = {user: not candidate_set.issubset(history)
                      for user, history in self.seen.items()}
        eligible = [i for i, (indices, _) in enumerate(matrix)
                    if can_sample[int(indices[0]) - 1]]
        if not eligible and len(short_rows) < len(matrix):
            raise ValueError("Not enough negative pairs to balance positives")

        optimizer = torch.optim.Adam(
            [self.bias, self.linear.weight, self.v.weight],
            lr=self.lr, weight_decay=self.weight_decay,
        )
        self.losses = []
        for _ in tqdm(range(self.num_epochs), leave=False):
            selected = rng.choice(len(short_rows), min(len(short_rows), len(matrix)), replace=False)
            negative = [short_rows[i] for i in selected]
            negative_weights = [short_weight] * len(negative)
            for _ in tqdm(range(len(matrix) - len(negative)),
                          desc="Negative sampling", leave=False):
                indices, _ = matrix[eligible[int(rng.integers(len(eligible)))]]
                user = int(indices[0]) - 1
                item = int(rng.choice(candidates))
                while item in self.seen[user]:
                    item = int(rng.choice(candidates))
                ids, vals = self.item_features[item]
                user_ids, user_vals = self.user_features[user]
                negative.append(([user + 1, ids[0], *user_ids, *ids[1:]],
                                 [1.0, vals[0], *user_vals, *vals[1:]]))
                negative_weights.append(1.0)
            rows = list(matrix) + negative
            labels = torch.cat((torch.ones(len(matrix)), torch.zeros(len(negative))))
            weights = torch.tensor([1.0] * len(matrix) + negative_weights)
            order = rng.permutation(len(rows))
            total_loss = 0
            for start in tqdm(range(0, len(order), self.batch_size), leave=False):
                batch = order[start:start + self.batch_size]
                indices, values = self.batch([rows[i] for i in batch])
                loss = nn.functional.binary_cross_entropy_with_logits(
                    self.forward(indices, values), labels[batch], reduction="none",
                )
                loss = (loss * weights[batch]).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch)
            self.losses.append(total_loss / len(rows))
        return self

    def predict_score(self, user):
        user = int(user)
        if getattr(self, "_item_indices", None) is None:
            self._prediction_items = np.array(sorted(self.item_features))
            rows = []
            for item in self._prediction_items:
                ids, vals = self.item_features[item]
                rows.append(([0, ids[0], *([0] * self.user_feature_count), *ids[1:]],
                             [1.0, vals[0], *([0.0] * self.user_feature_count), *vals[1:]]))
            self._item_indices, self._item_values = self.batch(rows)
        scores = np.full(self.n_items, -np.inf)
        with torch.no_grad():
            for start in range(0, len(self._prediction_items), self.batch_size):
                end = start + self.batch_size
                indices = self._item_indices[start:end].clone()
                indices[:, 0] = user + 1
                values = self._item_values[start:end].clone()
                ids, vals = self.user_features[user]
                indices[:, 2:2 + self.user_feature_count] = torch.as_tensor(ids, dtype=torch.long)
                values[:, 2:2 + self.user_feature_count] = torch.as_tensor(vals, dtype=torch.float32)
                scores[self._prediction_items[start:end]] = self.forward(indices, values).numpy()
        return scores
