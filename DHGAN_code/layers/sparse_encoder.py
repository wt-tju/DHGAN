import torch
from torch import nn


class SparseFeatureEncoder(nn.Module):
    def __init__(self, sparse_feature_maxs, sparse_feature_embedding_dim):
        super().__init__()
        assert len(sparse_feature_maxs) == len(sparse_feature_embedding_dim)
        if isinstance(sparse_feature_maxs, int):
            sparse_feature_maxs = [sparse_feature_maxs]
        if isinstance(sparse_feature_embedding_dim, int):
            sparse_feature_embedding_dim = [sparse_feature_embedding_dim]
        self.sparse_embeddings = nn.ModuleList()
        for idx, max_value in enumerate(sparse_feature_maxs):
            self.sparse_embeddings.append(
                nn.Embedding(max_value + 1, sparse_feature_embedding_dim[idx]))
        # self.sparse_feature_embedding_dim = sparse_feature_embedding_dim

    def forward(self, inputs):
        if torch.is_tensor(inputs):
            inputs = torch.split(inputs, 1, dim=-1)
        assert isinstance(
            inputs,
            (list, tuple)), 'invalid inputs type for SparseFeatureEncoder'
        assert len(inputs) == len(self.sparse_embeddings)
        embeddings = [
            torch.squeeze(sparse_embedding(sparse_feature), dim=-2)
            for sparse_embedding, sparse_feature in zip(
                self.sparse_embeddings, inputs)
        ]
        return embeddings

