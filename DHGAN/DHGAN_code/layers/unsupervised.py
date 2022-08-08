import torch
import torch.nn.functional as F


class UnsupervisedModel(torch.nn.Module):
    def __init__(self, node_nums, embedding_dim):
        super().__init__()
        assert node_nums > 1
        assert embedding_dim > 0
        self.embedding_dim = embedding_dim
        self.embedder = torch.nn.Embedding(node_nums + 1, embedding_dim)
        # self.node_embedding = NodeEmbedding(num_embeddings=node_nums, embedding_dim=emb_dim, name='node_emb')
        self.context_embedder = None

    def reset_parameters(self):
        if self.embedder:
            self.embedder.reset_parameters()
        if self.context_embedder:
            self.context_embedder.reset_parameters()

    def forward(self, inputs):
        src, pos, negs = self.get_sample(inputs)
        embedding = self.embedder(src)
        pos_embedding = self.context_embedder(pos)
        negs_embedding = self.context_embedder(negs)

        logits = torch.sum(embedding * pos_embedding, dim=2)
        negs_logits = torch.sum(embedding * negs_embedding, dim=2)

        # _mrr = mrr(logits, negs_logits)

        pos_loss = F.logsigmoid(logits).sum(dim=-1)
        negs_loss = F.logsigmoid(-negs_logits).sum(dim=-1)
        loss = -(pos_loss + negs_loss).mean()
        return loss

    # def get_embedding(self, inputs):
    #     raise NotImplementedError

    def get_sample(self, inputs):
        '''sample may include sources, positive, negative'''
        raise NotImplementedError



