import torch
import torch.nn as nn


class ResidualAddBlock(nn.Module):
    def __init__(self, res_add: nn.Sequential):
        super(ResidualAddBlock, self).__init__()
        self.res_add = res_add

    def forward(self, x: torch.tensor, **kwargs) -> torch.tensor:
        res = x
        x = self.res_add(x, **kwargs)
        return x + res


class FeedForwardBlock(nn.Module):
    def __init__(self,
            embedding_dim: int,
            expansion: int,
            feed_dropout: float
        ):
        super(FeedForwardBlock, self).__init__()
        self.sequence = nn.Sequential(
            nn.Linear(embedding_dim, expansion * embedding_dim),
            nn.GELU(),
            nn.Dropout(feed_dropout),
            nn.Linear(expansion * embedding_dim, embedding_dim),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.sequence(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self,
            embedding_dim: int,
            num_heads: int,
            attention_dropout: float,
            feed_dropout: float
        ):
        super(TransformerEncoderBlock, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, attention_dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.layers_after_attention =  nn.Sequential(
            ResidualAddBlock(
                nn.Sequential(
                    nn.LayerNorm(embedding_dim),
                    FeedForwardBlock(embedding_dim, expansion=2, feed_dropout=feed_dropout),
                    nn.Dropout(feed_dropout)
                )
            )
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        res = x.clone()
        x = self.layer_norm(x)
        x, _ = self.attention(x, x, x)
        x = self.attention_dropout(x)
        x += res
        return self.layers_after_attention(x)


class Classification(nn.Module):
    def __init__(self,
            embedding_dim: int = 32,
            sequence_length: int = 128,
            feed_dropout: float = 0.1,
            n_classes: int = 1
        ):
        super(Classification, self).__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Dropout(feed_dropout),
            nn.Linear(embedding_dim, n_classes)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = torch.mean(x, dim=1, keepdims=True)
        return self.classifier(x)
