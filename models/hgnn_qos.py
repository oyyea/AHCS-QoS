import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hyperinter import HGNNLayer


class HGNNQoSModel(nn.Module):
    def __init__(self, config, n_users, n_wss, user_hg, ws_hg):
        super(HGNNQoSModel, self).__init__()
        embed_dim = config["embedding_size"]

        self.user_hg = user_hg
        self.ws_hg = ws_hg

        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.ws_embed = nn.Embedding(n_wss, embed_dim)
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.ws_embed.weight)

        # 属性分组加权版本（已停用）：
        # self.user_hgnn = HGNNLayer(
        #     embed_dim,
        #     config["user_hgnn_layers"],
        #     config["us_drop_rate"],
        #     num_attrs=len(config["user_attr"]),
        # )
        # self.ws_hgnn = HGNNLayer(
        #     embed_dim,
        #     config["ws_hgnn_layers"],
        #     config["ws_drop_rate"],
        #     num_attrs=len(config["ws_attr"]),
        # )

        # 恢复原始合并超图 HGNNP 基线。
        self.user_hgnn = HGNNLayer(
            embed_dim,
            config["user_hgnn_layers"],
            config["us_drop_rate"],
        )
        self.ws_hgnn = HGNNLayer(
            embed_dim,
            config["ws_hgnn_layers"],
            config["ws_drop_rate"],
        )

        self.user_mapping = self._build_mapping(embed_dim)
        self.ws_mapping = self._build_mapping(embed_dim)
        self.user_norm = nn.LayerNorm(embed_dim)
        self.ws_norm = nn.LayerNorm(embed_dim)

        self.user_head = self._build_mapping(embed_dim)
        self.ws_head = self._build_mapping(embed_dim)

    # 属性分组加权版本的权重导出接口（已停用）：
    # def get_attribute_weights(self):
    #     return self.user_hgnn.get_attribute_weights(), self.ws_hgnn.get_attribute_weights()

    @staticmethod
    def _build_mapping(embed_dim):
        return nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, user_ids, ws_ids):
        user_x = self.user_hgnn(self.user_embed.weight, self.user_hg)
        ws_x = self.ws_hgnn(self.ws_embed.weight, self.ws_hg)

        user_x = self.user_mapping(user_x[user_ids])
        ws_x = self.ws_mapping(ws_x[ws_ids])

        user_x = self.user_head(self.user_norm(user_x))
        ws_x = self.ws_head(self.ws_norm(ws_x))
        pred = F.softplus((user_x * ws_x).sum(dim=-1))

        return pred, pred, pred, pred

    def extract_embeddings(self):
        user_x = self.user_hgnn(self.user_embed.weight, self.user_hg)
        ws_x = self.ws_hgnn(self.ws_embed.weight, self.ws_hg)

        user_x = self.user_mapping(user_x)
        ws_x = self.ws_mapping(ws_x)

        user_x = self.user_head(self.user_norm(user_x))
        ws_x = self.ws_head(self.ws_norm(ws_x))
        return user_x,ws_x