import torch
import torch.nn as nn
from dhg.nn import HGNNPConv
import torch.nn.functional as F

class HGNNLayer(nn.Module):
    def __init__(self, embed_dim, num_layers, drop_rate):
        super(HGNNLayer, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            is_last_layer = (i == num_layers - 1)
            self.layers.append(
                HGNNPConv(
                    in_channels=embed_dim,
                    out_channels=embed_dim,
                    use_bn=True,
                    drop_rate=drop_rate,
                    is_last=is_last_layer
                )
            )
    def forward(self, x, hg):
        for layer in self.layers:
            x = layer(x, hg)
        return x

class HyperInterModel(nn.Module):
    def __init__(self, config, n_users, n_wss, use_hg, ws_hg):
        super(HyperInterModel, self).__init__()
        self.num_users = n_users
        self.num_wss = n_wss
        self.user_hg = use_hg
        self.ws_hg = ws_hg
        self.embed_dim = config['embedding_size']
        self.us_drop_rate = config['us_drop_rate']
        self.ws_drop_rate = config['ws_drop_rate']
        self.user_hgnn_layers = config['user_hgnn_layers']
        self.ws_hgnn_layers = config['ws_hgnn_layers']
        self.hgnn = config['hgnn']
        self.device = config['device']

        self.user_embed = nn.Embedding(self.num_users, self.embed_dim)
        self.ws_embed = nn.Embedding(self.num_wss, self.embed_dim)
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.ws_embed.weight)

        if self.hgnn == 'hgnn':
            self.user_hgnn = HGNNLayer(self.embed_dim, self.user_hgnn_layers, self.us_drop_rate)
            self.ws_hgnn = HGNNLayer(self.embed_dim, self.ws_hgnn_layers, self.ws_drop_rate)

        self.inter_user_embed = nn.Embedding(self.num_users, self.embed_dim)
        self.inter_ws_embed = nn.Embedding(self.num_wss, self.embed_dim)
        nn.init.xavier_uniform_(self.inter_user_embed.weight)
        nn.init.xavier_uniform_(self.inter_ws_embed.weight)
        self.fusion_user = EmbedFusion(self.embed_dim)
        self.fusion_ws = EmbedFusion(self.embed_dim)
    
    def forward(self, user_ids, ws_ids):
        if self.hgnn == 'hgnn':
            user_hg_x = self.user_hgnn(self.user_embed.weight, self.user_hg)
            ws_hg_x = self.ws_hgnn(self.ws_embed.weight, self.ws_hg)
        elif self.hgnn == 'none':
            user_hg_x = self.user_embed.weight
            ws_hg_x = self.ws_embed.weight
        inter_user_x = self.inter_user_embed.weight
        inter_ws_x = self.inter_ws_embed.weight
        curr_user_hg = user_hg_x[user_ids]
        curr_ws_hg = ws_hg_x[ws_ids]
        curr_user_in = inter_user_x[user_ids]
        curr_ws_in = inter_ws_x[ws_ids]

        final_user_x, cold_user, _ ,_, _= self.fusion_user(curr_user_hg, curr_user_in)
        final_ws_x, cold_ws, _, _, _ = self.fusion_ws(curr_ws_hg, curr_ws_in)
        return F.softplus((final_user_x * final_ws_x).sum(dim=-1)), F.softplus((cold_user * cold_ws).sum(dim=-1)), \
               F.softplus((final_user_x * cold_ws).sum(dim=-1)), F.softplus((cold_user * final_ws_x).sum(dim=-1))


    def extract_embeddings(self):
        if self.hgnn == 'hgnn':
            user_hg_x = self.user_hgnn(self.user_embed.weight, self.user_hg)
            ws_hg_x = self.ws_hgnn(self.ws_embed.weight, self.ws_hg)
        elif self.hgnn == 'none':
            user_hg_x = self.user_embed.weight
            ws_hg_x = self.ws_embed.weight
        inter_user_x = self.inter_user_embed.weight
        inter_ws_x = self.inter_ws_embed.weight
        final_user_x, cold_user,user_hg_x, inter_user_x,user_gate = self.fusion_user(user_hg_x, inter_user_x)
        final_ws_x, cold_ws,ws_hg_x, inter_ws_x,ws_gate = self.fusion_ws(ws_hg_x, inter_ws_x)
        return final_user_x,final_ws_x,cold_user,cold_ws,user_hg_x, ws_hg_x, inter_user_x, inter_ws_x,user_gate,ws_gate


class EmbedFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.prop_mapping = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.inter_mapping = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.gate_layer = nn.Linear(embed_dim * 2, embed_dim)
        self.align_norm = nn.LayerNorm(embed_dim)
        self.shared_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    def forward(self, u_hg, u_lg):
        hg = self.prop_mapping(u_hg)
        lg = self.inter_mapping(u_lg)
        gate_weight = torch.sigmoid(self.gate_layer(torch.cat([hg, lg], dim=-1)))
        return self.shared_mlp(self.align_norm(hg+gate_weight*lg+hg*lg)), self.shared_mlp(self.align_norm(hg)), hg, lg, gate_weight