

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max


# --- RMSKernel ---
class RMSKernel(nn.Module):
    def forward(self, x):
        if x.numel() == 0:
            return torch.tensor(0.0, device=x.device)
        return torch.sqrt(torch.mean(x**2) + 1e-8)



# --- MixedKernel ---
class MixedKernel(nn.Module):
    def __init__(self, max_edges=1000):
        super().__init__()
        self.max_edges = max_edges
        self.rms_kernel = RMSKernel()

        self.linear_kernel = nn.Sequential(
            nn.Linear(max_edges, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )


        self.pos_weights = nn.Parameter(torch.randn(max_edges))
        self.register_buffer('pos_enc',
            torch.linspace(0, 1, max_edges))


        self.feature_net = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):

        if x.numel() == 0:
            return torch.tensor(0.0, device=x.device)


        n = x.size(0)
        x_padded = F.pad(x, (0, self.max_edges - n))


        linear_feat = self.linear_kernel(x_padded.unsqueeze(0))
        features = [linear_feat.squeeze()]


        pos_weight = F.softmax(self.pos_weights + self.pos_enc, dim=0)
        weighted_sum = (x_padded * pos_weight).sum()
        features.append(weighted_sum)


        x = x_padded[:n]
        features.append(x.mean())

        features.append(self.rms_kernel(x))


        feat_tensor = torch.stack(features)
        return self.feature_net(feat_tensor.unsqueeze(0)).squeeze()



# --- FCIHLayer ---
class FCIHLayer(nn.Module):


    def __init__(
        self,
        node_type_dims,
        edge_types,
        edge_type_counts,
        num_heads=2,
        concat=False,
        dropout=0.3,
        two_hop_coef=0.2
    ):
        super().__init__()
        self.num_heads = num_heads
        self.concat = concat
        self.node_types = list(node_type_dims.keys())
        self.edge_types = list(edge_types)
        self.et2idx = {et: i for i, et in enumerate(self.edge_types)}
        assert 'cites' in self.et2idx, "edge_types must include 'cites'"

        self.dropout = nn.Dropout(p=dropout)
        self.leakyrelu = nn.LeakyReLU(0.3)


        # Node-type feature transformation
        self.mlp_transforms = nn.ModuleDict({
            nt: nn.Sequential(
                nn.Linear(dim, 1024),
                nn.ReLU(),
                nn.LayerNorm(1024),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.LayerNorm(512),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.LayerNorm(256),
                nn.Dropout(0.3),
                nn.Linear(256, 160),
            )
            for nt, dim in node_type_dims.items()
        })


        self.edge_type_mlps = nn.ModuleDict({
            et: nn.Sequential(
                nn.Linear(320, 150),
                nn.ReLU(),
                nn.LayerNorm(150),
                nn.Linear(150, 16),
            )
            for et in self.edge_types
        })


        # Edge-type-guided node-level attention
        self.attention_linear = nn.ModuleDict({
            et: nn.ModuleList([
                nn.Sequential(
                    nn.Linear(336, 170),
                    nn.ReLU(),
                    nn.LayerNorm(170),
                    nn.Linear(170, 1)
                )
                for _ in range(num_heads)
            ])
            for et in self.edge_types
        })


        # Global edge-type weighting
        self.edge_type_attention = nn.ModuleDict({
            et: nn.ModuleList([
                MixedKernel(max_edges=edge_type_counts.get(et, 1))
                for _ in range(num_heads)
            ])
            for et in self.edge_types
        })


        # Node-wise cross-view gate
        self.gate_view_1hop = nn.Sequential(
            nn.Linear(320, 512), nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.gate_view_2hop = nn.Sequential(
            nn.Linear(320, 512), nn.ReLU(),
            nn.Linear(512, 1)
        )


        self.two_hop_coef = float(two_hop_coef)

    @staticmethod
    def _init_gate_to_value(gate_module: nn.Module, p=0.001):
        last_linear = None
        for m in reversed(list(gate_module.modules())):
            if isinstance(m, nn.Linear):
                last_linear = m
                break
        assert last_linear is not None
        b = math.log(p / (1 - p))
        nn.init.zeros_(last_linear.weight)
        nn.init.constant_(last_linear.bias, b)

    def _compute_edge_type_emb(self, transformed, src, dst, edge_type_indices, et_list):


        emb = {}
        for et in et_list:
            et_idx = self.et2idx[et]
            mask = (edge_type_indices == et_idx)
            if mask.sum() == 0:
                continue

            sum_src = transformed[src[mask]].mean(dim=0)
            sum_dst = transformed[dst[mask]].mean(dim=0)
            combined = torch.cat([sum_src.mean(dim=0), sum_dst.mean(dim=0)], dim=0)
            emb[et] = self.edge_type_mlps[et](combined)
        return emb

    def _encode_view(
        self,
        transformed,
        edge_index,
        edge_type_indices,
        num_nodes,
        et_list,
        use_global_edge_type_weight: bool
    ):


        device = transformed.device
        E = edge_index.size(1)
        if E == 0:
            return [torch.zeros(num_nodes, 160, device=device) for _ in range(self.num_heads)]

        src, dst = edge_index[0], edge_index[1]
        edge_type_emb = self._compute_edge_type_emb(transformed, src, dst, edge_type_indices, et_list)

        out_heads = []
        for head in range(self.num_heads):
            e = torch.zeros(E, device=device)


            for et in et_list:
                et_idx = self.et2idx[et]
                mask = (edge_type_indices == et_idx)
                if mask.sum() == 0:
                    continue

                h_src = transformed[src[mask], head]
                h_dst = transformed[dst[mask], head]

                et_emb = edge_type_emb.get(et, None)
                if et_emb is None:

                    et_feat = torch.zeros(h_src.size(0), 16, device=device)
                else:
                    et_feat = et_emb.unsqueeze(0).repeat(h_src.size(0), 1)

                cat = torch.cat([h_src, h_dst, et_feat], dim=1)
                attn = self.attention_linear[et][head](cat).squeeze(-1)
                e[mask] = self.leakyrelu(attn)


            with torch.no_grad():
                m = scatter_max(e, dst, dim=0, dim_size=num_nodes)[0][dst]
                e_stable = e - m
            exp_e = torch.exp(e_stable)
            denom = torch.zeros(num_nodes, device=device).scatter_add_(0, dst, exp_e)
            alpha = exp_e / (denom[dst] + 1e-16)
            alpha = F.dropout(alpha, p=0.6, training=self.training)


            if use_global_edge_type_weight and len(et_list) > 1:
                et_scores = []
                et_keys = []
                for et in et_list:
                    et_idx = self.et2idx[et]
                    mask = (edge_type_indices == et_idx)
                    if mask.sum() == 0:
                        et_scores.append(torch.tensor(0.0, device=device))
                    else:
                        et_scores.append(self.edge_type_attention[et][head](e_stable[mask].detach()))
                    et_keys.append(et)
                w = F.softmax(torch.stack(et_scores), dim=0)
                et_w = dict(zip(et_keys, w))
            else:
                et_w = {et: torch.tensor(1.0, device=device) for et in et_list}

            scaled_alpha = torch.zeros_like(alpha)
            for et in et_list:
                et_idx = self.et2idx[et]
                mask = (edge_type_indices == et_idx)
                if mask.sum() == 0:
                    continue
                scaled_alpha[mask] = alpha[mask] * et_w[et]


            msg = transformed[src, head] * scaled_alpha.unsqueeze(-1)
            h_1hop = torch.zeros(num_nodes, 160, device=device).index_add_(0, dst, msg)


            h_2hop = optimized_two_hop_propagate(edge_index, scaled_alpha, transformed[:, head, :], num_nodes)
            g = torch.sigmoid(self.gate_view_2hop(torch.cat([h_1hop, h_2hop], dim=-1)))
            out = h_1hop + self.two_hop_coef*g * h_2hop
            out_heads.append(out)

        return out_heads

    def forward(
        self,
        h_dict,
        edge_index_multi,
        edge_type_list_multi,
        edge_index_target,
        num_nodes,
        node_type_indices
    ):
        device = next(self.parameters()).device


        transformed = torch.zeros(num_nodes, self.num_heads, 160, device=device)
        for nt in self.node_types:
            idx = node_type_indices[nt]
            if idx.numel() == 0:
                continue
            feat = self.mlp_transforms[nt](h_dict[nt].to(device))
            transformed[idx] = feat.unsqueeze(1).repeat(1, self.num_heads, 1)


        et_list_multi = self.edge_types

        # Multi-relational view
        heads_multi = self._encode_view(
            transformed=transformed,
            edge_index=edge_index_multi.to(device),
            edge_type_indices=edge_type_list_multi.to(device),
            num_nodes=num_nodes,
            et_list=et_list_multi,
            use_global_edge_type_weight=True
        )


        E_target = edge_index_target.size(1)
        target_et_idx = self.et2idx['cites']
        edge_type_list_target = torch.full((E_target,), target_et_idx, dtype=torch.long, device=device)

        # Target-relation view
        heads_target = self._encode_view(
            transformed=transformed,
            edge_index=edge_index_target.to(device),
            edge_type_indices=edge_type_list_target,
            num_nodes=num_nodes,
            et_list=['cites'],
            use_global_edge_type_weight=False
        )


        # Node-wise adaptive residual fusion
        fused_heads = []
        for h_multi, h_target in zip(heads_multi, heads_target):
            logits_node = self.gate_view_1hop(torch.cat([h_multi, h_target], dim=-1))


            g_soft = torch.sigmoid(logits_node)


            h = h_multi + g_soft * h_target
            fused_heads.append(h)


        h_heads = torch.stack(fused_heads, dim=1)


        if self.concat:
            h_out = h_heads.reshape(num_nodes, -1) + transformed.reshape(num_nodes, -1)
        else:
            h_out = h_heads.mean(dim=1) + transformed.mean(dim=1)

        return h_out



# --- k_hop_propagate_edgewise ---
def k_hop_propagate_edgewise(edge_index, alpha, features, num_nodes, k: int):


    src, dst = edge_index[0], edge_index[1]
    h = features
    for _ in range(k):
        out = torch.zeros(num_nodes, h.size(1), device=h.device, dtype=h.dtype)
        out.index_add_(0, dst, h[src] * alpha.unsqueeze(-1))
        h = out
    return h



# --- optimized_three_hop_propagate ---
def optimized_three_hop_propagate(edge_index, alpha, features, num_nodes):
    return k_hop_propagate_edgewise(edge_index, alpha, features, num_nodes, k=3)



# --- optimized_two_hop_propagate ---
def optimized_two_hop_propagate(edge_index, alpha, features, num_nodes):
    return k_hop_propagate_edgewise(edge_index, alpha, features, num_nodes, k=2)



# --- DualViewGraphNetwork ---
class DualViewGraphNetwork(nn.Module):
    def __init__(self, node_type_dims, edge_type_counts, num_heads, node_counts, edge_types):
        super().__init__()
        self.node_counts = node_counts

        self.gat_layer = FCIHLayer(
            node_type_dims=node_type_dims,
            edge_types=edge_types,
            edge_type_counts=edge_type_counts,
            num_heads=num_heads,
            concat=False,
            dropout=0.3
        )

        self.edge_predictor = nn.Sequential(
            nn.Linear(160 * 2, 160 * 2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(320, 1),
        )

    def forward(self, h_dict, edge_index_multi, edge_type_list_multi, edge_index_target,
                edges, num_total_nodes, node_type_indices):

        h = self.gat_layer(
            h_dict=h_dict,
            edge_index_multi=edge_index_multi,
            edge_type_list_multi=edge_type_list_multi,
            edge_index_target=edge_index_target,
            num_nodes=num_total_nodes,
            node_type_indices=node_type_indices
        )

        node1 = edges[:, 0].long()
        node2 = edges[:, 1].long()
        edge_emb = torch.cat([h[node1], h[node2]], dim=1)
        return self.edge_predictor(edge_emb).squeeze(-1)

