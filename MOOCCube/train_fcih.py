import json
import argparse
import networkx as nx
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_scatter import segment_coo
from torch_scatter import scatter_max
import math


# Runtime configuration
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./data/integrated_user_course_teacher_school10.json")
parser.add_argument("--feature_path", type=str, default="./data/node_features.pkl")
parser.add_argument("--bert_model", type=str, default="")
parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
parser.add_argument("--seed", type=int, default=64)
parser.add_argument("--output_dir", type=str, default="./outputs")
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--patience", type=int, default=30)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--weight_decay", type=float, default=1e-5)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

logging.basicConfig(filename='data_processing.log', level=logging.WARNING,
                    format='%(asctime)s - %(levelname)s - %(message)s')


integrated_file_path = args.data_path

with open(integrated_file_path, 'r', encoding='utf-8') as f:
    integrated_data = json.load(f)

print(f"Total records: {len(integrated_data)}")


# Build the graph
G = nx.DiGraph()


for entry in tqdm(integrated_data, desc="Building graph"):
    user = entry['user']
    course = entry['course']
    course_school = entry['course-school']
    course_teacher = entry['course-teacher']
    teacher_school = entry['teacher-school']


    if not G.has_node(user):
        G.add_node(user, node_type='user')


    if not G.has_node(course):
        G.add_node(course, node_type='course')


    if not G.has_node(teacher_school):
        G.add_node(teacher_school, node_type='teacher_school')


    if not G.has_node(course_teacher):
        G.add_node(course_teacher, node_type='course_teacher')


    G.add_edge(course, user, relation='course_user')


    G.add_edge(course_teacher, course, relation='teacher_course')


    G.add_edge(teacher_school, course_teacher, relation='school_teacher')


print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

node_type_counts = {}
for _, data in G.nodes(data=True):
    node_type = data.get('node_type', 'unknown')
    node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
print("Node type distribution:", node_type_counts)


edge_type_counts = {}
for _, _, data in G.edges(data=True):
    relation = data.get('relation', 'unknown')
    edge_type_counts[relation] = edge_type_counts.get(relation, 0) + 1
print("Edge type distribution:", edge_type_counts)


edge_types = ['course_user', 'teacher_course', 'school_teacher']
edge_type_to_idx = {edge_type: idx for idx, edge_type in enumerate(edge_types)}


feature_dim = 768
node_features = {{}}

# Generate fixed BERT node features
def get_bert_embedding(text, tokenizer, bert_model, max_length=64):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=max_length)
    inputs = {{key: value.to(device) for key, value in inputs.items()}}
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()

if os.path.exists(args.feature_path):
    with open(args.feature_path, 'rb') as f:
        node_features = pickle.load(f)
    print(f"Node features loaded from {{args.feature_path}}")
else:
    if not args.bert_model:
        raise ValueError("--bert_model is required when --feature_path does not exist")
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    bert_model = BertModel.from_pretrained(args.bert_model).to(device)
    bert_model.eval()
    for node, _ in tqdm(G.nodes(data=True), desc="Generating node features"):
        node_features[node] = get_bert_embedding(str(node), tokenizer, bert_model)
    feature_dir = os.path.dirname(os.path.abspath(args.feature_path))
    os.makedirs(feature_dir, exist_ok=True)
    with open(args.feature_path, 'wb') as f:
        pickle.dump(node_features, f)
    print(f"Node features saved to {{args.feature_path}}")


# Map nodes to integer indices
node_to_idx = {node: i for i, node in enumerate(G.nodes())}
idx_to_node = {i: node for node, i in node_to_idx.items()}
print(f"Total nodes: {len(node_to_idx)}")


edges = []
edge_type_list = []
for u, v, data in G.edges(data=True):
    edge_type = data.get('relation', 'unknown')
    edge_type_idx = edge_type_to_idx[edge_type]
    edges.append((node_to_idx[u], node_to_idx[v]))
    edge_type_list.append(edge_type_idx)


edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
edge_type_tensor = torch.tensor(edge_type_list, dtype=torch.long).to(device)


assert len(edge_type_list) == len(edges), "edge type and edge counts do not match"
assert edge_type_tensor.size(0) == len(edges), "edge type tensor and edge counts do not match"


edge_to_idx = {tuple(edge): idx for idx, edge in enumerate(edges)}


node_type_mapping = {'user': 0, 'course': 1, 'course_teacher': 2, 'teacher_school': 3}


node_type_indices = {
    node_type: torch.tensor([node_to_idx[node] for node, data in G.nodes(data=True) if data['node_type'] == node_type], dtype=torch.long).to(device)
    for node_type in node_type_mapping.keys()
}


global_to_local_index = {}
for node_type, indices in node_type_indices.items():
    for local_idx, global_idx in enumerate(indices.tolist()):
        global_to_local_index[global_idx] = (node_type, local_idx)


feature_dim = 768
features = []
for idx in range(len(node_to_idx)):
    node = idx_to_node[idx]
    if node in node_features:
        features.append(node_features[node])
    else:

        features.append(np.zeros(feature_dim, dtype=np.float32))

features = np.array(features)
features_tensor = torch.tensor(features, dtype=torch.float).to(device)


h_dict = {
    node_type: features_tensor[node_type_indices[node_type]]
    for node_type in node_type_mapping.keys()
}


node_type_dims = {node_type: feature_dim for node_type in node_type_mapping.keys()}


trans_node_type_dims = {node_type: 100 for node_type in node_type_mapping.keys()}


node_counts = {
    node_type: len(node_type_indices[node_type])
    for node_type in trans_node_type_dims.keys()
}

print("Nodes by type:", node_counts)


def set_seed(seed: int):

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
SEED = args.seed
model_save_path = os.path.join(args.output_dir, f"best_model_seed{SEED}.pt")
set_seed(SEED)


positive_edges = [(u, v) for u, v, d in G.edges(data=True)
                  if d['relation'] == 'course_user']
print(f"Positive course-user edges: {len(positive_edges)}")


users   = [n for n, d in G.nodes(data=True) if d['node_type'] == 'user']
courses = [n for n, d in G.nodes(data=True) if d['node_type'] == 'course']
positive_set = set(positive_edges)

num_positive = len(positive_edges)
num_negative = num_positive
print(f"Target negative samples: {num_negative}")


def sample_negative_edges(courses, users,
                          positive_set, num_neg, seed=42,
                          max_trials_per_edge=10):
    rng       = random.Random(seed)
    neg_set   = set()
    neg_list  = []
    attempts  = 0
    max_total = num_neg * max_trials_per_edge

    while len(neg_list) < num_neg and attempts < max_total:
        u = rng.choice(courses)
        v = rng.choice(users)
        if u != v and (u, v) not in positive_set and (u, v) not in neg_set:
            neg_set.add((u, v))
            neg_list.append((u, v))
        attempts += 1

    if len(neg_list) < num_neg:
        raise RuntimeError(f"Generated only {len(neg_list)} negative samples after {max_total} attempts")
    return neg_list


negative_candidates = sample_negative_edges(
    courses, users,
    positive_set=positive_set,
    num_neg=num_negative,
    seed=SEED)

print(f"Generated negative samples: {len(negative_candidates)}")


all_edges  = positive_edges + negative_candidates
all_labels = [1] * len(positive_edges) + [0] * len(negative_candidates)
print(f"Total samples: {len(all_edges)} ({len(positive_edges)} positive, {len(negative_candidates)} negative)")


train_edges, temp_edges, train_labels, temp_labels = train_test_split(
    all_edges,  all_labels,
    test_size=0.4, random_state=SEED, shuffle=True, stratify=all_labels)

val_edges,   test_edges,  val_labels,  test_labels = train_test_split(
    temp_edges, temp_labels,
    test_size=0.5, random_state=SEED, shuffle=True, stratify=temp_labels)

print(f"Train: {len(train_edges)}, validation: {len(val_edges)}, test: {len(test_edges)}")


# Dataset and graph inputs
class EdgeDataset(Dataset):
    def __init__(self, edges, labels):
        self.edges = edges
        self.labels = labels

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx):
        edge = self.edges[idx]
        label = self.labels[idx]
        return edge, label


def map_edges_to_idx(edges, node_to_idx):
    mapped_edges = []
    for u, v in edges:
        if u in node_to_idx and v in node_to_idx:
            mapped_edges.append((node_to_idx[u], node_to_idx[v]))
        else:
            logging.warning(f"Node {u} or {v} is not in node_to_idx.")
    return mapped_edges


train_edges_mapped = map_edges_to_idx(train_edges, node_to_idx)
val_edges_mapped = map_edges_to_idx(val_edges, node_to_idx)
test_edges_mapped = map_edges_to_idx(test_edges, node_to_idx)


train_dataset = EdgeDataset(train_edges_mapped, train_labels)
val_dataset = EdgeDataset(val_edges_mapped, val_labels)
test_dataset = EdgeDataset(test_edges_mapped, test_labels)


teacher_course_edges = [(u, v) for u, v, d in G.edges(data=True) if d['relation'] == 'teacher_course']
school_teacher_edges = [(u, v) for u, v, d in G.edges(data=True) if d['relation'] == 'school_teacher']


teacher_course_edges_mapped = map_edges_to_idx(teacher_course_edges, node_to_idx)
school_teacher_edges_mapped = map_edges_to_idx(school_teacher_edges, node_to_idx)


train_positive_edges_mapped = [edge for edge, label in zip(train_edges_mapped, train_labels) if label == 1]

train_edge_list = train_positive_edges_mapped + teacher_course_edges_mapped + school_teacher_edges_mapped
train_edge_index = torch.tensor(train_edge_list, dtype=torch.long).t().contiguous().to(device)


train_edge_types = [edge_type_to_idx['course_user']] * len(train_positive_edges_mapped) + \
                   [edge_type_to_idx['teacher_course']] * len(teacher_course_edges_mapped) + \
                   [edge_type_to_idx['school_teacher']] * len(school_teacher_edges_mapped)
train_edge_type_tensor = torch.tensor(train_edge_types, dtype=torch.long).to(device)


course_user_count = len(train_positive_edges_mapped)
teacher_course_count = len(teacher_course_edges_mapped)
school_teacher_count = len(school_teacher_edges_mapped)
edge_type_counts = {
    'course_user': course_user_count,
    'teacher_course': teacher_course_count,
    'school_teacher': school_teacher_count
}


val_edge_index = train_edge_index
val_edge_type_tensor = train_edge_type_tensor

test_edge_index = train_edge_index
test_edge_type_tensor = train_edge_type_tensor


train_target_edge_index = torch.tensor(train_positive_edges_mapped, dtype=torch.long).t().contiguous().to(device)
val_target_edge_index   = torch.tensor(train_positive_edges_mapped,   dtype=torch.long).t().contiguous().to(device)
test_target_edge_index  = torch.tensor(train_positive_edges_mapped,  dtype=torch.long).t().contiguous().to(device)

g = torch.Generator()
g.manual_seed(SEED)

batch_size = args.batch_size

def collate_fn(batch):
    edges = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    edges = torch.tensor(edges, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float)
    return edges, labels

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,generator=g, collate_fn=collate_fn,num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


import torch
import torch.nn as nn
import torch.nn.functional as F
class RMSKernel(nn.Module):
    def forward(self, x):
        if x.numel() == 0:
            return torch.tensor(0.0, device=x.device)
        return torch.sqrt(torch.mean(x**2) + 1e-8)

# Global edge-type weighting kernel
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


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max

# FCIH layer
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
        assert 'course_user' in self.et2idx, "edge_types must include 'course_user'"

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

        nn.init.zeros_(self.gate_view_1hop[-1].weight)
        nn.init.constant_(self.gate_view_1hop[-1].bias, -5.0)
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
        target_et_idx = self.et2idx['course_user']
        edge_type_list_target = torch.full((E_target,), target_et_idx, dtype=torch.long, device=device)

        # Target-relation view
        heads_target = self._encode_view(
            transformed=transformed,
            edge_index=edge_index_target.to(device),
            edge_type_indices=edge_type_list_target,
            num_nodes=num_nodes,
            et_list=['course_user'],
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


def k_hop_propagate_edgewise(edge_index, alpha, features, num_nodes, k: int):


    src, dst = edge_index[0], edge_index[1]
    h = features
    for _ in range(k):
        out = torch.zeros(num_nodes, h.size(1), device=h.device, dtype=h.dtype)
        out.index_add_(0, dst, h[src] * alpha.unsqueeze(-1))
        h = out
    return h

def optimized_three_hop_propagate(edge_index, alpha, features, num_nodes):
    return k_hop_propagate_edgewise(edge_index, alpha, features, num_nodes, k=3)

def optimized_two_hop_propagate(edge_index, alpha, features, num_nodes):
    return k_hop_propagate_edgewise(edge_index, alpha, features, num_nodes, k=2)


# Link prediction network
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


model = DualViewGraphNetwork(
    node_type_dims={'user': feature_dim, 'course': feature_dim, 'course_teacher': feature_dim, 'teacher_school': feature_dim},
    edge_type_counts=edge_type_counts,
    num_heads=2,
    node_counts=node_counts,
    edge_types=edge_types,
).to(device)

model.to(device)


criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def save_model(model, optimizer, epoch, val_accuracy, path, map_location=device):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_accuracy
    }, path)
    print(f"Model saved to {model_save_path}")


num_epochs = args.epochs
patience = args.patience
best_val_accuracy = 0.0
early_stop_counter = 0

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.2,
    patience=20,
    verbose=True
    )

import torch
import numpy as np
from sklearn.metrics import average_precision_score, ndcg_score

from typing import Optional
import torch

def reciprocal_rank(y_true: torch.Tensor, y_pred: torch.Tensor, k: Optional[int] = None) -> float:


    if y_pred.numel() == 0 or y_true.numel() == 0:
        return 0.0
    kk = min(k, y_pred.numel()) if k is not None else y_pred.numel()


    _, top_idx = torch.topk(y_pred, kk)
    top_idx = top_idx.to(y_true.device)

    for rank, idx in enumerate(top_idx):
        if y_true[idx].item() == 1:
            return 1.0 / (rank + 1)
    return 0.0


def precision_at_k(y_true, y_pred, k1):


    k = min(k1, y_pred.size(0))
    _, top_k_indices = torch.topk(y_pred, k)
    top_k_labels = y_true[top_k_indices]

    if k > 0:
        precision = top_k_labels.sum().float() / k
        return precision.item()
    else:
        return 0.0


def recall_at_k(y_true, y_pred, k1):


    k = min(k1, y_pred.size(0))
    _, top_k_indices = torch.topk(y_pred, k)
    top_k_labels = y_true[top_k_indices]

    if y_true.sum().float() > 0:
        recall = top_k_labels.sum().float() / (y_true.sum().float())
        return recall.item()
    else:
        return 0.0


def f1_at_k(y_true, y_pred, k1):


    precision = precision_at_k(y_true, y_pred, k1)
    recall = recall_at_k(y_true, y_pred, k1)

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        return f1
    else:
        return 0.0


def ap_from_ranking(y_true, y_score, k=None):


    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_score, torch.Tensor):
        y_score = y_score.detach().cpu().numpy()

    y_true = y_true.astype(int)
    if y_true.size == 0:
        return 0.0


    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]

    if k is not None:
        y_true_sorted = y_true_sorted[:k]

    R = int(y_true.sum())
    if R == 0:
        return 0.0

    hits = np.cumsum(y_true_sorted)
    ranks = np.arange(1, len(y_true_sorted) + 1)
    precision_at_i = hits / ranks

    denom = R if k is None else min(R, len(y_true_sorted))
    return float((precision_at_i * y_true_sorted).sum() / denom)


def ndcg_at_k(y_true, y_score, k):


    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_score, torch.Tensor):
        y_score = y_score.detach().cpu().numpy()

    k = int(min(k, y_score.shape[0]))
    if k <= 0:
        return 0.0

    order = np.argsort(-y_score, kind="mergesort")[:k]
    gains = y_true[order].astype(float)


    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains * discounts))


    ideal_k = int(min(k, int(y_true.sum())))
    if ideal_k == 0:
        return 0.0
    idcg = float(np.sum(discounts[:ideal_k]))

    return dcg / idcg

def gather_references_for_paper(edge_map, y_true, y_pred, pred, paper_node):
    edge_indices = edge_map.get(paper_node, [])
    if len(edge_indices) == 0:
        return (torch.tensor([], dtype=torch.float32),
                torch.tensor([], dtype=torch.float32),
                torch.tensor([], dtype=torch.float32))


    edge_indices = torch.tensor(edge_indices, dtype=torch.long, device=y_true.device)

    paper_y_true = y_true[edge_indices]
    paper_y_pred = y_pred[edge_indices]
    paper_pred = pred[edge_indices]
    return paper_y_true, paper_y_pred, paper_pred


from collections import defaultdict

def create_edge_index_map(edges):


    edge_map = defaultdict(list)
    for i, (src, _) in enumerate(edges):
        edge_map[src].append(i)
    return edge_map
def cat_if_list(x):

    if isinstance(x, (list, tuple)):
        x = [t for t in x if isinstance(t, torch.Tensor)]
        return torch.cat(x, dim=0) if len(x) > 0 else torch.tensor([], dtype=torch.float32)
    return x


test_edge_map = create_edge_index_map(test_edges_mapped)
all_test_paper_nodes = list(test_edge_map.keys())

train_edge_map = create_edge_index_map(train_edges_mapped)
all_train_paper_nodes = list(train_edge_map.keys())

val_edge_map = create_edge_index_map(val_edges_mapped)
all_val_paper_nodes = list(val_edge_map.keys())

assert len(train_edges_mapped) == len(train_labels), "train edge-label size mismatch"
assert len(val_edges_mapped) == len(val_labels), "validation edge-label size mismatch"
assert len(test_edges_mapped) == len(test_labels), "test edge-label size mismatch"

scaler = GradScaler()


train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
best_val_loss = float('inf')
best_val_MAP = 0
num_total_nodes = len(node_to_idx)
k1 = 5
k2 = 10
k3 = 20
start_epoch = 1


# Training
for epoch in range(num_epochs):
    model.train()

    epoch_train_loss = 0.0
    epoch_train_acc = 0.0
    all_train_preds = []
    all_train_labels = []
    all_train_y_preds = []
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
        for batch_edges, batch_labels in train_loader:
            batch_edges = batch_edges.to(device)
            batch_labels = batch_labels.to(device).float()
            optimizer.zero_grad()


            edge_predictions = model(
                h_dict=h_dict,
                edge_index_multi=train_edge_index,
                edge_type_list_multi=train_edge_type_tensor,
                edges=batch_edges,
                num_total_nodes=num_total_nodes,
                node_type_indices=node_type_indices,
            )


            loss = criterion(edge_predictions, batch_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            preds = (torch.sigmoid(edge_predictions) > 0.5).long()
            y_preds=torch.sigmoid(edge_predictions)
            all_train_preds.append(preds.cpu())
            all_train_labels.append(batch_labels.cpu())
            all_train_y_preds.append(y_preds.cpu())
            epoch_train_loss += loss.item()

            batch_accuracy = accuracy_score(batch_labels.cpu().numpy(), preds.cpu().numpy())
            epoch_train_acc += batch_accuracy


            pbar.set_postfix({"Loss": loss.item(), "Accuracy": batch_accuracy})
            pbar.update(1)


    all_train_preds = torch.cat(all_train_preds)
    all_train_labels = torch.cat(all_train_labels)
    all_train_y_preds = torch.cat(all_train_y_preds)

    train_accuracy = accuracy_score(all_train_labels.numpy(), all_train_preds.numpy())
    train_precision = precision_score(all_train_labels.numpy(), all_train_preds.numpy())
    train_f1 = f1_score(all_train_labels.numpy(), all_train_preds.numpy())
    avg_train_loss = epoch_train_loss / len(train_loader)

    print(f"Train Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}, "
          f"Precision: {train_precision:.4f}, F1: {train_f1:.4f}, ")


    model.eval()
    epoch_val_loss = 0.0
    all_val_preds = []
    all_val_labels = []
    all_val_y_preds = []
    with torch.no_grad():
        with tqdm(total=len(val_loader), desc=f"Validation {epoch + 1}/{num_epochs}", unit="batch") as pbar:
            for batch_edges, batch_labels in val_loader:
                batch_edges = batch_edges.to(device)
                batch_labels = batch_labels.to(device).float()
                val_predictions = model(
                    h_dict=h_dict,
                    edge_index_multi=val_edge_index,
                    edge_type_list_multi=val_edge_type_tensor,
                    edges=batch_edges,
                    num_total_nodes=num_total_nodes,
                    node_type_indices=node_type_indices,
                )
                preds = (torch.sigmoid(val_predictions) > 0.5).long()
                y_preds=torch.sigmoid(val_predictions)
                all_val_preds.append(preds.cpu())
                all_val_labels.append(batch_labels.cpu())
                all_val_y_preds.append(y_preds.cpu())


                batch_loss = criterion(val_predictions, batch_labels)
                epoch_val_loss += batch_loss.item()

                pbar.update(1)


    avg_val_loss = epoch_val_loss / len(val_loader)


    all_val_preds  = cat_if_list(all_val_preds)
    all_val_labels = cat_if_list(all_val_labels)
    all_val_y_preds= cat_if_list(all_val_y_preds)


    val_accuracy = accuracy_score(all_val_labels.numpy(), all_val_preds.numpy())
    val_precision = precision_score(all_val_labels.numpy(), all_val_preds.numpy())
    val_recall = recall_score(all_val_labels.numpy(), all_val_preds.numpy())
    val_f1 = f1_score(all_val_labels.numpy(), all_val_preds.numpy())


    val_mean_precision1 = 0.0
    val_mean_recall1 = 0.0
    val_mean_f11 = 0.0
    val_map1 = 0.0
    val_mean_ndcg1 = 0.0
    val_mean_rr1 = 0.0
    val_mean_precision2 = 0.0
    val_mean_recall2 = 0.0
    val_mean_f12 = 0.0
    val_map2 = 0.0
    val_mean_ndcg2 = 0.0
    val_mean_rr2 = 0.0
    valid_val_count = 0

    for val_paper_node in all_val_paper_nodes:
        paper_y_true_val, paper_y_pred_val, paper_pred_val = gather_references_for_paper(
            val_edge_map, all_val_labels, all_val_preds, all_val_y_preds, val_paper_node
        )

        if paper_y_true_val.numel() == 0 or paper_y_true_val.sum() == 0:
            continue

        precision = precision_at_k(paper_y_true_val, paper_pred_val, k1)
        recall = recall_at_k(paper_y_true_val, paper_pred_val, k1)
        f1 = f1_at_k(paper_y_true_val, paper_pred_val, k1)
        ap = ap_from_ranking(paper_y_true_val, paper_pred_val, None)
        ndcg = ndcg_at_k(paper_y_true_val, paper_pred_val, k1)
        rr = reciprocal_rank(paper_y_true_val, paper_pred_val, None)

        val_mean_precision1 += precision
        val_mean_recall1 += recall
        val_mean_f11 += f1
        val_map1 += ap
        val_mean_ndcg1 += ndcg
        val_mean_rr1 += rr
        valid_val_count += 1
        precision2 = precision_at_k(paper_y_true_val, paper_pred_val, k2)
        recall2 = recall_at_k(paper_y_true_val, paper_pred_val, k2)
        f12 = f1_at_k(paper_y_true_val, paper_pred_val, k2)
        ap2 = ap_from_ranking(paper_y_true_val, paper_pred_val, None)
        ndcg2 = ndcg_at_k(paper_y_true_val, paper_pred_val, k2)
        rr2 = reciprocal_rank(paper_y_true_val, paper_pred_val, None)
        val_mean_precision2 += precision2
        val_mean_recall2 += recall2
        val_mean_f12 += f12
        val_map2 += ap2
        val_mean_ndcg2 += ndcg2
        val_mean_rr2 += rr2

    if valid_val_count > 0:
        val_mean_precision1 /= valid_val_count
        val_mean_recall1   /= valid_val_count
        val_mean_f11       /= valid_val_count
        val_map1          /= valid_val_count
        val_mean_ndcg1     /= valid_val_count
        val_mean_rr1       /= valid_val_count
        val_mean_precision2 /= valid_val_count
        val_mean_recall2   /= valid_val_count
        val_mean_f12       /= valid_val_count
        val_map2           /= valid_val_count
        val_mean_ndcg2     /= valid_val_count
        val_mean_rr2       /= valid_val_count
    else:
        val_mean_precision = val_mean_recall = val_mean_f1 = val_map = val_mean_ndcg = val_mean_rr = 0.0

    print(f"avg_val_loss: {avg_val_loss}, val_Precision@{k1}: {val_mean_precision1:.4f}, val_Recall@{k1}: {val_mean_recall1:.4f}, "
          f"val_F1@{k1}: {val_mean_f11:.4f}, val_MAP: {val_map1:.4f}, "
          f"val_nDCG@{k1}: {val_mean_ndcg1:.4f}, val_MRR: {val_mean_rr1:.4f}")

    print(f"val_Precision@{k2}: {val_mean_precision2:.4f}, val_Recall@{k2}: {val_mean_recall2:.4f}, "
          f"val_F1@{k2}: {val_mean_f12:.4f}, val_MAP: {val_map2:.4f}, "
          f"val_nDCG@{k2}: {val_mean_ndcg2:.4f}, val_MRR: {val_mean_rr2:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}, "
          f"Precision: {val_precision:.4f}, F1: {val_f1:.4f},")

    scheduler.step(val_map2)


    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current learning rate: {current_lr}")


    if val_map2 > best_val_MAP:
        best_val_MAP = val_map2
        early_stop_counter = 0
        save_model(model, optimizer, epoch + 1, best_val_MAP, model_save_path)
    else:
        early_stop_counter += 1
        print(f"Validation loss did not improve, early stop counter: {early_stop_counter}/{patience}")

    if early_stop_counter >= patience:
        print(f"Validation loss has not improved for {patience} consecutive epochs, stopping training.")
        break
print("Training completed")


# Evaluation
def load_model(model, path):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    val_MAP = checkpoint['val_accuracy']
    print(f"Model loaded successfully, epoch: {epoch}, val_accuracy: {best_val_MAP}")
    return model


model = load_model(model, model_save_path)
model.eval()
all_test_preds = []
all_test_labels = []
test_loss = 0.0
all_test_y_preds = []
with torch.no_grad():
    with tqdm(total=len(test_loader), desc="Test", unit="batch") as pbar:
        for batch_edges, batch_labels in test_loader:
            batch_edges = batch_edges.to(device)
            batch_labels = batch_labels.to(device).float()


            test_predictions = model(
                h_dict=h_dict,
                edge_index_multi=test_edge_index,
                edge_type_list_multi=test_edge_type_tensor,
                edges=batch_edges,
                num_total_nodes=num_total_nodes,
                node_type_indices=node_type_indices,
            )
            preds = (torch.sigmoid(test_predictions) > 0.5).long()
            y_preds=torch.sigmoid(test_predictions)
            all_test_preds.append(preds.cpu())
            all_test_labels.append(batch_labels.cpu())
            all_test_y_preds.append(y_preds.cpu())

            batch_loss = criterion(test_predictions, batch_labels)
            test_loss += batch_loss.item()

            pbar.update(1)


if len(all_test_preds) > 0 and len(all_test_labels) > 0:
    all_test_preds   = cat_if_list(all_test_preds)
    all_test_labels  = cat_if_list(all_test_labels)
    all_test_y_preds = cat_if_list(all_test_y_preds)

    test_accuracy = accuracy_score(all_test_labels.numpy(), all_test_preds.numpy())
    test_precision = precision_score(all_test_labels.numpy(), all_test_preds.numpy())
    test_f1 = f1_score(all_test_labels.numpy(), all_test_preds.numpy())
    print(f"Test Loss: {test_loss / len(test_loader):.4f}, Accuracy: {test_accuracy:.4f}, "
          f"Precision: {test_precision:.4f}, F1: {test_f1:.4f},")
else:
    print("Test loader produced no predictions/labels.")


test_mean_precision1 = 0.0
test_mean_recall1 = 0.0
test_mean_f11 = 0.0
test_map1 = 0.0
test_mean_ndcg1 = 0.0
test_mean_rr1 = 0.0
test_mean_precision2 = 0.0
test_mean_recall2 = 0.0
test_mean_f12 = 0.0
test_map2 = 0.0
test_mean_ndcg2 = 0.0
test_mean_rr2 = 0.0
test_mean_precision3 = 0.0
test_mean_recall3 = 0.0
test_mean_f13 = 0.0
test_map3 = 0.0
test_mean_ndcg3 = 0.0
test_mean_rr3 = 0.0
valid_count = 0

for paper_node in all_test_paper_nodes:
    paper_y_true, paper_y_pred, paper_pred = gather_references_for_paper(
        test_edge_map, all_test_labels, all_test_preds, all_test_y_preds, paper_node
    )

    if paper_y_true.numel() == 0 or paper_y_true.sum() == 0:
        continue

    precision = precision_at_k(paper_y_true, paper_pred, k1)
    recall = recall_at_k(paper_y_true, paper_pred, k1)
    f1 = f1_at_k(paper_y_true, paper_pred, k1)
    ap = ap_from_ranking(paper_y_true, paper_pred, None)
    ndcg = ndcg_at_k(paper_y_true, paper_pred, k1)
    rr = reciprocal_rank(paper_y_true, paper_pred, None)

    test_mean_precision1 += precision
    test_mean_recall1 += recall
    test_mean_f11 += f1
    test_map1 += ap
    test_mean_ndcg1 += ndcg
    test_mean_rr1 += rr
    valid_count += 1
    precision2 = precision_at_k(paper_y_true, paper_pred, k2)
    recall2 = recall_at_k(paper_y_true, paper_pred, k2)
    f12 = f1_at_k(paper_y_true, paper_pred, k2)
    ap2 = ap_from_ranking(paper_y_true, paper_pred, None)
    ndcg2 = ndcg_at_k(paper_y_true, paper_pred, k2)
    rr2 = reciprocal_rank(paper_y_true, paper_pred, None)
    test_mean_precision2 += precision2
    test_mean_recall2 += recall2
    test_mean_f12 += f12
    test_map2 += ap2
    test_mean_ndcg2 += ndcg2
    test_mean_rr2 += rr2
    precision3 = precision_at_k(paper_y_true, paper_pred, k3)
    recall3 = recall_at_k(paper_y_true, paper_pred, k3)
    f13 = f1_at_k(paper_y_true, paper_pred, k3)
    ap3 = ap_from_ranking(paper_y_true, paper_pred, None)
    ndcg3 = ndcg_at_k(paper_y_true, paper_pred, k3)
    rr3 = reciprocal_rank(paper_y_true, paper_pred, None)
    test_mean_precision3 += precision3
    test_mean_recall3 += recall3
    test_mean_f13 += f13
    test_map3 += ap3
    test_mean_ndcg3 += ndcg3
    test_mean_rr3 += rr3
if valid_count > 0:
    test_mean_precision1 /= valid_count
    test_mean_recall1   /= valid_count
    test_mean_f11       /= valid_count
    test_map1          /= valid_count
    test_mean_ndcg1     /= valid_count
    test_mean_rr1       /= valid_count
    test_mean_precision2 /= valid_count
    test_mean_recall2   /= valid_count
    test_mean_f12       /= valid_count
    test_map2           /= valid_count
    test_mean_ndcg2     /= valid_count
    test_mean_rr2       /= valid_count
    test_mean_precision3 /= valid_count
    test_mean_recall3   /= valid_count
    test_mean_f13       /= valid_count
    test_map3           /= valid_count
    test_mean_ndcg3     /= valid_count
    test_mean_rr3       /= valid_count


print(f"test_Precision@{k1}: {test_mean_precision1:.4f}, test_Recall@{k1}: {test_mean_recall1:.4f}, "
      f"test_F1@{k1}: {test_mean_f11:.4f}, test_MAP: {test_map1:.4f}, "
      f"test_nDCG@{k1}: {test_mean_ndcg1:.4f}, test_MRR: {test_mean_rr1:.4f}")

print(f"test_Precision@{k2}: {test_mean_precision2:.4f}, test_Recall@{k2}: {test_mean_recall2:.4f}, "
      f"test_F1@{k2}: {test_mean_f12:.4f}, test_MAP: {test_map2:.4f}, "
      f"test_nDCG@{k2}: {test_mean_ndcg2:.4f}, test_MRR: {test_mean_rr2:.4f}")
print(f"test_Precision@{k3}: {test_mean_precision3:.4f}, test_Recall@{k3}: {test_mean_recall3:.4f}, "
      f"test_F1@{k3}: {test_mean_f13:.4f}, test_MAP: {test_map3:.4f}, "
      f"test_nDCG@{k3}: {test_mean_ndcg3:.4f}, test_MRR: {test_mean_rr3:.4f}")
