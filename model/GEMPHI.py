# GEM-PHI 主模型: GEMPHIModel
# 单层HGT：HGTConv
# HGT堆叠块：HGTStack
# 三塔分类：TriTowerModel
# 三塔分类中需要的边特征处理：EdgeFeatureProjector

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple
from torch_geometric.utils import softmax 


# GEM-PHI 主模型结构
class GEMPHIModel(nn.Module):
    def __init__(self, fusion_model, hgt_stack, prediction_head):
        super().__init__()
        self.fusion = fusion_model
        self.hgt = hgt_stack
        self.prediction_head = prediction_head

    def forward(self, hetero_data, host_idx, phage_idx, pair_features):
        fused_data = self.fusion(hetero_data) # 特征融合
        hgt_output_dict = self.hgt(fused_data) # HGT + JK
        
        # 拼接各层HGT输出，得到最终节点表示
        final_host_x = torch.cat(hgt_output_dict['host'], dim=-1)
        final_phage_x = torch.cat(hgt_output_dict['phage'], dim=-1)
        
        # 选择出当前批次的节点，进行预测
        logits = self.prediction_head(
            host_x=final_host_x[host_idx], 
            phage_x=final_phage_x[phage_idx],
            pair_features=pair_features
        )
        return logits



class HGTConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, metadata: Tuple[List[str], List[Tuple[str, str, str]]], edge_feature_dims: Dict[str, int], heads: int = 4, use_edge_feature: bool = True, dropout: float = 0.1, ablation_tricks: dict = None):
        super().__init__()
        self.ablation_tricks = ablation_tricks if ablation_tricks is not None else {}
        self.in_channels, self.out_channels, self.heads, self.use_edge_feature = in_channels, out_channels, heads, use_edge_feature
        self.head_dim = out_channels // heads
        assert out_channels % heads == 0
        node_types, edge_types_triplets = metadata
        self.node_types = node_types
        self.edge_types = list(edge_feature_dims.keys())
        self.edge_types_triplets = edge_types_triplets
        self.lin_key = nn.ModuleDict({nt: nn.Linear(in_channels, heads * self.head_dim) for nt in node_types})
        self.lin_value = nn.ModuleDict({nt: nn.Linear(in_channels, heads * self.head_dim) for nt in node_types})
        self.lin_query = nn.ModuleDict({nt: nn.Linear(in_channels, heads * self.head_dim) for nt in node_types})
        self.lin_edge = nn.ModuleDict()
        if use_edge_feature:
            for rel, feat_dim in edge_feature_dims.items(): self.lin_edge[rel] = nn.Linear(feat_dim, heads * self.head_dim)
        self.lin_out = nn.Linear(out_channels, out_channels)
        self.residual = nn.ModuleDict()
        for node_type in node_types:
            if in_channels != out_channels: self.residual[node_type] = nn.Linear(in_channels, out_channels)
            else: self.residual[node_type] = nn.Identity()
        self.norm = nn.ModuleDict({nt: nn.LayerNorm(out_channels) for nt in node_types})
        self.dropout_attn, self.dropout_out = nn.Dropout(dropout), nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in [self.lin_key, self.lin_value, self.lin_query]:
            for lin in layer.values(): nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
        for lin in self.lin_edge.values(): nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
        nn.init.xavier_uniform_(self.lin_out.weight); nn.init.zeros_(self.lin_out.bias)
        for res in self.residual.values():
            if isinstance(res, nn.Linear): nn.init.xavier_uniform_(res.weight); nn.init.zeros_(res.bias)

    def forward(self, data: HeteroData) -> HeteroData:
        node_messages = {nt: torch.zeros((data[nt].num_nodes, self.out_channels), device=data[nt].x.device) for nt in self.node_types}
        
        edge_types_to_process = self.edge_types
        if self.ablation_tricks.get("NO_SIMILARITY_EDGES", False):
            edge_types_to_process = [et for et in self.edge_types if et != 'similar']

        for edge_type_str in edge_types_to_process:
            relevant_triplets = [et for et in self.edge_types_triplets if et[1] == edge_type_str]
            
            for full_edge_type in relevant_triplets:
                if full_edge_type not in data.edge_index_dict: continue
                
                src_type, rel_type, dst_type = full_edge_type
                
                edge_index, edge_embed = data.edge_index_dict[full_edge_type], data[full_edge_type].get('edge_embed')
                src_feat, dst_feat = data[src_type].x, data[dst_type].x
                
                key = self.lin_key[src_type](src_feat)[edge_index[0]]
                value = self.lin_value[src_type](src_feat)[edge_index[0]]
                query = self.lin_query[dst_type](dst_feat)[edge_index[1]]
                
                if self.use_edge_feature and edge_embed is not None:
                    # --- 安全补丁：检查并修复尺寸不匹配 ---
                    num_edges = edge_index.size(1)
                    num_attrs = edge_embed.size(0)
                    if num_edges != num_attrs:
                        # print(f"  [HGTConv Safety Patch] Mismatch detected in '{full_edge_type}': {num_edges} edges vs {num_attrs} attributes. Patching...")
                        num_missing = num_edges - num_attrs
                        feature_dim = edge_embed.size(1)
                        padding = torch.zeros((num_missing, feature_dim), dtype=edge_embed.dtype, device=edge_embed.device)
                        edge_embed = torch.cat([edge_embed, padding], dim=0)
                    # --- 补丁结束 ---
                    key += self.lin_edge[rel_type](edge_embed)
                
                key = key.view(-1, self.heads, self.head_dim)
                value = value.view(-1, self.heads, self.head_dim)
                query = query.view(-1, self.heads, self.head_dim)
                
                attn_scores = (query * key).sum(dim=-1) / (self.head_dim ** 0.5)
                
                attn_weights = softmax(attn_scores, edge_index[1], num_nodes=data[dst_type].num_nodes)
                attn_weights = self.dropout_attn(attn_weights)

                weighted_value = value * attn_weights.unsqueeze(-1)
                
                dst_nodes = edge_index[1]
                msg = torch.zeros((data[dst_type].num_nodes, self.heads, self.head_dim), device=weighted_value.device)

                # --- 关键修正：恢复您原始代码中正确的索引扩展方式 ---
                index = dst_nodes.unsqueeze(-1).unsqueeze(-1).expand(-1, self.heads, self.head_dim)
                msg.scatter_add_(0, index, weighted_value)
                # --- 修正结束 ---
                
                node_messages[dst_type] += msg.view(data[dst_type].num_nodes, self.out_channels)
                
        for node_type in self.node_types:
            if node_messages[node_type].abs().sum() == 0: continue
            
            msg = self.dropout_out(self.lin_out(node_messages[node_type]))
            
            if self.ablation_tricks.get("NO_HGT_RESIDUAL", False):
                 out = msg
            else:
                 out = self.residual[node_type](data[node_type].x) + msg

            out = self.norm[node_type](out)
            data[node_type].x = F.leaky_relu(out, negative_slope=0.2)
            
        return data


# 默认两层HGT堆叠
class HGTStack(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, 
                 metadata: Tuple[List[str], List[Tuple[str, str, str]]], 
                 edge_feature_dims: Dict[str, int], num_layers: int = 2, heads: int = 4, 
                 use_edge_feature: bool = True, dropout: float = 0.25, 
                 ablation_tricks: dict = None):
        super().__init__()
        self.layers = nn.ModuleList()
        # 为第一层 HGTConv 传递参数
        self.layers.append(HGTConv(in_channels, hidden_channels, metadata, edge_feature_dims, heads, 
                                   use_edge_feature, dropout, ablation_tricks=ablation_tricks))
        # 为后续 HGTConv 层传递参数
        for _ in range(1, num_layers):
            self.layers.append(HGTConv(hidden_channels, hidden_channels, metadata, edge_feature_dims, heads, 
                                       use_edge_feature, dropout, ablation_tricks=ablation_tricks))
# 使用跳跃连接
    def forward(self, data: HeteroData) -> Dict[str, List[torch.Tensor]]:
        intermediate_outputs = {node_type: [data[node_type].x] for node_type in data.node_types}
        current_data = data
        for layer in self.layers:
            current_data = layer(current_data)
            for node_type in data.node_types:
                intermediate_outputs[node_type].append(current_data[node_type].x)
        return intermediate_outputs



# 三塔分类
class TriTowerModel(nn.Module):
    def __init__(self, node_dim: int, edge_feat_dim: int, edge_hidden_dim: int, edge_out_dim: int,
                 tower_hidden_dim: int, tower_out_dim: int, stats: dict, dropout_rate: float = 0.3,
                 ablate_edge_feature: str = None, ablation_tricks: dict = None):
        super().__init__()
        self.host_tower = nn.Sequential(
            nn.Linear(node_dim, tower_hidden_dim), nn.ReLU(), nn.LayerNorm(tower_hidden_dim),
            nn.Dropout(dropout_rate), nn.Linear(tower_hidden_dim, tower_out_dim)
        )
        self.phage_tower = nn.Sequential(
            nn.Linear(node_dim, tower_hidden_dim), nn.ReLU(), nn.LayerNorm(tower_hidden_dim),
            nn.Dropout(dropout_rate), nn.Linear(tower_hidden_dim, tower_out_dim)
        )
        self.edge_tower = EdgeFeatureProjector(
            in_dim=edge_feat_dim,
            hidden_dim=edge_hidden_dim,
            out_dim=edge_out_dim,
            stats=stats,
            ablate_edge_feature=ablate_edge_feature,
            ablation_tricks=ablation_tricks  
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * tower_out_dim + edge_out_dim, tower_hidden_dim), nn.ReLU(),
            nn.LayerNorm(tower_hidden_dim), nn.Dropout(dropout_rate), nn.Linear(tower_hidden_dim, 1)
        )
        
    def forward(self, host_x, phage_x, pair_features):
        current_device = host_x.device
        host_emb = self.host_tower(host_x)
        phage_emb = self.phage_tower(phage_x)
        edge_emb = self.edge_tower(pair_features, device=current_device)
        combined_for_classifier = torch.cat([host_emb, phage_emb, edge_emb], dim=1)
        logits = self.classifier(combined_for_classifier).squeeze(-1)
        return logits



# 三塔分类中需要的边特征处理
class EdgeFeatureProjector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, stats: dict, ablate_edge_feature: str = None, ablation_tricks: dict = None):
        super().__init__()
        self.stats = stats['edge'] if stats and 'edge' in stats else None

        self.ablate_edge_feature = ablate_edge_feature
        if self.ablate_edge_feature:
            print(f"--- [ABLATION MODE] Edge feature group '{self.ablate_edge_feature}' will be removed. ---")

        self.feat_schema = ['6-mer_d2*_sim', 'homology_total_bp', 'homology_hits', 'phage_covered_ratio', 'host_covered_ratio', 'matched_region_count', 'mean_bitscore', 'mean_match_len', 'spacer_hit_count', 'unique_spacers', 'has_hit']

        self.ablation_groups = {
            "kmer_sim": ['6-mer_d2*_sim'],
            "blast": [
                'homology_total_bp', 'homology_hits', 'phage_covered_ratio', 
                'host_covered_ratio', 'matched_region_count', 'mean_bitscore', 'mean_match_len'
            ],
            "spacer": ['spacer_hit_count', 'unique_spacers', 'has_hit']
        }

        self.log_scale_keys = {"homology_total_bp", "homology_hits", "spacer_hit_count", "unique_spacers"}

        if ablation_tricks and ablation_tricks.get("EDGE_PROJECTOR_SIMPLIFIED", False):
            # print("--- [ABLATION TRICK] Using simplified edge projector (single Linear layer). ---")
            self.projector = nn.Linear(in_dim, out_dim) 
        else:
            self.projector = nn.Sequential( 
                nn.Linear(in_dim, hidden_dim), 
                nn.ReLU(), 
                nn.LayerNorm(hidden_dim), 
                nn.Linear(hidden_dim, out_dim)
            )

    def _normalize(self, x, vmin, vmax):
        denominator = vmax - vmin
        if denominator.abs() > 1e-8: x = (x - vmin) / (denominator + 1e-8)
        else: x = torch.zeros_like(x)
        return x.clamp(0, 1)

    def _preprocess_and_stack(self, features_list: list, device: torch.device):
        import pandas as pd
        if not features_list: return torch.empty(0, len(self.feat_schema), device=device)

        stacked_features = []
        for feat_name in self.feat_schema:
            raw_values = [d.get(feat_name, 0) for d in features_list]
            series = pd.to_numeric(pd.Series(raw_values), errors='coerce').fillna(0)
            tensor = torch.tensor(series.values, dtype=torch.float, device=device).unsqueeze(1)

            if feat_name in self.log_scale_keys: tensor = torch.log1p(tensor.clamp(min=0))

            if self.stats and feat_name in self.stats:
                feat_stats = self.stats.get(feat_name)
                vmin, vmax = feat_stats['min'].to(device), feat_stats['max'].to(device)
                tensor = self._normalize(tensor, vmin, vmax)
            stacked_features.append(tensor)

        if self.ablate_edge_feature:
            features_to_zero_out = self.ablation_groups.get(self.ablate_edge_feature, [])
            for i, feat_name in enumerate(self.feat_schema):
                if feat_name in features_to_zero_out:
                    stacked_features[i] = torch.zeros_like(stacked_features[i])

        return torch.cat(stacked_features, dim=1)

    def forward(self, pair_features: list, device: torch.device):
        if not pair_features: return torch.empty(0, self.projector[-1].out_features, device=device)
        processed_tensor = self._preprocess_and_stack(pair_features, device)
        return self.projector(processed_tensor)


