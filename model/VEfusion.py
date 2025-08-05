# 四类节点特征融合: NodeFeatureFusion
# 三类边特征融合：
# 节点 + 边 总代码：HeteroFeatureFusion

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData


class NodeFeatureFusion(nn.Module):
    def __init__(self, d_model=512, stats=None, ablate_feature: str = None, ablation_tricks: dict = None):
        super().__init__()
        self.stats = stats
        self.ablate_feature = ablate_feature  
        if self.ablate_feature:
            print(f"--- [ABLATION MODE] Feature '{self.ablate_feature}' will be removed during fusion. ---")
        
        self.gc_expand = nn.Sequential(nn.Linear(1, 32), nn.ReLU())
        
        self.proj_kmer = self._create_proj(4096, d_model)
        self.proj_evo  = self._create_proj(1024, d_model) # 其实是NT30，是1024维
        self.proj_gc   = nn.Sequential(nn.Linear(32, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.proj_rbp  = nn.Sequential(nn.Linear(1280, d_model), nn.LayerNorm(d_model), nn.GELU())
        fusion_input_dim = d_model * 4
        

        if ablation_tricks and ablation_tricks.get("NODE_FUSION_SIMPLIFIED", False):
            # print("--- [ABLATION TRICK] Using simplified node fusion (single Linear layer). ---")
            self.fusion_head = nn.Linear(fusion_input_dim, d_model)
        else:
            self.fusion_head = nn.Sequential( 
                nn.Linear(fusion_input_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model)
            )

        self.missing_rbp_embedding = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.type_embedding = nn.Embedding(2, d_model) # 0 for phage, 1 for host
        self.final_norm = nn.LayerNorm(d_model)

    def _create_proj(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, 1024), nn.GELU(), 
            nn.Linear(1024, out_dim), nn.LayerNorm(out_dim), nn.GELU()
        )

    def _preprocess_features(self, features, node_type_str, device):
        # 辅助函数，用于标准化输入特征
        out = {}
        stats_node = self.stats['node'][node_type_str]
        
        for feat_name in ['kmer', 'evo']:
            if feat_name in features and features[feat_name] is not None:
                x = torch.log2((1 + features[feat_name]).clamp(min=1e-8))
                mean, std = stats_node[feat_name]['mean'].to(device), stats_node[feat_name]['std'].to(device)
                x_norm = torch.zeros_like(x)
                non_zero_std_mask = std > 1e-8
                x_norm[:, non_zero_std_mask] = (x[:, non_zero_std_mask] - mean[non_zero_std_mask]) / std[non_zero_std_mask]
                out[feat_name] = x_norm

        if 'gc' in features and features['gc'] is not None:
            out['gc'] = self.gc_expand(features['gc'].clamp(0, 1))
        if 'rbp' in features and features['rbp'] is not None:
            out['rbp'] = features['rbp']
        return out

    def forward(self, features, node_type_str, node_type_tensor):
        device = next(iter(f for f in features.values() if f is not None)).device
        prep_features = self._preprocess_features(features, node_type_str, device)

        # --- K-mer ---
        proj_kmer = self.proj_kmer(prep_features['kmer'])
        if self.ablate_feature == 'kmer':
            proj_kmer = torch.zeros_like(proj_kmer) 

        # --- Evo ---
        proj_evo = self.proj_evo(prep_features['evo'])
        if self.ablate_feature == 'evo':
            proj_evo = torch.zeros_like(proj_evo) 

        # --- GC ---
        proj_gc = self.proj_gc(prep_features['gc'])
        if self.ablate_feature == 'gc':
            proj_gc = torch.zeros_like(proj_gc) 

        # --- RBP ---
        # 宿主没有RBP，默认是零向量
        if node_type_str == 'host':
            proj_rbp = torch.zeros_like(proj_kmer)
        else: # 噬菌体
            rbp_feat = prep_features.get('rbp')
            if rbp_feat is not None:
                rbp_mask = (rbp_feat.abs().sum(dim=1) < 1e-8)
                proj_rbp_tmp = self.proj_rbp(rbp_feat)
                proj_rbp_tmp[rbp_mask] = self.missing_rbp_embedding
                proj_rbp = proj_rbp_tmp
            else:
                proj_rbp = self.missing_rbp_embedding.expand(proj_kmer.size(0), -1)

            if self.ablate_feature == 'rbp':
                proj_rbp = torch.zeros_like(proj_rbp)


        concatenated_features = torch.cat([proj_kmer, proj_evo, proj_gc, proj_rbp], dim=1)

        fused_embedding = self.fusion_head(concatenated_features)

        final_embedding = self.final_norm(fused_embedding + self.type_embedding(node_type_tensor))
        return F.gelu(final_embedding)



class HeteroFeatureFusion(nn.Module):
    def __init__(self, d_model=512, stats_path=None, ablate_feature: str = None, ablation_tricks: dict = None):
        super().__init__()
        try:
            stats = torch.load(stats_path) if stats_path else None
        except FileNotFoundError:
            stats = None
            print(f"Warning: Stats file not found at {stats_path}. Running without normalization statistics.")

        self.node_fusion = NodeFeatureFusion(d_model, stats=stats, ablate_feature=ablate_feature, ablation_tricks=ablation_tricks)
        self.edge_fusion = FeatureMatrixModule(stats=stats) 

    def forward(self, data: HeteroData) -> HeteroData:
        # --- 节点特征融合 ---
        for i, node_type in enumerate(['phage', 'host']):
            node_feat_dict = {k.split('x_')[1]: v for k, v in data[node_type].items() if k.startswith('x_')}
            num_nodes = data[node_type].num_nodes
            
            fused_list = []
            batch_size = 512 # 可根据显存调整
            for j in range(0, num_nodes, batch_size):
                batch_feat = {k: v[j:j+batch_size] for k, v in node_feat_dict.items()}
                device = next(iter(f for f in batch_feat.values() if f is not None)).device
                type_tensor = torch.full((len(next(iter(batch_feat.values()))),), i, dtype=torch.long, device=device)
                
                fused_list.append(self.node_fusion(batch_feat, node_type, type_tensor))
                
            data[node_type].x = torch.cat(fused_list, dim=0)
        
        # --- 边特征融合 ---
        edge_type_str_map = {
            ('host', 'interacts', 'phage'): "phage-host",
            ('phage', 'interacts', 'host'): "phage-host",
            ('phage', 'similar', 'phage'): "phage-phage",
            ('host', 'similar', 'host'): "host-host"
        }
        for edge_type in data.edge_types:
            if edge_type not in edge_type_str_map: continue
            edge_dict = {k: v for k, v in data[edge_type].items() if k != "edge_index"}
            if not edge_dict: continue

            edge_type_str = edge_type_str_map.get(edge_type)
            data[edge_type].edge_embed, _, _ = self.edge_fusion(edge_dict, edge_type_str)
            
        return data



# 三类边特征融合
class FeatureMatrixModule(nn.Module):
    def __init__(self, stats=None):
        super().__init__()
        self.stats = stats
        self.feat_schema = { "phage-host": ['6-mer_d2*_sim', 'homology_total_bp', 'homology_hits', 'phage_covered_ratio', 'host_covered_ratio', 'matched_region_count', 'mean_bitscore', 'mean_match_len', 'spacer_hit_count', 'unique_spacers', 'has_hit'], "phage-phage": ['NT_30random_sim', 'RBP_embedding_sim'], "host-host": ['NT_30random_sim', 'phylo_sim']}
        self.log_scale_keys = {"homology_total_bp", "homology_hits", "spacer_hit_count", "unique_spacers"}
        self.evidence_gate = nn.ModuleDict({key: nn.Sequential(nn.Linear(len(self.feat_schema[key]), len(self.feat_schema[key])), nn.Sigmoid()) for key in self.feat_schema})
        self.post_proj = nn.ModuleDict({key: nn.Sequential(nn.Linear(len(self.feat_schema[key]), len(self.feat_schema[key])), nn.LayerNorm(len(self.feat_schema[key])), nn.GELU()) for key in self.feat_schema})
    def _normalize(self, x, vmin, vmax):
        vmin, vmax = vmin.to(x.device), vmax.to(x.device)
        denominator = vmax - vmin
        if denominator.abs() > 1e-8: x = (x - vmin) / (denominator + 1e-8)
        else: x = torch.zeros_like(x)
        return x.clamp(0, 1)
    def _preprocess_feat(self, name, x):
        x = x.float(); 
        if name in self.log_scale_keys: x = torch.log1p(x.clamp(min=0))
        return x
    def forward(self, features: dict, edge_type: str):
        schema = self.feat_schema[edge_type]; feats, masks = [], []
        for feat in schema:
            raw_x = features.get(feat)
            if raw_x is None: N = next(iter(features.values())).size(0); feats.append(torch.zeros(N, 1, device=next(iter(features.values())).device)); masks.append(torch.ones(N, 1, device=next(iter(features.values())).device)); continue
            current_mask = torch.isnan(raw_x); x = raw_x.clone(); x[current_mask] = 0
            if feat in {"RBP_embedding_sim", "phylo_sim"}: x_norm = x
            else:
                x_processed = self._preprocess_feat(feat, x)
                if self.stats and self.stats.get('edge'):
                    feat_stats = self.stats['edge'].get(feat)
                    if feat_stats:
                        x_norm = self._normalize(x_processed, feat_stats['min'], feat_stats['max'])
                    else: x_norm = x_processed
                else: x_norm = x_processed
            feats.append(x_norm); masks.append(current_mask.float())
        edge_feat = torch.cat(feats, dim=1); edge_mask = torch.cat(masks, dim=1)
        edge_feat = edge_feat * (1 - edge_mask); gate = self.evidence_gate[edge_type](edge_feat)
        edge_feat = edge_feat * gate; edge_embed = self.post_proj[edge_type](edge_feat)
        return edge_embed, edge_mask, gate