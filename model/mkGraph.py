import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData


def fill_missing_with_zero(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dropna().empty:
            continue
        sample = df[col].dropna().iloc[0]
        if isinstance(sample, (int, float, np.integer, np.floating)):
            df[col] = df[col].fillna(0).astype(type(sample))
        elif isinstance(sample, (list, np.ndarray)):
            dim = len(sample)
            def fill_vec(x):
                if isinstance(x, (list, np.ndarray)) and len(x) == dim:
                    return np.array(x)
                return np.zeros(dim, dtype=np.float32)
            df[col] = df[col].apply(fill_vec)
    return df

def fill_array_column(series, dim):
    return np.stack([
        np.array(x) if isinstance(x, (list, np.ndarray)) and len(x) == dim else np.zeros(dim)
        for x in series
    ])

def fill_edge_col(df, mask, col, shape=1):
    vals = df.loc[mask, col]
    if shape == 1:
        return vals.fillna(0.0).astype(float).values.reshape(-1, 1)
    else:
        return np.stack([
            np.array(x) if isinstance(x, (list, np.ndarray)) and len(x)==shape else np.zeros(shape)
            for x in vals
        ])

def safe_map(series, id_map):
    mapped = series.map(id_map)
    mask = mapped.notna()
    return mapped.astype('Int64'), mask

def get_edge_index(df, src_map, tgt_map, src_col, tgt_col):
    src = df[src_col].map(src_map)
    tgt = df[tgt_col].map(tgt_map)
    valid = src.notna() & tgt.notna()
    return np.vstack([src[valid].astype(int), tgt[valid].astype(int)]), valid


def build_hetero_graph(node_path, ph_path, pp_path, hh_path, save_path=None):
    node_df = fill_missing_with_zero(pd.read_pickle(node_path))
    ph_df   = fill_missing_with_zero(pd.read_pickle(ph_path))
    pp_df   = fill_missing_with_zero(pd.read_pickle(pp_path))
    hh_df   = fill_missing_with_zero(pd.read_pickle(hh_path))

    phage_df = node_df[node_df['type'] == 'phage'].reset_index(drop=True)
    host_df  = node_df[node_df['type'] == 'host'].reset_index(drop=True)
    phage_id_map = {nid: i for i, nid in enumerate(phage_df['Node_id'])}
    host_id_map  = {nid: i for i, nid in enumerate(host_df['Node_id'])}

    # 构建边索引
    hp_src, src_mask = safe_map(ph_df['source_node'], host_id_map)
    hp_tgt, tgt_mask = safe_map(ph_df['target_node'], phage_id_map)
    hp_mask = src_mask & tgt_mask
    hp_edge_index = np.vstack([hp_src[hp_mask].astype(int), hp_tgt[hp_mask].astype(int)])
    ph_edge_index = np.flip(hp_edge_index, axis=0)

    pp_edge_index, pp_mask = get_edge_index(pp_df, phage_id_map, phage_id_map, 'source_node', 'target_node')
    hh_edge_index, hh_mask = get_edge_index(hh_df, host_id_map, host_id_map, 'source_node', 'target_node')

    # 节点特征
    kmer_dim = len(phage_df['6mer_vector'].iloc[0])
    evo_dim  = len(phage_df['NT_30random'].iloc[0])
    rbp_dim  = 1280

    data = HeteroData()

    data['phage'].x_kmer = torch.tensor(fill_array_column(phage_df['6mer_vector'], kmer_dim), dtype=torch.float)
    data['phage'].x_gc   = torch.tensor(phage_df['gc_content'].values).unsqueeze(1).float()
    data['phage'].x_evo  = torch.tensor(fill_array_column(phage_df['NT_30random'], evo_dim), dtype=torch.float)
    data['phage'].x_rbp  = torch.tensor(np.stack([
        x if isinstance(x, np.ndarray) and len(x)==rbp_dim else np.zeros(rbp_dim)
        for x in phage_df['RBP_embedding']
    ]), dtype=torch.float)

    data['host'].x_kmer = torch.tensor(fill_array_column(host_df['6mer_vector'], kmer_dim), dtype=torch.float)
    data['host'].x_gc   = torch.tensor(host_df['gc_content'].values).unsqueeze(1).float()
    data['host'].x_evo  = torch.tensor(fill_array_column(host_df['NT_30random'], evo_dim), dtype=torch.float)
    torch.manual_seed(777)
    data['host'].x_rbp  = torch.nn.Parameter(torch.randn(len(host_df), rbp_dim)).data

    # 构建边索引
    data['host', 'interacts', 'phage'].edge_index = torch.tensor(hp_edge_index, dtype=torch.long)
    data['phage', 'interacts', 'host'].edge_index = torch.tensor(ph_edge_index, dtype=torch.long)
    data['phage', 'similar', 'phage'].edge_index = torch.tensor(pp_edge_index, dtype=torch.long)
    data['host',  'similar', 'host'].edge_index = torch.tensor(hh_edge_index, dtype=torch.long)

    # 边特征
    ph_cols = [
        'evidence_score', '6-mer_d2*_sim', '6-mer_cos_sim',
        'homology_total_bp', 'homology_hits', 'phage_covered_ratio', 'host_covered_ratio', 'matched_region_count',
        'spacer_hit_count', 'unique_spacers', 'has_hit', 'mean_bitscore', 'mean_match_len'
    ]
    for col in ph_cols:
        feat = fill_edge_col(ph_df, hp_mask, col)
        data['host', 'interacts', 'phage'][col] = torch.tensor(feat, dtype=torch.float)
        data['phage', 'interacts', 'host'][col] = torch.tensor(feat, dtype=torch.float)

    for edge, df, mask, cols in [
        (('phage', 'similar', 'phage'), pp_df, pp_mask, ['NT_30random_sim', 'RBP_embedding_sim']),
        (('host', 'similar', 'host'), hh_df, hh_mask, ['NT_30random_sim', 'phylo_sim'])
    ]:
        for col in cols:
            feat = fill_edge_col(df, mask, col)
            data[edge][col] = torch.tensor(feat, dtype=torch.float)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(data, save_path)
        print(f"✅ 图数据保存至: {save_path}")

    # return data
    return data, phage_id_map, host_id_map  # 添加返回映射字典
