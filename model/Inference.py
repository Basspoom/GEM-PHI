import sys
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from colorama import Fore, Style, init
from pathlib import Path

from .VEfusion import HeteroFeatureFusion
from .GEMPHI import HGTStack, GEMPHIModel, TriTowerModel

class EvaluationDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def load_model_and_base_data(config):
    print(f"{Fore.CYAN}Loading base model, graph, and ID maps...{Style.RESET_ALL}")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {Fore.GREEN}{device}{Style.RESET_ALL}")
    
    cfg = config['model_params']
    model_weights_path = config['paths']['model_path']

    fusion_model = HeteroFeatureFusion(d_model=cfg['d_model'], stats_path=config['paths']['stats_path'], ablation_tricks=config['ablation_tricks'])
    graph_template = torch.load(config['paths']['graph_path'], weights_only=False)
    hgt_stack = HGTStack(cfg['d_model'], cfg['d_model'], cfg['d_model'], graph_template.metadata(), {'interacts': 11, 'similar': 2}, num_layers=cfg['num_hgt_layers'], ablation_tricks=config['ablation_tricks'])
    stats = torch.load(config['paths']['stats_path'], map_location='cpu')
    prediction_head = TriTowerModel(node_dim=(cfg['num_hgt_layers'] + 1) * cfg['d_model'], edge_feat_dim=cfg['edge_feat_dim'], edge_hidden_dim=cfg['edge_hidden_dim'], edge_out_dim=cfg['edge_out_dim'], tower_hidden_dim=cfg['tower_hidden_dim'], tower_out_dim=cfg['tower_out_dim'], stats=stats, dropout_rate=cfg['dropout_rate'], ablation_tricks=config['ablation_tricks'])
    
    model = GEMPHIModel(fusion_model, hgt_stack, prediction_head).to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    print(f"{Fore.GREEN}Successfully loaded model weights.{Style.RESET_ALL}")
    
    base_graph_data = torch.load(config['paths']['graph_path'], weights_only=False)
    id_maps = torch.load(config['paths']['id_map_path'])
    
    return model, base_graph_data, id_maps, device

def augment_graph_with_new_data(base_graph, id_maps, node_info_df, pp_edges_df, hh_edges_df):
    print(f"\n{Fore.CYAN}Step 1: Augmenting graph with new nodes and edges...{Style.RESET_ALL}")
    augmented_graph = base_graph.clone()
    host_map, phage_map = id_maps['host_id_map'], id_maps['phage_id_map']

    new_nodes = node_info_df.set_index('Node_id').to_dict('index')
    for node_id, attrs in tqdm(new_nodes.items(), desc=f"{Fore.YELLOW}Adding new nodes{Style.RESET_ALL}"):
        node_type = attrs['type']
        
        if node_type == 'host' and node_id not in host_map:
            host_map[node_id] = len(host_map)
            augmented_graph['host'].x_gc = torch.cat([augmented_graph['host'].x_gc, torch.tensor([[attrs['gc_content']]], dtype=torch.float)], dim=0)
            augmented_graph['host'].x_kmer = torch.cat([augmented_graph['host'].x_kmer, torch.tensor(attrs['6mer_vector'], dtype=torch.float).unsqueeze(0)], dim=0)
            augmented_graph['host'].x_evo = torch.cat([augmented_graph['host'].x_evo, torch.tensor(attrs['NT_embedding'], dtype=torch.float).unsqueeze(0)], dim=0)
            zero_rbp = torch.zeros((1, augmented_graph['host'].x_rbp.shape[1]), dtype=torch.float)
            augmented_graph['host'].x_rbp = torch.cat([augmented_graph['host'].x_rbp, zero_rbp], dim=0)

        elif node_type == 'phage' and node_id not in phage_map:
            phage_map[node_id] = len(phage_map)
            augmented_graph['phage'].x_gc = torch.cat([augmented_graph['phage'].x_gc, torch.tensor([[attrs['gc_content']]], dtype=torch.float)], dim=0)
            augmented_graph['phage'].x_kmer = torch.cat([augmented_graph['phage'].x_kmer, torch.tensor(attrs['6mer_vector'], dtype=torch.float).unsqueeze(0)], dim=0)
            augmented_graph['phage'].x_evo = torch.cat([augmented_graph['phage'].x_evo, torch.tensor(attrs['NT_embedding'], dtype=torch.float).unsqueeze(0)], dim=0)
            
            rbp_embedding = attrs['RBP_embedding']
            if rbp_embedding is not None:
                rbp_tensor = torch.tensor(rbp_embedding, dtype=torch.float).unsqueeze(0)
            else:
                # print(f"  - Warning: Phage node {node_id} has no RBP_embedding. Using a zero vector.")
                zero_rbp = torch.zeros((1, augmented_graph['phage'].x_rbp.shape[1]), dtype=torch.float)
                rbp_tensor = zero_rbp
            
            augmented_graph['phage'].x_rbp = torch.cat([augmented_graph['phage'].x_rbp, rbp_tensor], dim=0)
            
    augmented_graph['host'].num_nodes = len(host_map)
    augmented_graph['phage'].num_nodes = len(phage_map)
    print(f"{Fore.GREEN}Graph updated. Total hosts: {augmented_graph['host'].num_nodes}, Total phages: {augmented_graph['phage'].num_nodes}{Style.RESET_ALL}")

    new_hh_edges = []; new_hh_attrs = []
    for _, row in tqdm(hh_edges_df.iterrows(), total=len(hh_edges_df), desc=f"{Fore.YELLOW}Adding HH edges{Style.RESET_ALL}"):
        src_id, dst_id = host_map.get(row['source_node']), host_map.get(row['target_node'])
        if src_id is not None and dst_id is not None:
            new_hh_edges.append([src_id, dst_id])
            new_hh_attrs.append([row['phylo_sim'], row['NT_sim']])
    
    if new_hh_edges:
        edge_index_tensor = torch.tensor(new_hh_edges, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.tensor(new_hh_attrs, dtype=torch.float)
        hh_edge_key = ('host', 'similar', 'host')
        
        num_original_edges = augmented_graph[hh_edge_key].edge_index.size(1)
        num_edge_features = edge_attr_tensor.size(1)
        # print(f"  - Rebuilding {Fore.MAGENTA}'edge_embed'{Style.RESET_ALL} for {hh_edge_key}...")
        padding = torch.zeros((num_original_edges, num_edge_features), dtype=torch.float)
        final_edge_embed = torch.cat([padding, edge_attr_tensor], dim=0)
        
        augmented_graph[hh_edge_key].edge_index = torch.cat([augmented_graph[hh_edge_key].edge_index, edge_index_tensor], dim=1)
        augmented_graph[hh_edge_key].edge_embed = final_edge_embed
        # print(f"  - Done. Final edge count: {Fore.GREEN}{augmented_graph[hh_edge_key].edge_index.size(1)}{Style.RESET_ALL}, Final embed count: {Fore.GREEN}{augmented_graph[hh_edge_key].edge_embed.size(0)}{Style.RESET_ALL}")

    new_pp_edges = []; new_pp_attrs = []
    for _, row in tqdm(pp_edges_df.iterrows(), total=len(pp_edges_df), desc=f"{Fore.YELLOW}Adding PP edges{Style.RESET_ALL}"):
        src_id, dst_id = phage_map.get(row['source_node']), phage_map.get(row['target_node'])
        if src_id is not None and dst_id is not None:
            new_pp_edges.append([src_id, dst_id])
            new_pp_attrs.append([row['NT_sim'], row['RBP_embedding_sim']])
            
    if new_pp_edges:
        edge_index_tensor = torch.tensor(new_pp_edges, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.tensor(new_pp_attrs, dtype=torch.float)
        pp_edge_key = ('phage', 'similar', 'phage')

        num_original_edges = augmented_graph[pp_edge_key].edge_index.size(1)
        num_edge_features = edge_attr_tensor.size(1)
        # print(f"  - Rebuilding {Fore.MAGENTA}'edge_embed'{Style.RESET_ALL} for {pp_edge_key}...")
        padding = torch.zeros((num_original_edges, num_edge_features), dtype=torch.float)
        final_edge_embed = torch.cat([padding, edge_attr_tensor], dim=0)

        augmented_graph[pp_edge_key].edge_index = torch.cat([augmented_graph[pp_edge_key].edge_index, edge_index_tensor], dim=1)
        augmented_graph[pp_edge_key].edge_embed = final_edge_embed
        # print(f"  - Done. Final edge count: {Fore.GREEN}{augmented_graph[pp_edge_key].edge_index.size(1)}{Style.RESET_ALL}, Final embed count: {Fore.GREEN}{augmented_graph[pp_edge_key].edge_embed.size(0)}{Style.RESET_ALL}")

    print(f"{Fore.GREEN}Graph augmentation complete.{Style.RESET_ALL}")
    return augmented_graph, {'host_id_map': host_map, 'phage_id_map': phage_map}

def prepare_prediction_data(ph_edges_df):
    print(f"\n{Fore.CYAN}Step 2: Preparing prediction pairs...{Style.RESET_ALL}")
    column_rename_map = {
        'crispr_spacer_hits': 'spacer_hit_count',
        'crispr_unique_spacers': 'unique_spacers',
        'crispr_has_hit': 'has_hit',
        'crispr_mean_bitscore': 'mean_bitscore',
        'crispr_min_evalue': 'min_evalue',
        'homology_max_length': 'homology_max_length',
        'homology_total_bp': 'homology_total_bp',
        'homology_hits': 'homology_hits',
        'phage_covered_ratio': 'phage_covered_ratio',
        'host_covered_ratio': 'host_covered_ratio',
        'matched_region_count': 'matched_region_count',
        'avg_match_identity': 'avg_match_identity',
        '6-mer_d2*_sim': 'd2star_sim'
    }
    
    rename_map_for_df = {k: v for k, v in column_rename_map.items() if k in ph_edges_df.columns}
    if rename_map_for_df:
        ph_edges_df.rename(columns=rename_map_for_df, inplace=True)
        # print(f"  - Renamed columns: {list(rename_map_for_df.keys())}")

    prediction_list = ph_edges_df.to_dict('records')
    print(f"Loaded {Fore.GREEN}{len(prediction_list)}{Style.RESET_ALL} pairs to predict.")
    return prediction_list

def collate_fn_predict(batch, host_map, phage_map):
    host_names, phage_names, features = [], [], []
    host_indices, phage_indices = [], []
    
    for item in batch:
        h_idx = host_map.get(item['target_node'])
        p_idx = phage_map.get(item['source_node'])

        if h_idx is not None and p_idx is not None:
            host_indices.append(h_idx)
            phage_indices.append(p_idx)
            host_names.append(item['target_node'])
            phage_names.append(item['source_node'])
            features.append(item)
            
    if not host_indices: return None
    
    return (host_names, phage_names, features,
            torch.tensor(host_indices, dtype=torch.long), 
            torch.tensor(phage_indices, dtype=torch.long))

@torch.no_grad()
def run_inference(model, dataloader, device, augmented_graph):
    model.eval()
    
    print(f"\n{Fore.CYAN}Step 3: Performing graph-level node enrichment...{Style.RESET_ALL}")
    graph_on_device = augmented_graph.to(device)
    fused_data = model.fusion(graph_on_device)
    hgt_output_dict = model.hgt(fused_data)
    final_host_x = torch.cat(hgt_output_dict['host'], dim=-1)
    final_phage_x = torch.cat(hgt_output_dict['phage'], dim=-1)
    print(f"{Fore.GREEN}Node enrichment complete.{Style.RESET_ALL}")

    all_results = []
    print(f"\n{Fore.CYAN}Step 4: Predicting interactions for each pair...{Style.RESET_ALL}")
    for batch_data in tqdm(dataloader, desc=f"{Fore.YELLOW}Predicting Batches{Style.RESET_ALL}"):
        if batch_data is None: continue
        host_names, phage_names, features, host_idx, phage_idx = batch_data
        
        logits = model.prediction_head(
            host_x=final_host_x[host_idx.to(device)], 
            phage_x=final_phage_x[phage_idx.to(device)], 
            pair_features=features
        )
        probabilities = torch.sigmoid(logits).cpu().numpy().flatten()
        
        for i in range(len(host_names)):
            all_results.append({
                'host': host_names[i],
                'phage': phage_names[i],
                'probability': probabilities[i]
            })

    print(f"{Fore.GREEN}Prediction complete.{Style.RESET_ALL}")
    return pd.DataFrame(all_results)