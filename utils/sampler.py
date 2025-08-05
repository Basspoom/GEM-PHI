import pandas as pd
import numpy as np
import torch
from collections import defaultdict
import random

import pandas as pd
import numpy as np
import torch
from collections import defaultdict
import random

class NegativePoolSampler:
    def __init__(self, train_data: object, neg_pool_path: str, 
                 host_id_map: dict, phage_id_map: dict, device='cpu'):
        self.device = device
        self.host_id_map = host_id_map
        self.phage_id_map = phage_id_map
        
        pos_edge_index = train_data['host', 'interacts', 'phage'].edge_index
        self.pos_edges = pos_edge_index.T.cpu().numpy()

        print("Building lookup dictionary for positive sample features...")
        self.pos_features_map = {}
        pos_edge_data = train_data['host', 'interacts', 'phage']
        feature_keys = [key for key in pos_edge_data.keys() if key != 'edge_index']
        features_squeezed = {key: pos_edge_data[key].squeeze(-1).cpu().numpy() for key in feature_keys}
        for i in range(len(self.pos_edges)):
            edge = tuple(self.pos_edges[i])
            self.pos_features_map[edge] = {key: features_squeezed[key][i] for key in feature_keys}
        print("Positive feature lookup ready.")


        print(f"Loading negative sample pool from {neg_pool_path}...")
        neg_df = pd.read_pickle(neg_pool_path)
        self.neg_records = neg_df.to_dict('records')
        self.neg_indices = list(range(len(self.neg_records)))

        print("Building integer-based lookup dictionaries for faster contrastive sampling...")
        self.int_phage_to_neg_indices = defaultdict(list)
        self.int_host_to_neg_indices = defaultdict(list)

        for i, row in enumerate(self.neg_records):
            phage_id_str = row.get('target_node')
            host_id_str = row.get('source_node')

            phage_id_int = self.phage_id_map.get(phage_id_str)
            host_id_int = self.host_id_map.get(host_id_str)
            
            if phage_id_int is not None:
                self.int_phage_to_neg_indices[phage_id_int].append(i)
            if host_id_int is not None:
                self.int_host_to_neg_indices[host_id_int].append(i)

        print("Sampler initialized successfully.")
        print(f"  - Positive samples: {len(self.pos_edges)}")
        print(f"  - Negative pool size: {len(self.neg_records)}")


    def sample_contrastive(self, batch_size: int, K: int = 10):
        pos_sample_indices = np.random.choice(len(self.pos_edges), batch_size, replace=False)
        pos_sample = self.pos_edges[pos_sample_indices]
        pos_hosts, pos_phages = pos_sample[:, 0], pos_sample[:, 1]
        
        pos_features = [self.pos_features_map.get(tuple(edge), {}) for edge in pos_sample]

        neg_hosts_list, neg_phages_list = [], []
        neg_host_pair_features, neg_phage_pair_features = [], []

        for i in range(batch_size):
            h_pos_idx, p_pos_idx = pos_hosts[i], pos_phages[i]           
            candidate_indices_h = self.int_phage_to_neg_indices.get(p_pos_idx, [])

            neg_h_for_p, neg_h_feats = [], []
            if not candidate_indices_h:
                neg_h_for_p = [0] * K
                neg_h_feats = [{}] * K
            else:
                sampled_indices = random.choices(candidate_indices_h, k=K)
                for neg_idx in sampled_indices:
                    record = self.neg_records[neg_idx]
                    if record.get('source_node') in self.host_id_map:
                        neg_h_for_p.append(self.host_id_map[record['source_node']])
                        neg_h_feats.append(record)
            while len(neg_h_for_p) < K:
                neg_h_for_p.append(0); neg_h_feats.append({})
            
            neg_hosts_list.append(neg_h_for_p[:K])
            neg_host_pair_features.append(neg_h_feats[:K])


            candidate_indices_p = self.int_host_to_neg_indices.get(h_pos_idx, [])

            neg_p_for_h, neg_p_feats = [], []
            if not candidate_indices_p:
                neg_p_for_h = [0] * K
                neg_p_feats = [{}] * K
            else:
                sampled_indices = random.choices(candidate_indices_p, k=K)
                for neg_idx in sampled_indices:
                    record = self.neg_records[neg_idx]
                    if record.get('target_node') in self.phage_id_map:
                        neg_p_for_h.append(self.phage_id_map[record['target_node']])
                        neg_p_feats.append(record)
            while len(neg_p_for_h) < K:
                neg_p_for_h.append(0); neg_p_feats.append({})

            neg_phages_list.append(neg_p_for_h[:K])
            neg_phage_pair_features.append(neg_p_feats[:K])

        pos_hosts = torch.tensor(pos_hosts, dtype=torch.long, device=self.device)
        pos_phages = torch.tensor(pos_phages, dtype=torch.long, device=self.device)
        neg_hosts = torch.tensor(neg_hosts_list, dtype=torch.long, device=self.device)
        neg_phages = torch.tensor(neg_phages_list, dtype=torch.long, device=self.device)

        return pos_hosts, pos_phages, neg_hosts, neg_phages, pos_features, neg_host_pair_features, neg_phage_pair_features


    def sample_bce(self, batch_size: int, neg_ratio: int = 1):
        pos_sample_indices = np.random.choice(len(self.pos_edges), batch_size, replace=False)
        pos_sample = self.pos_edges[pos_sample_indices]
        pos_hosts, pos_phages = pos_sample[:, 0], pos_sample[:, 1]
        pos_features = [self.pos_features_map.get(tuple(edge), {}) for edge in pos_sample]
        n_neg = int(batch_size * neg_ratio)
        neg_sample_indices = np.random.choice(self.neg_indices, n_neg, replace=False)
        neg_hosts, neg_phages, neg_features = [], [], []
        for i in neg_sample_indices:
            record = self.neg_records[i]
            if record.get('source_node') in self.host_id_map and record.get('target_node') in self.phage_id_map:
                neg_hosts.append(self.host_id_map[record['source_node']])
                neg_phages.append(self.phage_id_map[record['target_node']])
                neg_features.append(record)
        while len(neg_hosts) < n_neg:
            if neg_hosts:
                neg_hosts.append(neg_hosts[-1]); neg_phages.append(neg_phages[-1]); neg_features.append(neg_features[-1])
            else: break 
        host_indices = np.concatenate([pos_hosts, np.array(neg_hosts)])
        phage_indices = np.concatenate([pos_phages, np.array(neg_phages)])
        labels = np.concatenate([np.ones(len(pos_hosts)), np.zeros(len(neg_hosts))])
        pair_features = pos_features + neg_features
        shuffle_idx = np.random.permutation(len(host_indices))
        host_indices = torch.tensor(host_indices[shuffle_idx], dtype=torch.long, device=self.device)
        phage_indices = torch.tensor(phage_indices[shuffle_idx], dtype=torch.long, device=self.device)
        labels = torch.tensor(labels[shuffle_idx], dtype=torch.float, device=self.device)
        pair_features = [pair_features[i] for i in shuffle_idx]
        return host_indices, phage_indices, labels, pair_features

