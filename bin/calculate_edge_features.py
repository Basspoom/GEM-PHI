import argparse
import sys
import yaml
from pathlib import Path
from colorama import Fore, Style, init

init(autoreset=True)

class CustomFormatter(argparse.HelpFormatter):
    def _format_action_invocation(self, action):
        if not action.option_strings or action.nargs == 0:
            return super()._format_action_invocation(action)
        return ' '.join(action.option_strings) + ' ' + self._get_default_metavar_for_optional(action)

    def _get_help_string(self, action):
        help_string = super()._get_help_string(action)
        if action.dest in ['config_path', 'output_dir', 'num_workers']:
            return f"{Fore.GREEN}{help_string}{Style.RESET_ALL}"
        elif action.dest in ['device']:
            return f"{Fore.YELLOW}{help_string}{Style.RESET_ALL}"
        return help_string

def device_type(value):
    if value.lower() == 'cpu': return 'cpu'
    if value.lower().startswith('cuda:'): value = value[5:]
    if not value: raise argparse.ArgumentTypeError("If specifying 'cuda', you must provide a device ID. e.g., 'cuda:0' or '0'.")
    try:
        device_ids = [int(x) for x in value.split(',')]
        return [f'cuda:{i}' for i in device_ids]
    except (ValueError, IndexError):
        raise argparse.ArgumentTypeError("Invalid device format. Use 'cpu', a single GPU ID (e.g., '0'), or a comma-separated list of IDs (e.g., '0,1,2').")

# Only import heavy libraries when not just requesting help
if '--help' not in sys.argv and '-h' not in sys.argv:
    import os, itertools, pandas as pd, subprocess
    from multiprocessing import Pool
    from tqdm import tqdm
    from Bio import SeqIO, AlignIO
    from Bio.Phylo.TreeConstruction import DistanceCalculator
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import shutil
    from intervaltree import IntervalTree
    import numpy as np, traceback
    import torch, random

    # -- Helper Functions (Unified and Robust) --

    def prepare_embeddings(embedding_series: pd.Series, device: str):
        try:
            first_valid_embedding = embedding_series.dropna().iloc[0]
        except IndexError:
            print(f"{Fore.YELLOW}[Warning] No valid embeddings found, returning empty tensor.{Style.RESET_ALL}")
            return torch.tensor([]).to(device)
        embedding_dim = len(first_valid_embedding)
        zero_vector = np.zeros(embedding_dim, dtype=np.float32)
        processed_embeddings = [np.array(emb, dtype=np.float32) if isinstance(emb, np.ndarray) and len(emb) == embedding_dim else zero_vector for emb in tqdm(embedding_series, desc=f"{Fore.YELLOW}Preparing embeddings{Style.RESET_ALL}")]
        return torch.tensor(np.stack(processed_embeddings), dtype=torch.float32).to(device)

    def _extract_16s_from_host(args):
        node_id, genome_path, lineage, temp_dir = args
        try:
            out_fasta, tmp_bed, tmp_raw_fasta = temp_dir / f"{node_id}_16S.fasta", temp_dir / f"{node_id}_16S.bed", temp_dir / f"{node_id}_16S_raw.fasta"
            if not Path(genome_path).exists(): return node_id, 'fail'
            cmd = f"barrnap --kingdom bac --threads 1 {genome_path} --quiet | awk '$3 == \"rRNA\" && $0 ~ /16S/' > {tmp_bed}"
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if not tmp_bed.exists() or tmp_bed.stat().st_size == 0:
                if tmp_bed.exists(): tmp_bed.unlink()
                return node_id, 'fail'
            getfa_cmd = f"bedtools getfasta -fi {genome_path} -bed {tmp_bed} -fo {tmp_raw_fasta}"
            subprocess.run(getfa_cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            tmp_bed.unlink()
            seqs = list(SeqIO.parse(tmp_raw_fasta, "fasta")); tmp_raw_fasta.unlink()
            if not seqs: return node_id, 'fail'
            longest_seq = max(seqs, key=lambda s: len(s.seq)); longest_seq.id, longest_seq.description = f"{node_id}", ""
            SeqIO.write(longest_seq, out_fasta, "fasta")
            return node_id, 'success'
        except Exception as e:
            print(f"{Fore.RED}Error extracting 16S for {node_id}: {e}{Style.RESET_ALL}"); traceback.print_exc(); return node_id, 'fail'

    def _calculate_d2star_sim(vectors):
        vec1, vec2 = vectors
        if not isinstance(vec1, (np.ndarray, list)) or not isinstance(vec2, (np.ndarray, list)): return None
        vec1, vec2 = np.array(vec1, dtype=np.float32), np.array(vec2, dtype=np.float32)
        vec1_centered, vec2_centered = vec1 - np.mean(vec1), vec2 - np.mean(vec2)
        norm1, norm2 = np.linalg.norm(vec1_centered), np.linalg.norm(vec2_centered)
        if norm1 == 0 or norm2 == 0: return 0.0
        return np.dot(vec1_centered, vec2_centered) / (norm1 * norm2)

    def _create_blast_db(fasta_path: Path, db_prefix: Path):
        try:
            db_prefix.parent.mkdir(exist_ok=True, parents=True)
            if not list(db_prefix.parent.glob('*.n*')):
                cmd = f"makeblastdb -in {fasta_path} -dbtype nucl -out {db_prefix} -title {db_prefix.name}"
                subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # print(f"{Fore.GREEN}Created BLAST DB at {db_prefix}{Style.RESET_ALL}")
            return True
        except (subprocess.CalledProcessError, Exception) as e:
            print(f"{Fore.RED}Error creating BLAST DB for {fasta_path}: {e}{Style.RESET_ALL}"); return False

    def _create_window_fasta(fasta_path: Path, win_fasta_path: Path, window_size: int, step_size: int):
        try:
            win_fasta_path.parent.mkdir(exist_ok=True, parents=True)
            if not win_fasta_path.exists():
                cmd = f"seqkit sliding -s {step_size} -W {window_size} -g {fasta_path} -o {win_fasta_path}"
                subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # print(f"{Fore.GREEN}Created window FASTA at {win_fasta_path}{Style.RESET_ALL}")
            return True
        except (subprocess.CalledProcessError, Exception) as e:
            print(f"{Fore.RED}Error creating window FASTA for {fasta_path}: {e}{Style.RESET_ALL}"); return False

    def get_seq_length(fasta_path):
        try: return sum(len(record.seq) for record in SeqIO.parse(fasta_path, "fasta"))
        except FileNotFoundError: return 0

    def _compute_interval_coverage(intervals, seq_len):
        if not intervals: return 0.0, 0
        tree = IntervalTree.from_tuples(intervals); tree.merge_overlaps()
        total_coverage = sum(iv.end - iv.begin for iv in tree)
        return total_coverage / seq_len if seq_len > 0 else 0.0, len(tree)

    def _summarize_homology(blast_df, phage_len, host_len):
        if blast_df.empty: return [0, 0, 0, 0.0, 0.0, 0, 0.0]
        total_bp = blast_df["length"].sum(); max_len = blast_df["length"].max()
        hits = blast_df.shape[0]
        avg_identity = round((blast_df["pident"] * blast_df["length"]).sum() / total_bp, 2) if total_bp > 0 else 0.0
        phage_intervals = [(min(r.qstart, r.qend), max(r.qstart, r.qend)) for r in blast_df.itertuples()]
        host_intervals = [(min(r.sstart, r.send), max(r.sstart, r.send)) for r in blast_df.itertuples()]
        phage_cov_ratio, region_count = _compute_interval_coverage(phage_intervals, phage_len)
        host_cov_ratio, _ = _compute_interval_coverage(host_intervals, host_len)
        return [max_len, total_bp, hits, round(phage_cov_ratio, 4), round(host_cov_ratio, 4), region_count, avg_identity]

    def _run_blast_and_summarize(args):
        phage_id, host_id, genome_paths, cache_dirs, params = args
        default_metrics = [0, 0, 0, 0.0, 0.0, 0, 0.0]
        try:
            phage_fasta, host_fasta = genome_paths.get(phage_id), genome_paths.get(host_id)
            if not (isinstance(phage_fasta, (str, Path)) and isinstance(host_fasta, (str, Path))): return (phage_id, host_id, *default_metrics)
            phage_win_path = cache_dirs['win_fasta'] / f"{phage_id}_win.fasta"
            host_db_prefix = cache_dirs['blast_db'] / host_id / host_id
            if not phage_win_path.exists() or not (host_db_prefix.parent / f"{host_id}.nhr").exists(): return (phage_id, host_id, *default_metrics)
            blast_out_path = cache_dirs['blast_out'] / f"{phage_id}-{host_id}.tsv"
            if not blast_out_path.exists() or blast_out_path.stat().st_size == 0:
                blast_cmd = (f"blastn -query {phage_win_path} -db {host_db_prefix} -out {blast_out_path} "
                             f"-evalue {params['evalue']} -perc_identity {params['identity']} "
                             f"-outfmt '6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore' "
                             f"-num_threads 1 -dust no -soft_masking false -word_size 7 -max_target_seqs 1000")
                subprocess.run(blast_cmd, shell=True, check=True, capture_output=True)
            cols = ["qseqid", "sseqid", "pident", "length", "mismatch", "gapopen", "qstart", "qend", "sstart", "send", "evalue", "bitscore"]
            try:
                blast_df = pd.read_csv(blast_out_path, sep="\t", names=cols)
                blast_df = blast_df[(blast_df["length"] >= params['min_len']) & (blast_df["pident"] >= params['identity'])]
            except pd.errors.EmptyDataError: blast_df = pd.DataFrame(columns=cols)
            phage_len, host_len = get_seq_length(phage_fasta), get_seq_length(host_fasta)
            metrics = _summarize_homology(blast_df, phage_len, host_len)
            return (phage_id, host_id, *metrics)
        except Exception:
            print(f"{Fore.RED}An error occurred while processing pair ({phage_id}, {host_id}):{Style.RESET_ALL}"); traceback.print_exc()
            return (phage_id, host_id, *default_metrics)

    def _run_single_ccf_cmd(cmd_tuple):
        cmd, working_dir = cmd_tuple
        try:
            subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, cwd=working_dir, timeout=600)
            return "success"
        except subprocess.TimeoutExpired: return "timeout"
        # except subprocess.CalledProcessError: print(f"{Fore.RED}--- CRISPRCasFinder returned an error (output silenced) ---\nCMD: {cmd}{Style.RESET_ALL}"); return "error"
        except subprocess.CalledProcessError: return "error"

    def _aggregate_single_host(host_id, ccf_out_dir, spacer_results_dir):
        try:
            modified_host_id = host_id.split('.')[0]
            spacer_source_dir = ccf_out_dir / host_id / modified_host_id
            if not spacer_source_dir.is_dir(): return host_id, "NO_OUTPUT_DIR"
            all_records, spacer_counter = [], 1
            spacer_files = sorted(list(spacer_source_dir.glob("Spacers_*")))
            if not spacer_files: return host_id, "NO_SPACERS_FOUND"
            for fpath in spacer_files:
                for r in SeqIO.parse(fpath, "fasta"):
                    r.id, r.description = f"{host_id}_spacer{spacer_counter}", ""
                    all_records.append(r); spacer_counter += 1
            if all_records:
                final_out_fasta = spacer_results_dir / f"{host_id}_spacer.fasta"
                SeqIO.write(all_records, final_out_fasta, "fasta")
                return host_id, "SUCCESS"
            else: return host_id, "NO_SPACERS_FOUND"
        except Exception as e:
            print(f"{Fore.RED}Error aggregating spacers for {host_id}: {e}{Style.RESET_ALL}"); traceback.print_exc(); return host_id, "AGGREGATION_ERROR"
            
    def _process_blast_pair(phage_id, host_id, spacer_dir, blast_db_dir, blast_results_dir):
        base_result = {"source_node": phage_id, "target_node": host_id, "crispr_spacer_hits": 0, "crispr_unique_spacers": 0, "crispr_has_hit": False, "crispr_mean_bitscore": None, "crispr_min_evalue": None,}
        try:
            spacer_fasta = spacer_dir / f"{host_id}_spacer.fasta"
            db_prefix = blast_db_dir / phage_id / phage_id
            if not spacer_fasta.exists() or not list(db_prefix.parent.glob('*.n*')): return base_result
            out_raw, out_final = blast_results_dir / f"{phage_id}-{host_id}.raw.tsv", blast_results_dir / f"{phage_id}-{host_id}.final.tsv"
            blast_cmd = (f'blastn -query {spacer_fasta} -db {db_prefix} -outfmt "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore" '
                         f'-num_threads 1 -dust no -soft_masking false -word_size 7 -max_target_seqs 1 -out {out_raw}')
            subprocess.run(blast_cmd, shell=True, check=True, capture_output=True)
            filter_cmd = f"awk '$3 >= 98.0 && $4 >= 15' {out_raw} > {out_final}"; subprocess.run(filter_cmd, shell=True, check=True)
            out_raw.unlink()
            if not out_final.exists() or out_final.stat().st_size == 0: return base_result
            cols = ["qseqid", "sseqid", "pident", "length", "mismatch", "gapopen", "qstart", "qend", "sstart", "send", "evalue", "bitscore"]
            df = pd.read_csv(out_final, sep="\t", names=cols);
            if df.empty: return base_result
            return {"source_node": phage_id, "target_node": host_id, "crispr_spacer_hits": len(df), "crispr_unique_spacers": df["qseqid"].nunique(), "crispr_has_hit": True,
                    "crispr_mean_bitscore": df["bitscore"].mean(), "crispr_min_evalue": df["evalue"].min(),}
        except Exception:
            print(f"{Fore.RED}Error processing BLAST for pair ({phage_id}, {host_id}):{Style.RESET_ALL}"); traceback.print_exc(); return base_result

    # -- Main Functions --

    def calculate_pp_edges(new_nodes_pkl_path, ref_nodes_pkl_path, output_dir, device='cpu', batch_size=1024):
        print(f"\n{Fore.CYAN}---- 03. Calculating edge features ... ----{Style.RESET_ALL}")
        print(f"\n{Fore.CYAN}---- A. Calculating Phage-Phage (PP) similarity edges... ----{Style.RESET_ALL}")
        output_path = Path(output_dir); output_path.mkdir(parents=True, exist_ok=True)
        new_nodes_df, ref_nodes_df = pd.read_pickle(new_nodes_pkl_path), pd.read_pickle(ref_nodes_pkl_path)
        new_phages_df = new_nodes_df[new_nodes_df['type'] == 'phage'].copy().rename(columns={'NT_embedding': 'NT_30random'})
        all_phages_df = pd.concat([new_phages_df, ref_nodes_df[ref_nodes_df['type'] == 'phage']], ignore_index=True).drop_duplicates(subset=['Node_id']).reset_index(drop=True)
        if new_phages_df.empty: print(f"{Fore.YELLOW}No new phages found. Skipping PP edge calculation.{Style.RESET_ALL}"); return
        print(f"Comparing {len(new_phages_df)} new phages against a total of {len(all_phages_df)} phages.")
        all_nt_embeddings, all_rbp_embeddings = prepare_embeddings(all_phages_df['NT_30random'], device), prepare_embeddings(all_phages_df['RBP_embedding'], device)
        new_nt_embeddings, new_rbp_embeddings = prepare_embeddings(new_phages_df['NT_30random'], device), prepare_embeddings(new_phages_df['RBP_embedding'], device)
        all_nt_embeddings, all_rbp_embeddings = torch.nn.functional.normalize(all_nt_embeddings, p=2, dim=1), torch.nn.functional.normalize(all_rbp_embeddings, p=2, dim=1)
        new_nt_embeddings, new_rbp_embeddings = torch.nn.functional.normalize(new_nt_embeddings, p=2, dim=1), torch.nn.functional.normalize(new_rbp_embeddings, p=2, dim=1)
        edge_list, all_phage_ids = [], all_phages_df['Node_id'].values
        for i in tqdm(range(0, len(new_phages_df), batch_size), desc=f"{Fore.YELLOW}Finding top 5 PP edges (in batches){Style.RESET_ALL}"):
            batch_end = min(i + batch_size, len(new_phages_df))
            batch_new_nt, batch_new_rbp = new_nt_embeddings[i:batch_end], new_rbp_embeddings[i:batch_end]
            batch_source_nodes = new_phages_df.iloc[i:batch_end]
            nt_sim_matrix, rbp_sim_matrix = torch.matmul(batch_new_nt, all_nt_embeddings.T), torch.matmul(batch_new_rbp, all_rbp_embeddings.T)
            avg_sim_matrix = (nt_sim_matrix + rbp_sim_matrix) / 2.0
            for j, source_node_id in enumerate(batch_source_nodes['Node_id']):
                self_idx = np.where(all_phage_ids == source_node_id)[0]
                if self_idx.size > 0: avg_sim_matrix[j, self_idx[0]] = -1.0
            top_k_scores, top_k_indices = torch.topk(avg_sim_matrix, k=5, dim=1)
            for j in range(len(batch_source_nodes)):
                source_node_id = batch_source_nodes['Node_id'].iloc[j]
                for k in range(5):
                    target_idx = top_k_indices[j, k].item(); target_node_id = all_phage_ids[target_idx]
                    edge_list.append({'source_node': source_node_id, 'target_node': target_node_id, 'edge_type': 'phage-phage', 'NT_sim': nt_sim_matrix[j, target_idx].item(), 'RBP_embedding_sim': rbp_sim_matrix[j, target_idx].item()})
        pp_edges_df = pd.DataFrame(edge_list)
        output_file = output_path / "PP_top5.pkl"; pp_edges_df.to_pickle(output_file)
        print(f"\n{Fore.GREEN}---- Done! Phage-Phage edge file saved to: {output_file} ----{Style.RESET_ALL}")
        if device.startswith('cuda'): del all_nt_embeddings, all_rbp_embeddings, new_nt_embeddings, new_rbp_embeddings; torch.cuda.empty_cache(); print(f"{Fore.GREEN}Cleared GPU cache after PP edge calculation.{Style.RESET_ALL}")

    def calculate_hh_edges(new_nodes_pkl_path, ref_nodes_pkl_path, ref_16s_fasta_path, output_dir, device='cpu', num_workers=4):
        print(f"\n{Fore.CYAN}---- B. Calculating Host-Host (HH) similarity edges... ----{Style.RESET_ALL}")
        output_path = Path(output_dir); phylo_temp_dir = output_path / "phylo_temp"; phylo_temp_dir.mkdir(exist_ok=True)
        phylo_sim_cache_path = output_path / "cached_phylo_sim.pkl"
        new_nodes_df, ref_nodes_df = pd.read_pickle(new_nodes_pkl_path), pd.read_pickle(ref_nodes_pkl_path)
        new_hosts_df = new_nodes_df[new_nodes_df['type'] == 'host'].copy(); ref_hosts_df = ref_nodes_df[ref_nodes_df['type'] == 'host'].copy()
        if new_hosts_df.empty: print(f"{Fore.YELLOW}No new hosts found. Skipping HH edge calculation.{Style.RESET_ALL}"); return
        all_hosts_for_phylo = pd.concat([new_hosts_df, ref_hosts_df], ignore_index=True).drop_duplicates(subset=['Node_id'])
        current_host_ids = sorted(all_hosts_for_phylo['Node_id'].unique()); phylo_sim_df = None
        if phylo_sim_cache_path.exists():
            print(f"{Fore.GREEN}Found cached phylogenetic similarity matrix. Checking for consistency...{Style.RESET_ALL}")
            cached_phylo_df = pd.read_pickle(phylo_sim_cache_path)
            cached_host_ids = sorted(cached_phylo_df.index.str.replace("_16S", "").unique())
            if cached_host_ids == current_host_ids:
                print(f"{Fore.GREEN}Cache is consistent. Skipping phylogenetic calculation.{Style.RESET_ALL}"); phylo_sim_df = cached_phylo_df
            else: print(f"{Fore.YELLOW}Host list has changed. Recalculating phylogenetic similarity.{Style.RESET_ALL}")
        if phylo_sim_df is None:
            print(f"{Fore.MAGENTA}--- a: Calculating phylogenetic similarity ---{Style.RESET_ALL}")
            new_16s_dir = phylo_temp_dir / "new_16s"; new_16s_dir.mkdir(exist_ok=True)
            tasks = [(row['Node_id'], row['genome_path'], row.get('lineage', ''), new_16s_dir) for _, row in new_hosts_df.iterrows()]
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                list(tqdm(executor.map(_extract_16s_from_host, tasks), total=len(tasks), desc=f"{Fore.YELLOW}Running barrnap{Style.RESET_ALL}"))
            combined_16s_path = phylo_temp_dir / "all_hosts_16S_combined.fasta"; os.system(f"cat {ref_16s_fasta_path} {new_16s_dir}/*.fasta > {combined_16s_path} 2>/dev/null")
            aligned_fasta_path = phylo_temp_dir / "16S_aligned.fasta"; print(f"{Fore.YELLOW}Running MAFFT for multiple sequence alignment...{Style.RESET_ALL}")
            os.system(f"mafft --thread {num_workers} --auto {combined_16s_path} > {aligned_fasta_path} 2>/dev/null")
            if not aligned_fasta_path.exists() or aligned_fasta_path.stat().st_size == 0:
                print(f"{Fore.RED}[ERROR] MAFFT alignment failed. Skipping HH edge calculation.{Style.RESET_ALL}"); shutil.rmtree(phylo_temp_dir); return
            print(f"{Fore.YELLOW}Building phylogenetic tree and distance matrix...{Style.RESET_ALL}")
            # alignment = SeqIO.parse(aligned_fasta_path, "fasta"); calculator = DistanceCalculator("identity"); dm = calculator.get_distance(list(alignment))
            alignment = AlignIO.read(aligned_fasta_path, "fasta"); calculator = DistanceCalculator("identity"); dm = calculator.get_distance(alignment)
            dist_df = pd.DataFrame(list(dm), index=dm.names, columns=dm.names)
            dist_matrix = dist_df.values.astype(float); max_dist = dist_matrix.max()
            if max_dist == 0: max_dist = 1
            phylo_sim_matrix = 1 - (dist_matrix / max_dist)
            phylo_sim_df = pd.DataFrame(phylo_sim_matrix, index=dm.names, columns=dm.names)
            phylo_sim_df.to_pickle(phylo_sim_cache_path); print(f"{Fore.GREEN}Saved new phylogenetic similarity matrix to cache: {phylo_sim_cache_path}{Style.RESET_ALL}")
        phylo_sim_df.index = phylo_sim_df.index.str.replace("_16S", ""); phylo_sim_df.columns = phylo_sim_df.columns.str.replace("_16S", "")
        print(f"\n{Fore.MAGENTA}--- b: Calculating NT embedding similarity ---{Style.RESET_ALL}")
        new_hosts_df.rename(columns={'NT_embedding': 'NT_30random'}, inplace=True)
        all_hosts_df = pd.concat([new_hosts_df, ref_hosts_df], ignore_index=True).drop_duplicates(subset=['Node_id']).reset_index(drop=True)
        all_nt_embeddings = prepare_embeddings(all_hosts_df['NT_30random'], device); all_nt_embeddings = torch.nn.functional.normalize(all_nt_embeddings, p=2, dim=1)
        nt_sim_matrix = torch.matmul(all_nt_embeddings, all_nt_embeddings.T).cpu().numpy(); nt_sim_df = pd.DataFrame(nt_sim_matrix, index=all_hosts_df['Node_id'], columns=all_hosts_df['Node_id'])
        print(f"\n{Fore.MAGENTA}--- c: Combining similarities and finding top 10 edges ---{Style.RESET_ALL}")
        edge_list = []; common_hosts = phylo_sim_df.index.intersection(nt_sim_df.index)
        phylo_sim_df, nt_sim_df = phylo_sim_df.loc[common_hosts, common_hosts], nt_sim_df.loc[common_hosts, common_hosts]
        avg_sim_df = (phylo_sim_df + nt_sim_df) / 2.0; ref_host_ids = ref_hosts_df['Node_id'].unique()
        for source_node_id in tqdm(new_hosts_df['Node_id'], desc=f"{Fore.YELLOW}Finding top 10 HH edges{Style.RESET_ALL}"):
            if source_node_id not in avg_sim_df.index: continue
            all_scores = avg_sim_df.loc[source_node_id]
            valid_ref_host_ids = [hid for hid in ref_host_ids if hid in all_scores.index]
            top_10 = all_scores[valid_ref_host_ids].nlargest(10)
            for target_node_id, avg_sim in top_10.items():
                edge_list.append({'source_node': source_node_id, 'target_node': target_node_id, 'edge_type': 'host-host', 'phylo_sim': phylo_sim_df.loc[source_node_id, target_node_id], 'NT_sim': nt_sim_df.loc[source_node_id, target_node_id]})
        hh_edges_df = pd.DataFrame(edge_list)
        output_file = output_path / "HH_top10.pkl"; hh_edges_df.to_pickle(output_file)
        print(f"\n{Fore.GREEN}---- Done! Host-Host edge file saved to: {output_file} ----{Style.RESET_ALL}"); shutil.rmtree(phylo_temp_dir); print(f"{Fore.GREEN}Cleaned up temporary directory.{Style.RESET_ALL}")

    def calculate_ph_edges(new_nodes_pkl_path, output_dir, num_workers=4):
        print(f"\n{Fore.CYAN}---- C. Calculating Phage-Host (PH) similarity edges (New Phages vs New Hosts ONLY)... ----{Style.RESET_ALL}")
        output_path = Path(output_dir); output_file = output_path / "PH_edges_information.pkl"
        new_nodes_df = pd.read_pickle(new_nodes_pkl_path)
        new_phages_df, new_hosts_df = new_nodes_df[new_nodes_df['type'] == 'phage'].copy(), new_nodes_df[new_nodes_df['type'] == 'host'].copy()
        if new_phages_df.empty or new_hosts_df.empty:
            print(f"{Fore.YELLOW}Either new phages or new hosts are missing. Skipping PH edge calculation.{Style.RESET_ALL}"); pd.DataFrame(columns=['source_node', 'target_node', 'edge_type']).to_pickle(output_file); return
        pairs = list(itertools.product(new_phages_df['Node_id'], new_hosts_df['Node_id'])); ph_edges_df = pd.DataFrame(pairs, columns=['source_node', 'target_node'])
        print(f"Created {len(ph_edges_df)} pairs between {len(new_phages_df)} new phages and {len(new_hosts_df)} new hosts.")
        print(f"\n{Fore.MAGENTA}--- a: Calculating 6-mer d2* similarity ---{Style.RESET_ALL}")
        merged_df = pd.merge(ph_edges_df, new_phages_df[['Node_id', '6mer_vector']], left_on='source_node', right_on='Node_id'); merged_df = pd.merge(merged_df, new_hosts_df[['Node_id', '6mer_vector']], left_on='target_node', right_on='Node_id', suffixes=('_phage', '_host'))
        merged_df.dropna(subset=['6mer_vector_phage', '6mer_vector_host'], inplace=True); d2star_sims = []
        if not merged_df.empty:
            vector_pairs = list(zip(merged_df['6mer_vector_phage'], merged_df['6mer_vector_host']))
            with Pool(processes=num_workers) as pool:
                d2star_sims = list(tqdm(pool.imap(_calculate_d2star_sim, vector_pairs), total=len(vector_pairs), desc=f"{Fore.YELLOW}Calculating 6-mer d2*{Style.RESET_ALL}"))
        sim_df = pd.DataFrame({'source_node': merged_df['source_node'], 'target_node': merged_df['target_node'], '6-mer_d2*_sim': d2star_sims}); ph_edges_df = pd.merge(ph_edges_df, sim_df, on=['source_node', 'target_node'], how='left')
        print(f"{Fore.GREEN}Finished 6-mer similarity calculation.{Style.RESET_ALL}")
        print(f"\n{Fore.MAGENTA}--- b: Calculating BLAST homology features ---{Style.RESET_ALL}")
        blast_cache_dir = output_path / "blast_cache"; cache_dirs = {'win_fasta': blast_cache_dir / "win_fasta", 'blast_db': blast_cache_dir / "blast_db", 'blast_out': blast_cache_dir / "blast_out"}
        for d in cache_dirs.values(): d.mkdir(parents=True, exist_ok=True)
        genome_paths = new_nodes_df.set_index('Node_id')['genome_path'].to_dict()
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            db_tasks = [executor.submit(_create_blast_db, Path(row['genome_path']), cache_dirs['blast_db'] / row['Node_id'] / row['Node_id']) for _, row in new_hosts_df.iterrows() if isinstance(row.get('genome_path'), str)]
            list(tqdm(as_completed(db_tasks), total=len(db_tasks), desc=f"{Fore.YELLOW}Building new host BLAST DBs{Style.RESET_ALL}"))
            win_tasks = [executor.submit(_create_window_fasta, Path(row['genome_path']), cache_dirs['win_fasta'] / f"{row['Node_id']}_win.fasta", 500, 250) for _, row in new_phages_df.iterrows() if isinstance(row.get('genome_path'), str)]
            list(tqdm(as_completed(win_tasks), total=len(win_tasks), desc=f"{Fore.YELLOW}Creating new phage window FASTAs{Style.RESET_ALL}"))
        blast_params = {'evalue': 1, 'identity': 90, 'min_len': 15}; blast_tasks = [(row['source_node'], row['target_node'], genome_paths, cache_dirs, blast_params) for _, row in ph_edges_df.iterrows()]
        blast_results = []
        if blast_tasks:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(_run_blast_and_summarize, task) for task in blast_tasks]
                for future in tqdm(as_completed(futures), total=len(blast_tasks), desc=f"{Fore.YELLOW}Running BLAST for PH pairs{Style.RESET_ALL}"): blast_results.append(future.result())
        if blast_results:
            homology_cols = ['homology_max_length', 'homology_total_bp', 'homology_hits', 'phage_covered_ratio', 'host_covered_ratio', 'matched_region_count', 'avg_match_identity']
            homology_df = pd.DataFrame(blast_results, columns=['source_node', 'target_node'] + homology_cols); ph_edges_df = pd.merge(ph_edges_df, homology_df, on=['source_node', 'target_node'], how='left')
        ph_edges_df['edge_type'] = 'phage-host'; final_cols = ['source_node', 'target_node', 'edge_type', '6-mer_d2*_sim'] + homology_cols
        for col in final_cols:
            if col not in ph_edges_df.columns: ph_edges_df[col] = np.nan
        ph_edges_df[final_cols].to_pickle(output_file) # ; print(f"\n{Fore.GREEN}---- Done! New Phage-Host edge file saved to: {output_file} ----{Style.RESET_ALL}")
        # print(f"{Fore.GREEN}Your BLAST result cache directory has been preserved.{Style.RESET_ALL}")

    def _create_blast_db_crispr(phage_id, genome_path, db_root_dir):
        try:
            db_dir = db_root_dir / phage_id; db_dir.mkdir(exist_ok=True); db_prefix = db_dir / phage_id
            if not list(db_dir.glob('*.n*')):
                cmd = f"makeblastdb -in {genome_path} -dbtype nucl -out {db_prefix} -title {phage_id}"
                subprocess.run(cmd, shell=True, check=True, capture_output=True)
                # print(f"{Fore.GREEN}Created CRISPR BLAST DB for {phage_id}.{Style.RESET_ALL}")
        except Exception:
            print(f"{Fore.RED}Failed to create CRISPR BLAST DB for {phage_id}:{Style.RESET_ALL}"); traceback.print_exc()

    def calculate_all_edge_features(
        output_dir: str, ref_nodes_file: str, ref_16s_file: str, ccf_script: str, ccf_so: str, num_workers: int = 4, device: str = 'cpu'
    ):
        output_path = Path(output_dir); cache_dir = output_path / "03_process_edges"; cache_dir.mkdir(parents=True, exist_ok=True)
        ph_edges_pkl, nodes_pkl, pp_edges_pkl, hh_edges_pkl = output_path/"PH_edges_information.pkl", output_path/"node_information.pkl", output_path/"PP_top5.pkl", output_path/"HH_top10.pkl"
        if not Path(nodes_pkl).exists(): print(f"{Fore.RED}[ERROR] Required input file '{nodes_pkl}' not found. Please run calculate_node_feature.py first.{Style.RESET_ALL}"); return
        
        calculate_pp_edges(nodes_pkl, ref_nodes_file, cache_dir, device, 1024)
        shutil.move(cache_dir / "PP_top5.pkl", pp_edges_pkl) # ; print(f"{Fore.GREEN}Moved PP_top5.pkl to {pp_edges_pkl}{Style.RESET_ALL}")
        
        if not Path(ref_16s_file).exists(): print(f"{Fore.RED}[ERROR] Reference 16S FASTA file not found at '{ref_16s_file}'. Skipping HH edge calculation.{Style.RESET_ALL}")
        else:
            calculate_hh_edges(nodes_pkl, ref_nodes_file, ref_16s_file, cache_dir, device, num_workers)
            shutil.move(cache_dir / "HH_top10.pkl", hh_edges_pkl); print(f"{Fore.GREEN}Moved HH_top10.pkl to {hh_edges_pkl}{Style.RESET_ALL}")
            
        calculate_ph_edges(nodes_pkl, cache_dir, num_workers)
        shutil.move(cache_dir / "PH_edges_information.pkl", ph_edges_pkl); print(f"{Fore.GREEN}Moved PH_edges_information.pkl to {ph_edges_pkl}{Style.RESET_ALL}")
        
        print(f"\n{Fore.MAGENTA}--- c: Calculating CRISPR features ---{Style.RESET_ALL}")
        if 'crispr_has_hit' in pd.read_pickle(ph_edges_pkl).columns: print("✅ CRISPR features already found. Nothing to do."); return
        
        ccf_out_dir, ccf_tmp_dir = cache_dir / "CRISPRCasFinder/predictions", cache_dir / "CRISPRCasFinder/tmp"
        spacer_results_dir, phage_db_dir, blast_results_dir = cache_dir / "CRISPRCasFinder/aggregated_spacers", cache_dir / "blast_cache_crispr/phage_db", cache_dir / "blast_cache_crispr/blast_results"
        for d in [ccf_tmp_dir, ccf_out_dir, spacer_results_dir, phage_db_dir, blast_results_dir]: d.mkdir(parents=True, exist_ok=True)
        
        ph_edges_df, nodes_df = pd.read_pickle(ph_edges_pkl), pd.read_pickle(nodes_pkl)
        genome_base_dir = Path(output_dir) / "01_process_genome"
        involved_host_ids = ph_edges_df['target_node'].unique()
        ccf_tasks = [(f"perl {ccf_script} -def G -i '{genome_base_dir}/host/{host_id}.fasta' -out '{(ccf_out_dir/host_id).resolve()}' -keep -force -so {ccf_so}", ccf_tmp_dir) for host_id in involved_host_ids if (genome_base_dir / "host" / f"{host_id}.fasta").exists()]
        if ccf_tasks:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                list(tqdm(executor.map(_run_single_ccf_cmd, ccf_tasks), total=len(ccf_tasks), desc=f"{Fore.YELLOW}Predicting CRISPR arrays{Style.RESET_ALL}"))
        
        agg_tasks = [(host_id, ccf_out_dir, spacer_results_dir) for host_id in involved_host_ids]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(_aggregate_single_host, *zip(*agg_tasks)), total=len(agg_tasks), desc=f"{Fore.YELLOW}Aggregating spacers{Style.RESET_ALL}"))
        
        involved_phage_ids = ph_edges_df['source_node'].unique()
        db_tasks = [(phage_id, genome_base_dir / "phage" / f"{phage_id}.fasta", phage_db_dir) for phage_id in involved_phage_ids if (genome_base_dir / "phage" / f"{phage_id}.fasta").exists()]
        if db_tasks:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                list(tqdm(executor.map(_create_blast_db_crispr, *zip(*db_tasks)), total=len(db_tasks), desc=f"{Fore.YELLOW}Building Phage BLAST DBs{Style.RESET_ALL}"))
        
        blast_tasks = [(row['source_node'], row['target_node'], spacer_results_dir, phage_db_dir, blast_results_dir) for _, row in ph_edges_df.iterrows()]
        all_blast_results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_process_blast_pair, *task) for task in blast_tasks]
            for future in tqdm(as_completed(futures), total=len(blast_tasks), desc=f"{Fore.YELLOW}BLASTing spacers vs phages{Style.RESET_ALL}"): all_blast_results.append(future.result())
            
        crispr_df = pd.DataFrame(all_blast_results)
        final_df = pd.merge(ph_edges_df, crispr_df, on=['source_node', 'target_node'], how='left')
        final_df.to_pickle(ph_edges_pkl)
        print(f"\n{Fore.GREEN}✅ Success! All edge calculations are complete!{Style.RESET_ALL}")
        # shutil.rmtree(cache_dir); print(f"{Fore.GREEN}Cleaned up temporary directory: {cache_dir}{Style.RESET_ALL}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=f"{Fore.BLUE}{Style.BRIGHT}Calculate edge features for GEM-PHI.{Style.RESET_ALL}",
        formatter_class=CustomFormatter,
    )
    parser.add_argument('--config_path', type=Path, required=True, help='Path to the config.yml file.')
    parser.add_argument('--output_dir', type=Path, required=True, help='Main output directory.')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of parallel workers.')
    parser.add_argument('--device', type=device_type, default='0', help="Specify device for model inference. Use 'cpu', a single GPU ID (e.g., '0'), or a comma-separated list of IDs (e.g., '0,1,2').")
    args = parser.parse_args()

    if '--help' not in sys.argv and '-h' not in sys.argv:
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        device = 'cpu'
        if args.device != 'cpu':
            if not torch.cuda.is_available():
                print(f"{Fore.RED}CUDA is not available, falling back to CPU.{Style.RESET_ALL}")
            else:
                try:
                    if isinstance(args.device, list):
                        device = args.device[0]
                    else:
                        device = f'cuda:{args.device}'
                except Exception:
                    print(f"{Fore.RED}Invalid device specified: {args.device}. Falling back to CPU.{Style.RESET_ALL}")

        calculate_all_edge_features(
            output_dir=args.output_dir,
            ref_nodes_file=config['paths']['ref_nodes_file'],
            ref_16s_file=config['paths']['ref_16s_file'],
            ccf_script=config['paths']['ccf_script'],
            ccf_so=config['paths']['ccf_so'],
            num_workers=args.num_workers,
            device=device
        )