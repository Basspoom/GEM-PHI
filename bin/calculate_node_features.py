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
        if action.dest in ['config', 'host_fasta', 'phage_fasta', 'output_dir', 'cpu_cores']:
            help_string = f"{Fore.GREEN}{help_string}{Style.RESET_ALL}"
        elif action.dest in ['device']:
            help_string = f"{Fore.YELLOW}{help_string}{Style.RESET_ALL}"
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

if '--help' not in sys.argv and '-h' not in sys.argv:
    import os, itertools, pandas as pd, subprocess
    from multiprocessing import Pool
    from tqdm import tqdm
    from Bio import SeqIO
    from collections import Counter
    import torch, random
    from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification
    import numpy as np, traceback

    def read_fasta(file_path):
        with open(file_path, 'r') as f:
            header, sequence = None, []
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith('>'):
                    if header: yield header, ''.join(sequence)
                    header, sequence = line[1:], []
                else: sequence.append(line)
            if header: yield header, ''.join(sequence)

    def run_prodigal_command(cmd):
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def calculate_kmer_gc_features(genome_path):
        try:
            seq = str(next(SeqIO.parse(genome_path, "fasta")).seq).upper()
        except Exception: return (None, None)
        if not seq: return (None, None)

        gc_content = (seq.count("G") + seq.count("C")) / len(seq) if len(seq) > 0 else 0
        all_6mers = [''.join(p) for p in itertools.product('ACGT', repeat=6)]
        counts = Counter(seq[i:i+6] for i in range(len(seq) - 5))
        total = sum(counts.values())
        kmer_vector = [counts.get(kmer, 0) / total for kmer in all_6mers] if total > 0 else [0.0] * len(all_6mers)
        return (gc_content, kmer_vector)

    def calculate_nt_embedding(args):
        genome_path, model, tokenizer, device, window_size, sample_fraction, seed = args
        try:
            seq = "".join(str(r.seq) for r in SeqIO.parse(genome_path, "fasta")).upper()
        except Exception: return None
        if not seq: return None

        windows = [seq]
        if len(seq) >= window_size:
            windows = []
            random.seed(seed)
            target_len = int(len(seq) * sample_fraction)
            attempts = 0
            while sum(len(w) for w in windows) < target_len and attempts < 50:
                start = random.randint(0, len(seq) - window_size)
                window = seq[start:start+window_size]
                if 'N' not in window: windows.append(window)
                attempts += 1
            if not windows: windows.append(seq[:window_size])

        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(windows), 8):
                batch = windows[i:i+8]
                tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1000).to(device)
                outputs = model(**tokens, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                attention_mask = tokens['attention_mask']
                masked_embs = hidden_states * attention_mask.unsqueeze(-1)
                summed_embs = masked_embs.sum(dim=1)
                counts = attention_mask.sum(dim=1, keepdim=True)
                mean_embs = summed_embs / counts
                all_embeddings.append(mean_embs.cpu())

        if not all_embeddings: return None
        final_embedding = torch.cat(all_embeddings, dim=0).mean(dim=0).numpy()
        return final_embedding

    def predict_and_save_rbps(protein_faa_path, rbp_faa_path, model, tokenizer, device):
        try:
            records = list(SeqIO.parse(protein_faa_path, "fasta"))
            if not records:
                open(rbp_faa_path, 'w').close()
                return
            
            rbp_records = []
            with torch.no_grad():
                for record in records:
                    encoding = tokenizer(str(record.seq), return_tensors="pt", truncation=True).to(device)
                    pred = int(model(**encoding).logits.argmax(dim=-1))
                    if pred == 1: rbp_records.append(record)
            SeqIO.write(rbp_records, rbp_faa_path, 'fasta')
        except Exception:
            open(rbp_faa_path, 'w').close()

    def calculate_rbp_embedding(rbp_faa_path, model, tokenizer, device):
        if not rbp_faa_path.exists() or rbp_faa_path.stat().st_size == 0: return None
        try:
            records = list(SeqIO.parse(rbp_faa_path, "fasta"))
            if not records: return None
            
            embeddings = []
            with torch.no_grad():
                for rec in records:
                    inputs = tokenizer(str(rec.seq), return_tensors="pt")['input_ids'].to(device)
                    embeddings.append(model(inputs).logits.squeeze(0).cpu().numpy())
            
            if not embeddings: return None
            return np.stack(embeddings).mean(axis=0)
        except Exception: return None

def process_genomes_for_phi_prediction(
    host_fasta_path, phage_fasta_path, output_dir, nt_model_path,
    rbp_detect_model_path, rbp_embed_model_path, device='cpu', pairs_csv_path=None, num_workers=4
):
    output_path = Path(output_dir)
    proc_dir = output_path / "01_process_genome"
    nodes_dir = output_path / "02_process_nodes"
    proc_dir.mkdir(parents=True, exist_ok=True)
    nodes_dir.mkdir(parents=True, exist_ok=True)
    host_proc_dir, phage_proc_dir = proc_dir / "host", proc_dir / "phage"
    host_proc_dir.mkdir(exist_ok=True); phage_proc_dir.mkdir(exist_ok=True)
    output_pkl_path = output_path / "node_information.pkl"
    phage_rbp_dir = nodes_dir / "phage_rbps"
    phage_rbp_dir.mkdir(exist_ok=True)

    print(f"\n{Fore.CYAN}---- 01. Preprocess genomes of hosts and phages... ----{Style.RESET_ALL}")
    if pairs_csv_path is None:
        all2all_csv_path = Path(host_fasta_path).parent / "all2all.csv"
        if not all2all_csv_path.exists():
            host_ids = [h.split()[0] for h, s in read_fasta(host_fasta_path)]
            phage_ids = [p.split()[0] for p, s in read_fasta(phage_fasta_path)]
            pd.DataFrame(list(itertools.product(phage_ids, host_ids)), columns=['phage', 'host']).to_csv(all2all_csv_path, index=False)

    node_info_list = []
    def prepare_and_run(fasta_path, proc_dir, genome_type):
        all_sequences = list(read_fasta(fasta_path))
        commands_to_run = []
        existing_ids = {p.stem for p in proc_dir.glob('*.faa')}
        for header, sequence in all_sequences:
            seq_id = header.split()[0]
            nuc_path, prot_path = proc_dir / f"{seq_id}.fasta", proc_dir / f"{seq_id}.faa"
            node_info_list.append({'Node_id': seq_id, 'type': genome_type, 'genome_path': str(nuc_path.resolve()), 'protein_path': str(prot_path.resolve())})
            if seq_id in existing_ids: continue
            with open(nuc_path, 'w') as f: f.write(f">{header}\n{sequence}\n")
            commands_to_run.append(f"prodigal -i {nuc_path} -a {prot_path} -p meta -q")

        print(f"{Fore.GREEN}Found {len(all_sequences)} {genome_type} genomes.{Style.RESET_ALL} Skipping {len(all_sequences) - len(commands_to_run)} (already processed). Processing {len(commands_to_run)} new genomes.")
        if not commands_to_run: return
        with Pool(processes=num_workers) as pool:
            list(tqdm(pool.imap_unordered(run_prodigal_command, commands_to_run), total=len(commands_to_run), desc=f"{Fore.YELLOW}Running prodigal on {genome_type}{Style.RESET_ALL}"))
    
    prepare_and_run(host_fasta_path, host_proc_dir, "host")
    prepare_and_run(phage_fasta_path, phage_proc_dir, "phage")
    print(f"{Fore.CYAN}---- Done! Processed genomes and proteins were saved into '{proc_dir}'! ----{Style.RESET_ALL}")

    current_nodes_df = pd.DataFrame(node_info_list)
    if 'gc_content' not in current_nodes_df.columns: current_nodes_df['gc_content'] = None
    if '6mer_vector' not in current_nodes_df.columns: current_nodes_df['6mer_vector'] = None
    if 'NT_embedding' not in current_nodes_df.columns: current_nodes_df['NT_embedding'] = None
    if 'RBP_embedding' not in current_nodes_df.columns: current_nodes_df['RBP_embedding'] = None
    
    if output_pkl_path.exists():
        print(f"{Fore.GREEN}Found existing node information file. Loading: {output_pkl_path}{Style.RESET_ALL}")
        existing_df = pd.read_pickle(output_pkl_path)
        current_nodes_df = pd.merge(current_nodes_df, existing_df.drop(columns=['type', 'genome_path', 'protein_path'], errors='ignore'), on='Node_id', how='left', suffixes=('_new', ''))
        for col in ['gc_content', '6mer_vector', 'NT_embedding', 'RBP_embedding']:
            if f'{col}_new' in current_nodes_df.columns:
                current_nodes_df[col] = current_nodes_df[col].fillna(current_nodes_df[f'{col}_new'])
                current_nodes_df = current_nodes_df.drop(columns=f'{col}_new')

    print(f"\n\n{Fore.CYAN}---- 02. Calculating node features... ----{Style.RESET_ALL}")
    
    # 6-mer and GC calculation
    kmer_todo_df = current_nodes_df[current_nodes_df['gc_content'].isnull()]
    print(f"6-mer and GC content: Skipping {len(current_nodes_df) - len(kmer_todo_df)} nodes (already computed). Processing {len(kmer_todo_df)} new nodes.")
    if not kmer_todo_df.empty:
        with Pool(processes=num_workers) as pool:
            results = pool.imap(calculate_kmer_gc_features, kmer_todo_df['genome_path'])
            for index, (gc, kmer_vec) in tqdm(zip(kmer_todo_df.index, results), total=len(kmer_todo_df), desc=f"{Fore.YELLOW}Calculating k-mer/GC{Style.RESET_ALL}"):
                current_nodes_df.at[index, 'gc_content'] = gc
                current_nodes_df.at[index, '6mer_vector'] = kmer_vec

        # Nucleotide Transformer embedding
        nt_todo_df = current_nodes_df[current_nodes_df['NT_embedding'].isnull()]
        print(f"\nNT embedding: Skipping {len(current_nodes_df) - len(nt_todo_df)} nodes (already computed). Processing {len(nt_todo_df)} new nodes.")
        if not nt_todo_df.empty:
            model_path = Path(nt_model_path)
            if not model_path.is_dir():
                print(f"{Fore.RED}[ERROR] NT model path not found: '{nt_model_path}'. Skipping this step.{Style.RESET_ALL}")
            else:
                print(f"Loading NT model from: {model_path}")
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
                for index, row in tqdm(nt_todo_df.iterrows(), total=len(nt_todo_df), desc=f"{Fore.YELLOW}Calculating NT Embeddings{Style.RESET_ALL}"):
                    embedding = calculate_nt_embedding((row['genome_path'], model, tokenizer, device, 10000, 0.3, index))
                    current_nodes_df.at[index, 'NT_embedding'] = embedding
                if device.startswith('cuda'): del model, tokenizer; torch.cuda.empty_cache(); print(f"{Fore.GREEN}Cleared GPU cache after NT embedding.{Style.RESET_ALL}")

        # RBP embedding for phages
        phage_df = current_nodes_df[current_nodes_df['type'] == 'phage']
        rbp_todo_df = phage_df[phage_df['RBP_embedding'].isnull()]
        print(f"\nRBP embedding: Skipping {len(phage_df) - len(rbp_todo_df)} phages (already computed). Processing {len(rbp_todo_df)} new phages.")
        if not rbp_todo_df.empty:
            rbp_detect_path, rbp_embed_path = Path(rbp_detect_model_path), Path(rbp_embed_model_path)

            # Step A: Predict and save RBPs
            if not rbp_detect_path.is_dir():
                print(f"\n{Fore.RED}[ERROR] RBP detection model path not found: '{rbp_detect_path}'. Skipping RBP prediction.{Style.RESET_ALL}")
            else:
                print(f"\n{Fore.MAGENTA}Predicting RBPs for {len(rbp_todo_df)} phages...{Style.RESET_ALL}")
                rbp_detect_tokenizer = AutoTokenizer.from_pretrained(rbp_detect_path)
                rbp_detect_model = AutoModelForSequenceClassification.from_pretrained(rbp_detect_path).to(device).eval()
                for _, row in tqdm(rbp_todo_df.iterrows(), total=len(rbp_todo_df), desc=f"{Fore.YELLOW}Predicting RBPs{Style.RESET_ALL}"):
                    protein_faa_path = Path(row['protein_path'])
                    rbp_faa_path = phage_rbp_dir / f"{protein_faa_path.stem}_RBP.faa"
                    predict_and_save_rbps(protein_faa_path, rbp_faa_path, rbp_detect_model, rbp_detect_tokenizer, device)
                if device.startswith('cuda'): del rbp_detect_model, rbp_detect_tokenizer; torch.cuda.empty_cache(); print(f"{Fore.GREEN}Cleared GPU cache after RBP prediction.{Style.RESET_ALL}")
            
            # Step B: Calculate RBP embeddings
            if not rbp_embed_path.is_dir():
                print(f"\n{Fore.RED}[ERROR] RBP embedding model path not found: '{rbp_embed_path}'. Skipping RBP embedding.{Style.RESET_ALL}")
            else:
                try:
                    model_parent_dir = str(rbp_embed_path.parent)
                    if model_parent_dir not in sys.path: sys.path.insert(0, model_parent_dir)
                    from model.tokenization_UniBioseq import UBSLMTokenizer
                    from model.modeling_UniBioseq import UniBioseqForEmbedding
                    rbp_embed_tokenizer = UBSLMTokenizer.from_pretrained(str(rbp_embed_path))
                    rbp_embed_model = UniBioseqForEmbedding.from_pretrained(str(rbp_embed_path)).to(device).eval()
                    print(f"{Fore.MAGENTA}Calculating RBP embeddings for {len(rbp_todo_df)} phages...{Style.RESET_ALL}")
                    for index, row in tqdm(rbp_todo_df.iterrows(), total=len(rbp_todo_df), desc=f"{Fore.YELLOW}Calculating RBP Embeddings{Style.RESET_ALL}"):
                        rbp_faa_path = phage_rbp_dir / f"{Path(row['protein_path']).stem}_RBP.faa"
                        embedding = calculate_rbp_embedding(rbp_faa_path, rbp_embed_model, rbp_embed_tokenizer, device)
                        current_nodes_df.at[index, 'RBP_embedding'] = embedding
                    if device.startswith('cuda'): del rbp_embed_model, rbp_embed_tokenizer; torch.cuda.empty_cache(); print(f"{Fore.GREEN}Cleared GPU cache after RBP embedding.{Style.RESET_ALL}")
                except (ImportError, Exception) as e:
                    print(f"{Fore.RED}[ERROR] An error occurred while loading the RBP embedding model: {e}{Style.RESET_ALL}"); traceback.print_exc()

        current_nodes_df.to_pickle(output_pkl_path)
        print(f"\n\n{Fore.CYAN}---- Done! Node information with all features saved to: {output_pkl_path} ----{Style.RESET_ALL}")
        print(f"\n{Fore.CYAN}✅ All node calculations are complete!{Style.RESET_ALL}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f"{Fore.BLUE}{Style.BRIGHT}Process phage-host genomes and calculate node features for GEM-PHI.{Style.RESET_ALL}", formatter_class=CustomFormatter,)

    parser.add_argument('--config', type=str, required=True, help='Path to config.yml file.',)
    parser.add_argument('--host_fasta', type=str, required=True, help='Path to host genome fasta file.',)
    parser.add_argument('--phage_fasta', type=str, required=True, help='Path to phage genome fasta file.',)
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output.',)
    parser.add_argument('--cpu_cores', type=int, default=16, help='Number of CPU cores to use for parallel processing (default: 16).',)
    parser.add_argument( '--device', type=device_type, default='0', help="Specify device for model inference. Use 'cpu', a single GPU ID (e.g., '0'), or a comma-separated list of IDs (e.g., '0,1,2').",)

    args = parser.parse_args()

    if '--help' not in sys.argv and '-h' not in sys.argv:
        import torch
        gpu_device = 'cpu'
        if args.device != 'cpu':
            if not torch.cuda.is_available():
                print(f"{Fore.RED}CUDA is not available, falling back to CPU.{Style.RESET_ALL}")
            else:
                try:
                    if isinstance(args.device, list):
                        gpu_device = args.device[0]
                    else:
                        gpu_device = f'cuda:{args.device}'
                except Exception:
                    print(f"{Fore.RED}Invalid device specified: {args.device}. Falling back to CPU.{Style.RESET_ALL}")

        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        process_genomes_for_phi_prediction(
            host_fasta_path=args.host_fasta,
            phage_fasta_path=args.phage_fasta,
            output_dir=args.output_dir,
            nt_model_path=config['paths']['nt_model_dir'],
            rbp_detect_model_path=config['paths']['rbp_detect_model_dir'],
            rbp_embed_model_path=config['paths']['rbp_embed_model_dir'],
            device=gpu_device,
            num_workers=args.cpu_cores,
        )