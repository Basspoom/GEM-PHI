import argparse
import subprocess
import yaml
import sys
from pathlib import Path
import os
from colorama import Fore, Style, init

init(autoreset=True)

def main():
    """
    Orchestrates the execution of the three GEM-PHI pipeline scripts
    (node features, edge features, and inference) in their respective Conda environments.
    """
    parser = argparse.ArgumentParser(
        description=f"{Fore.BLUE}{Style.BRIGHT}Run the GEM-PHI pipeline: node feature calculation, edge feature calculation, and final inference.{Style.RESET_ALL}"
    )
    parser.add_argument('--config_path', type=Path, required=True, help=f'{Fore.GREEN}Path to the config.yml file.{Style.RESET_ALL}')
    parser.add_argument('--host_fasta', type=Path, required=True, help=f'{Fore.GREEN}Path to host genome fasta file.{Style.RESET_ALL}')
    parser.add_argument('--phage_fasta', type=Path, required=True, help=f'{Fore.GREEN}Path to phage genome fasta file.{Style.RESET_ALL}')
    parser.add_argument('--output_dir', type=Path, required=True, help=f'{Fore.GREEN}Main output directory.{Style.RESET_ALL}')
    parser.add_argument('--num_workers', type=int, default=16, help=f'{Fore.GREEN}Number of CPU cores for parallel processing (default: 16).{Style.RESET_ALL}')
    parser.add_argument('--device', type=str, default='0', help=f"{Fore.YELLOW}Specify device for model inference. Use 'cpu', a single GPU ID (e.g., '0'), or a comma-separated list of IDs (e.g., '0,1').{Style.RESET_ALL}")
    args = parser.parse_args()

    try:
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"{Fore.RED}Error: config file not found at {args.config_path}{Style.RESET_ALL}")
        sys.exit(1)

    current_script_dir = Path(__file__).parent
    node_script_path = current_script_dir / "calculate_node_features.py"
    edge_script_path = current_script_dir / "calculate_edge_features.py"
    inference_script_path = current_script_dir / "final_inference.py"

    try:
        node_env = config['environments']['node_features']
        edge_env = config['environments']['edge_features']
        inference_env = config['environments']['inference']
    except KeyError:
        print(f"{Fore.RED}Error: 'environments' section not found in config.yml or a key is missing.{Style.RESET_ALL}")
        sys.exit(1)

    print(f"\n{Fore.CYAN}{Style.BRIGHT}--- Starting GEM-PHI pipeline with output directory: {args.output_dir} ---{Style.RESET_ALL}")

    print(f"\n{Fore.CYAN}--- Step 1/3: Calculate node features in environment '{node_env}' ---{Style.RESET_ALL}")
    node_cmd = [
        "conda", "run", "-n", node_env, "python", str(node_script_path),
        "--config", str(args.config_path),
        "--host_fasta", str(args.host_fasta),
        "--phage_fasta", str(args.phage_fasta),
        "--output_dir", str(args.output_dir),
        "--cpu_cores", str(args.num_workers),
        "--device", str(args.device)
    ]
    subprocess.run(node_cmd, check=True)
    print(f"{Fore.GREEN}--- Step 1 Completed ---{Style.RESET_ALL}")

    print(f"\n{Fore.CYAN}--- Step 2/3: Calculate edge features in environment '{edge_env}' ---{Style.RESET_ALL}")
    edge_cmd = [
        "conda", "run", "-n", edge_env, "python", str(edge_script_path),
        "--config_path", str(args.config_path),
        "--output_dir", str(args.output_dir),
        "--num_workers", str(args.num_workers),
        "--device", str(args.device)
    ]
    subprocess.run(edge_cmd, check=True)
    print(f"{Fore.GREEN}--- Step 2 Completed ---{Style.RESET_ALL}")

    print(f"\n{Fore.CYAN}--- Step 3/3: Final inference in environment '{inference_env}' ---{Style.RESET_ALL}")
    inference_cmd = [
        "conda", "run", "-n", inference_env, "python", str(inference_script_path),
        "--config_path", str(args.config_path),
        "--input_dir", str(args.output_dir),
        "--output_dir", str(args.output_dir)
    ]
    subprocess.run(inference_cmd, check=True)
    print(f"{Fore.GREEN}--- Step 3 Completed ---{Style.RESET_ALL}")

    print(f"\n{Fore.GREEN}{Style.BRIGHT}--- GEM-PHI pipeline completed successfully! ---{Style.RESET_ALL}")

if __name__ == "__main__":
    main()