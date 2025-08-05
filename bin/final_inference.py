import sys
import os
import yaml
import argparse
from pathlib import Path
from colorama import Fore, Style, init


init(autoreset=True)

if '--help' not in sys.argv and '-h' not in sys.argv:
    import torch
    import pandas as pd
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from functools import partial


class CustomFormatter(argparse.HelpFormatter):
    def _format_action_invocation(self, action):
        if not action.option_strings or action.nargs == 0:
            return super()._format_action_invocation(action)
        return ' '.join(action.option_strings) + ' ' + self._get_default_metavar_for_optional(action)

    def _get_help_string(self, action):
        help_string = super()._get_help_string(action)
        if action.dest in ['config_path', 'input_dir', 'output_dir']:
            return f"{Fore.GREEN}{help_string}{Style.RESET_ALL}"
        return help_string

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(
        description=f"{Fore.BLUE}{Style.BRIGHT}Run GEM-PHI inference on new data.{Style.RESET_ALL}",
        formatter_class=CustomFormatter
    )
    parser.add_argument('--config_path', type=Path, required=True, help='Path to the config.yml file.')
    parser.add_argument('--input_dir', type=Path, required=True, help='Directory containing the preprocessed node_information.pkl, PH_edges_information.pkl, PP_top5.pkl, and HH_top10.pkl.')
    parser.add_argument('--output_dir', type=Path, required=True, help='Directory to save inference results.')
    args = parser.parse_args()

    if '--help' not in sys.argv and '-h' not in sys.argv:
        print(f"{Fore.CYAN}--- GEM-PHI Inference Script ---{Style.RESET_ALL}")

        try:
            config = load_config(args.config_path)
            gemphi_project_path = config['paths']['gemphi_project_path']
            sys.path.append(gemphi_project_path)
            
            from model.Inference import (load_model_and_base_data, augment_graph_with_new_data, prepare_prediction_data, EvaluationDataset, collate_fn_predict, run_inference)

        except (FileNotFoundError, ImportError) as e:
            print(f"{Fore.RED}Error during setup: {e}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Please ensure your config file is correct and the GEMPHI project structure is intact.{Style.RESET_ALL}")
            sys.exit(1)

        required_files = {
            'Node_info': args.input_dir / 'node_information.pkl',
            'PH_edges': args.input_dir / 'PH_edges_information.pkl',
            'PP_edges': args.input_dir / 'PP_top5.pkl',
            'HH_edges': args.input_dir / 'HH_top10.pkl'
        }
        
        missing_files = [name for name, path in required_files.items() if not path.exists()]
        if missing_files:
            print(f"{Fore.RED}Error: The following required preprocessed files are missing in '{args.input_dir}':{Style.RESET_ALL}")
            for file_name in missing_files:
                print(f"  - {file_name}: {required_files[file_name]}")
            print(f"{Fore.YELLOW}Please ensure you have run the data preprocessing scripts to generate these files.{Style.RESET_ALL}")
            sys.exit(1)

        results_output_dir = args.output_dir / 'results'
        os.makedirs(results_output_dir, exist_ok=True)
        
        model, base_graph, base_id_maps, device = load_model_and_base_data(config)
        
        print(f"\n{Fore.CYAN}Loading data...{Style.RESET_ALL}")
        node_info_df = pd.read_pickle(required_files['Node_info'])
        ph_edges_df = pd.read_pickle(required_files['PH_edges'])
        pp_edges_df = pd.read_pickle(required_files['PP_edges'])
        hh_edges_df = pd.read_pickle(required_files['HH_edges'])

        augmented_graph, updated_id_maps = augment_graph_with_new_data(base_graph, base_id_maps, node_info_df, pp_edges_df, hh_edges_df)
        prediction_list = prepare_prediction_data(ph_edges_df)
        predict_dataset = EvaluationDataset(prediction_list)
        collate_fn = partial(collate_fn_predict, host_map=updated_id_maps['host_id_map'], phage_map=updated_id_maps['phage_id_map'])
        predict_loader = DataLoader(predict_dataset, batch_size=config['prediction']['batch_size'], shuffle=False, collate_fn=collate_fn)
        results_df = run_inference(model, predict_loader, device, augmented_graph)

        raw_output_path = results_output_dir / 'host_phage_predictions_raw.csv'
        results_df.sort_values(by='probability', ascending=False).to_csv(raw_output_path, index=False)
        filtered_df = results_df[results_df['probability'] >= config['prediction']['confidence_threshold']]
        
        if not filtered_df.empty:
            grouped_df = filtered_df.groupby('host')['phage'].apply(lambda x: '|'.join(x)).reset_index()
            grouped_df.rename(columns={'phage': 'phage_list'}, inplace=True)
            
            output_path = results_output_dir / 'host_phage_predictions.csv'
            grouped_df.to_csv(output_path, index=False)
            
            print(f"\n{Fore.CYAN}--- Inference Complete! ---{Style.RESET_ALL}")
            print(f"Found positive predictions for {Fore.GREEN}{len(filtered_df)}{Style.RESET_ALL} pairs, grouped into {Fore.GREEN}{len(grouped_df)}{Style.RESET_ALL} hosts.")
            print(f"Final aggregated results saved to: {Fore.GREEN}{output_path}{Style.RESET_ALL}")
            print(f"Raw probability results saved to: {Fore.GREEN}{raw_output_path}{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.YELLOW}--- Inference Complete! ---{Style.RESET_ALL}")
            print(f"No predictions met the confidence threshold of {Fore.YELLOW}{config['prediction']['confidence_threshold']}{Style.RESET_ALL}.")
            print(f"Raw probability results saved to: {Fore.GREEN}{raw_output_path}{Style.RESET_ALL}")

if __name__ == '__main__':
    main()