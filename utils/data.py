import pandas as pd
from torch.utils.data import Dataset

class EvaluationDataset(Dataset):
    def __init__(self, file_path: str):
        self.file_path = file_path
        try:
            self.samples_df = pd.read_pickle(self.file_path)
            print(f"成功加载评估文件: {self.file_path}, 共 {len(self.samples_df)} 个样本。")
        except FileNotFoundError:
            print(f"!!! 致命错误: 找不到评估文件 '{self.file_path}' !!!")
            print("!!! 请确保您已经成功运行了 build_final_eval_set.py 脚本来生成此文件。 !!!")
            self.samples_df = pd.DataFrame()

    def __len__(self):
        return len(self.samples_df)

    def __getitem__(self, idx):
        if self.samples_df.empty:
            raise IndexError(f"无法从空的Dataset中获取项目。请检查文件 '{self.file_path}' 是否正确生成。")
        return self.samples_df.iloc[idx]