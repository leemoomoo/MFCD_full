import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


class Logger:
    def __init__(self):
        self.log_file = None
        self.log_dir = None

    def log(self, s):
        if self.log_file is not None:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(s)

    def set_logdir(self, log_dir):
        self.log_dir = log_dir

    def set_filename(self, filename):
        self.log_file = filename

    def close(self):
        pass


def transform(user, item, item2knowledge, score, batch_size, knowledge_n):
    knowledge_emb = torch.zeros((len(item), knowledge_n), dtype=torch.float32)

    user = np.array(user)
    item = np.array(item)
    score = np.array(score)

    for idx in range(len(item)):
        item_id = item[idx]
        if item_id in item2knowledge:
            knowledge_emb[idx][np.array(item2knowledge[item_id], dtype=int)] = 1.0

    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64),
        torch.tensor(item, dtype=torch.int64),
        knowledge_emb,
        torch.tensor(score, dtype=torch.float32)
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)


def load_from_json(filepath):
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def sort_and_fill_dict(input_dict, max_key, min_key=0):
    if not input_dict:
        return {}

    example_length = len(next(iter(input_dict.values())))
    all_keys = list(range(min_key, max_key))

    for key in all_keys:
        if key not in input_dict:
            input_dict[key] = [0] * example_length

    sorted_dict = {key: input_dict[key] for key in sorted(all_keys)}
    return sorted_dict