import os
import time
import json
import pickle
import argparse
import ast
import numpy as np

import torch
import pandas as pd

from model import NCD_con, NCD_gen, NCD
from utils import Logger, transform, load_from_json, sort_and_fill_dict

logger = Logger()


def log(s):
    print(s)
    logger.log(str(s) + '\n')


def safe_load_json(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        content = f.read().strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start:end + 1])
        raise ValueError(f"无法解析 JSON 文件: {path}\n内容:\n{content}")


def check_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少文件: {path}")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-e', '--epoch', default=40, type=int)
    argparser.add_argument('-lr', '--learning-rate', default=1e-3, type=float)
    argparser.add_argument('-b', '--batch-size', default=256, type=int)
    argparser.add_argument('-g', '--gpu', type=str, default='cuda')
    argparser.add_argument('-i', '--index', default=0, type=int)
    argparser.add_argument('-mn', '--model-name', default='def', type=str)
    argparser.add_argument('-ds', '--dataset', default='python', type=str)
    args = argparser.parse_args()

    model_arch = args.model_name
    model_name = model_arch + "-NCD"
    dataset_name = args.dataset
    batch_size = args.batch_size
    epoch = args.epoch
    index = args.index
    learning_rate = args.learning_rate
    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')

    base_dir = os.path.dirname(os.path.abspath(__file__))

    data_dir = os.path.join(base_dir, 'dataset', 'dataset', 'open', dataset_name)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    log_dir = os.path.join(base_dir, '..', 'logs', model_name)
    model_dir = os.path.join(base_dir, '..', 'model', model_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    current_time = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
    log_name = os.path.join(log_dir, f"{model_name}_{current_time}.log")
    logger.set_filename(log_name)

    log(f"start training {model_name}")
    log(f"data_dir: {data_dir}")
    log(f"log file: {log_name}")

    item_csv = os.path.join(data_dir, 'item.csv')
    train_csv = os.path.join(data_dir, 'train.csv')
    valid_csv = os.path.join(data_dir, 'valid.csv')
    test_csv = os.path.join(data_dir, 'test.csv')
    info_json = os.path.join(data_dir, 'info.json')

    check_file(item_csv)
    check_file(train_csv)
    check_file(valid_csv)
    check_file(test_csv)
    check_file(info_json)

    df_item = pd.read_csv(item_csv)

    raw_item2knowledge = {}
    knowledge_set = set()

    for _, s in df_item.iterrows():
        item_id = int(s['item_id'])
        knowledge_codes = list(set(ast.literal_eval(str(s['knowledge_code']))))
        raw_item2knowledge[item_id] = knowledge_codes
        knowledge_set.update(knowledge_codes)

    knowledge_list = sorted(list(knowledge_set))
    knowledge_map = {k: i for i, k in enumerate(knowledge_list)}

    item2knowledge = {}
    for item_id, codes in raw_item2knowledge.items():
        item2knowledge[item_id] = [knowledge_map[c] for c in codes]

    info = safe_load_json(info_json)
    user_n = int(info['student_cnt'])
    item_n = int(info['problem_cnt'])
    knowledge_n = len(knowledge_list)

    log(f"user_n={user_n}, item_n={item_n}, knowledge_n={knowledge_n}")
    log(f"max_raw_knowledge_code={max(knowledge_list)}")

    train_data = pd.read_csv(train_csv)
    valid_data = pd.read_csv(valid_csv)
    test_data = pd.read_csv(test_csv)

    train_set = transform(
        train_data["user_id"],
        train_data["item_id"],
        item2knowledge,
        train_data["score"],
        batch_size,
        knowledge_n
    )

    valid_set = transform(
        valid_data["user_id"],
        valid_data["item_id"],
        item2knowledge,
        valid_data["score"],
        batch_size,
        knowledge_n
    )

    test_set = transform(
        test_data["user_id"],
        test_data["item_id"],
        item2knowledge,
        test_data["score"],
        batch_size,
        knowledge_n
    )

    log("data loaded.")

    user_emb_path = os.path.join(data_dir, "user_emb.pkl")
    item_emb_path = os.path.join(data_dir, "item_emb.pkl")

    if os.path.exists(user_emb_path) and os.path.exists(item_emb_path):
        with open(user_emb_path, 'rb') as f:
            usr_emb = pickle.load(f)

        with open(item_emb_path, 'rb') as f:
            itm_emb = pickle.load(f)

        log("loaded existing embeddings.")
    else:
        emb_dim = 128
        log(f"embedding files not found, creating random embeddings with dim={emb_dim}")

        rng = np.random.default_rng(2026)
        usr_emb = rng.normal(0, 0.01, size=(user_n, emb_dim)).astype(np.float32)
        itm_emb = rng.normal(0, 0.01, size=(item_n, emb_dim)).astype(np.float32)

        with open(user_emb_path, 'wb') as f:
            pickle.dump(usr_emb, f)

        with open(item_emb_path, 'wb') as f:
            pickle.dump(itm_emb, f)

        log("random embeddings saved.")

    best_model = os.path.join(model_dir, f"{current_time}_{index}_best_model")

    if model_arch == "def":
        cdm = NCD(
            knowledge_n, item_n, user_n,
            log_name, best_model,
            usr_emb, itm_emb
        )

    elif model_arch == "gen":
        stu_activity_path = os.path.join(data_dir, "stu_activity.json")
        exer_activity_path = os.path.join(data_dir, "exer_activity.json")

        check_file(stu_activity_path)
        check_file(exer_activity_path)

        stu_ratio = safe_load_json(stu_activity_path)
        exer_ratio = safe_load_json(exer_activity_path)

        stu_ratio = [1 - x for x in stu_ratio]
        exer_ratio = [1 - x for x in exer_ratio]

        cdm = NCD_gen(
            knowledge_n, item_n, user_n,
            log_name, best_model,
            usr_emb, itm_emb,
            stu_ratio, exer_ratio
        )

    elif model_arch == "con":
        similar_students_path = os.path.join(data_dir, "similar_students.json")
        similar_exercises_path = os.path.join(data_dir, "similar_exercises.json")

        check_file(similar_students_path)
        check_file(similar_exercises_path)

        similar_students = load_from_json(similar_students_path)
        similar_students = sort_and_fill_dict(similar_students, user_n)

        similar_exercises = load_from_json(similar_exercises_path)
        similar_exercises = sort_and_fill_dict(similar_exercises, item_n)

        cdm = NCD_con(
            knowledge_n, item_n, user_n,
            log_name, best_model,
            usr_emb, itm_emb,
            similar_students, similar_exercises
        )
    else:
        raise ValueError("model-name 必须是 def / gen / con")

    cdm.train(train_set, valid_set, epoch=epoch, device=device, lr=learning_rate)

    cdm.load(best_model)
    auc, accuracy, rmse = cdm.eval(test_set, device=device, save=True)
    log(f"TEST RESULT: AUC={auc:.6f}, ACC={accuracy:.6f}, RMSE={rmse:.6f}")