# coding: utf-8
# File: experiments/dina/run_dina_compare.py

import os
import sys
import json
import copy
import random
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# ===== models =====
from models.dina.DINA import DINA
from models.dina.DINA_Fluency import DINAFluency

# ===== datasets =====
from experiments.ncdm.math1_dataset import build_math1_splits_from_folder
from experiments.ncdm.math2_dataset import build_math2_splits_from_folder
from experiments.ncdm.frcsub_dataset import build_frcsub_splits_from_folder
from experiments.ncdm.Assist17_dataset import build_assist17_splits_from_folder
from experiments.ncdm.assist09_dataset import build_assist09_splits_from_folder
from experiments.ncdm.junyi_dataset import build_junyi_splits_from_csv


def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def print_meta(dataset_name: str, model_name: str, meta: dict):
    print("\n" + "=" * 70)
    print(f"Dataset: {dataset_name} | Model: {model_name}")
    print("=" * 70)
    for k, v in meta.items():
        print(f"{k}: {v}")


def safe_auc(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_pred))


def safe_acc(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_label = (np.asarray(y_prob) >= threshold).astype(int)
    return float(accuracy_score(y_true, y_label))


def safe_f1(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_label = (np.asarray(y_prob) >= threshold).astype(int)
    return float(f1_score(y_true, y_label, zero_division=0))


def safe_rmse(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    return float(np.sqrt(mean_squared_error(y_true, y_prob)))


def safe_mae(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    return float(np.mean(np.abs(y_true - y_prob)))


def get_score_by_metric(metrics: dict, metric_name: str) -> float:
    """
    Higher is better for auc/acc/f1.
    Lower is better for rmse/mae.
    """
    metric_name = metric_name.lower()
    value = metrics.get(metric_name, float("nan"))

    if np.isnan(value):
        return -float("inf")

    if metric_name in ["rmse", "mae"]:
        return -float(value)

    return float(value)


def estimate_dina_bytes(stu_num, prob_num, know_num, float_bytes=8):
    """
    Rough memory estimate for exhaustive-state DINA.

    Major arrays:
      all_states: (2^K, K)
      eta:        (2^K, J)
      like:       (N, 2^K)
      post:       (N, 2^K)
    """
    if know_num >= 60:
        return float("inf"), float("inf")

    state_num = 2 ** know_num
    total_elems = (
        state_num * know_num +          # all_states
        state_num * prob_num +          # eta
        2 * stu_num * state_num         # like + post
    )
    total_bytes = total_elems * float_bytes
    return state_num, total_bytes


def is_dina_feasible(stu_num, prob_num, know_num, max_gb=8.0):
    state_num, need_bytes = estimate_dina_bytes(stu_num, prob_num, know_num)
    if need_bytes == float("inf"):
        return False, state_num, need_bytes

    need_gb = need_bytes / (1024 ** 3)
    return need_gb <= max_gb, state_num, need_gb


def human_mem_str(x):
    if x == float("inf"):
        return "inf"
    return f"{x:.2f} GB"


def compress_q_matrix(q_m):
    """
    1) remove all-zero knowledge columns
    2) merge duplicate knowledge columns
    """
    q_m = np.asarray(q_m, dtype=np.float64)

    # remove all-zero columns
    nonzero_mask = np.any(q_m != 0, axis=0)
    q1 = q_m[:, nonzero_mask]

    if q1.shape[1] == 0:
        raise ValueError("Q-matrix became empty after removing all-zero columns.")

    # merge duplicate columns
    cols = q1.T
    uniq_cols, inverse = np.unique(cols, axis=0, return_inverse=True)
    q2 = uniq_cols.T.astype(np.float64)

    info = {
        "orig_knowledge_n": int(q_m.shape[1]),
        "after_remove_zero": int(q1.shape[1]),
        "after_merge_duplicate": int(q2.shape[1]),
    }
    return q2, info


def get_dataset(dataset_name, data_root, seed, use_fluency, alpha, beta):
    name = dataset_name.lower()

    if name == "math1":
        data_dir = os.path.join(data_root, "math2015", "Math1")
        return build_math1_splits_from_folder(
            data_dir,
            seed=seed,
            use_fluency=use_fluency,
            alpha=alpha,
            beta=beta,
            data_file="data.txt",
            q_file="q.txt",
            raw_file="rawdata.txt",
        )

    if name == "math2":
        data_dir = os.path.join(data_root, "math2015", "Math2")
        return build_math2_splits_from_folder(
            data_dir,
            seed=seed,
            use_fluency=use_fluency,
            alpha=alpha,
            beta=beta,
        )

    if name == "frcsub":
        data_dir = os.path.join(data_root, "math2015", "FrcSub")
        return build_frcsub_splits_from_folder(
            data_dir,
            seed=seed,
            use_fluency=use_fluency,
            alpha=alpha,
            beta=beta,
        )

    if name == "assist17":
        data_dir = os.path.join(data_root, "Assist17")
        return build_assist17_splits_from_folder(
            data_dir,
            use_fluency=use_fluency,
            alpha=alpha,
            beta=beta,
            seed=seed,
            response_file="response.csv",
            q_file="q_matrix.csv",
        )

    if name == "assist09":
        assist09_dir = os.path.join(data_root, "assist09")
        a0910_dir = os.path.join(data_root, "a0910")

        if os.path.isdir(assist09_dir) and os.path.isfile(os.path.join(assist09_dir, "train.csv")):
            data_dir = assist09_dir
        elif os.path.isdir(a0910_dir) and os.path.isfile(os.path.join(a0910_dir, "train.csv")):
            data_dir = a0910_dir
        else:
            raise FileNotFoundError(
                "assist09 dataset folder not found or missing train.csv. "
                f"Checked:\n  {assist09_dir}\n  {a0910_dir}"
            )

        return build_assist09_splits_from_folder(
            data_dir,
            use_fluency=use_fluency,
            alpha=alpha,
            beta=beta,
        )

    if name == "junyi":
        data_dir = os.path.join(data_root, "junyi")
        train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        valid_df = pd.read_csv(os.path.join(data_dir, "valid.csv"))
        test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

        return build_junyi_splits_from_csv(
            train_df,
            valid_df,
            test_df,
            use_fluency=use_fluency,
            alpha=alpha,
            beta=beta,
        )

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def dataset_to_dense_matrix(dataset, stu_num, prob_num, skip_value=-1):
    """
    Convert dataset samples to dense response matrix R.
    Compatible with:
      (user_id, item_id, knowledge_emb, y)
      (user_id, item_id, knowledge_emb, y, opp, acc_prior, stability)

    Returns:
      R: (stu_num, prob_num), skip entries = skip_value
    """
    R = np.full((stu_num, prob_num), skip_value, dtype=np.float64)

    for i in range(len(dataset)):
        sample = dataset[i]
        if len(sample) >= 4:
            user_id = int(sample[0])
            item_id = int(sample[1])
            y = float(sample[3])
        elif len(sample) == 3:
            user_id = int(sample[0])
            item_id = int(sample[1])
            y = float(sample[2])
        else:
            raise ValueError(f"Unexpected sample length in dataset_to_dense_matrix: {len(sample)}")

        R[user_id, item_id] = y

    return R


def extract_q_matrix(train_ds, valid_ds, test_ds, meta):
    """
    Try to get q-matrix from dataset/meta.
    Priority:
      1) dataset.q_matrix / q_m / Q
      2) meta['q_m'] / meta['q_matrix']
      3) infer from all knowledge_emb rows by item id
    """
    for ds in [train_ds, valid_ds, test_ds]:
        for attr in ["q_matrix", "q_m", "Q"]:
            if hasattr(ds, attr):
                q = getattr(ds, attr)
                q = np.asarray(q, dtype=np.float64)
                if q.ndim == 2:
                    return q

    for k in ["q_m", "q_matrix", "Q"]:
        if k in meta:
            q = np.asarray(meta[k], dtype=np.float64)
            if q.ndim == 2:
                return q

    exer_n = int(meta["exer_n"])
    knowledge_n = int(meta["knowledge_n"])
    q = np.zeros((exer_n, knowledge_n), dtype=np.float64)
    seen = np.zeros(exer_n, dtype=np.int64)

    for ds in [train_ds, valid_ds, test_ds]:
        for i in range(len(ds)):
            sample = ds[i]
            if len(sample) < 4:
                continue
            item_id = int(sample[1])
            knowledge_emb = np.asarray(sample[2], dtype=np.float64).reshape(-1)
            if knowledge_emb.shape[0] != knowledge_n:
                continue
            q[item_id] = knowledge_emb
            seen[item_id] = 1

    if np.sum(seen) != exer_n:
        missing = int(exer_n - np.sum(seen))
        raise ValueError(
            f"Failed to infer full q-matrix. Missing {missing} items. "
            "Please expose q_matrix/q_m in dataset or meta."
        )

    return q


def extract_student_fluency_from_dataset(train_ds, stu_num):
    """
    Build student-level proxy vectors from fluency dataset.

    Input samples are expected to be:
      (user_id, item_id, knowledge_emb, y, opp, acc_prior, stability)
    or
      (user_id, item_id, y, opp, acc_prior, stability)

    We aggregate by student with mean.
    """
    opp_sum = np.zeros(stu_num, dtype=np.float64)
    acc_sum = np.zeros(stu_num, dtype=np.float64)
    sta_sum = np.zeros(stu_num, dtype=np.float64)
    cnt = np.zeros(stu_num, dtype=np.float64)

    for i in range(len(train_ds)):
        sample = train_ds[i]
        if len(sample) == 7:
            user_id = int(sample[0])
            opp = float(np.asarray(sample[4]))
            acc = float(np.asarray(sample[5]))
            sta = float(np.asarray(sample[6]))
        elif len(sample) == 6:
            user_id = int(sample[0])
            opp = float(np.asarray(sample[3]))
            acc = float(np.asarray(sample[4]))
            sta = float(np.asarray(sample[5]))
        else:
            raise ValueError(
                f"Fluency dataset should return 6 or 7 fields, got len(sample)={len(sample)}"
            )

        opp_sum[user_id] += opp
        acc_sum[user_id] += acc
        sta_sum[user_id] += sta
        cnt[user_id] += 1.0

    cnt = np.maximum(cnt, 1.0)

    opportunity = opp_sum / cnt
    acc_prior = acc_sum / cnt
    stability = sta_sum / cnt

    opportunity = np.nan_to_num(opportunity, nan=0.0, posinf=0.0, neginf=0.0)
    acc_prior = np.nan_to_num(acc_prior, nan=0.5, posinf=0.5, neginf=0.5)
    stability = np.nan_to_num(stability, nan=0.5, posinf=0.5, neginf=0.5)

    return opportunity, acc_prior, stability


def dense_test_records_from_dataset(dataset):
    """
    Convert dataset to list[dict] required by DINA.eval():
      {"user_id": ..., "item_id": ..., "score": ...}
    """
    out = []
    for i in range(len(dataset)):
        sample = dataset[i]
        if len(sample) >= 4:
            user_id = int(sample[0])
            item_id = int(sample[1])
            score = float(sample[3])
        elif len(sample) == 3:
            user_id = int(sample[0])
            item_id = int(sample[1])
            score = float(sample[2])
        else:
            raise ValueError(f"Unexpected sample length in dense_test_records_from_dataset: {len(sample)}")

        out.append({
            "user_id": user_id,
            "item_id": item_id,
            "score": score,
        })
    return out


def predict_scores_from_model(model, records):
    """
    Build prediction list aligned with records.
    """
    if isinstance(model, DINAFluency):
        pred_score = model._build_pred_score_matrix()
    else:
        slip = np.asarray(model.slip, dtype=np.float64)
        guess = np.asarray(model.guess, dtype=np.float64)
        pred_score = (1 - slip) * model.eta + guess * (1 - model.eta)
        pred_score = np.clip(np.nan_to_num(pred_score, nan=0.5), 1e-3, 1 - 1e-3)
        pred_score = pred_score[model.theta, :]

    y_true, y_pred = [], []
    for rec in records:
        stu = rec["user_id"]
        item = rec["item_id"]
        score = rec["score"]
        pred = float(pred_score[stu, item])
        pred = float(np.nan_to_num(pred, nan=0.5, posinf=0.999, neginf=0.001))
        y_true.append(score)
        y_pred.append(pred)

    return y_true, y_pred


def evaluate_dina_metrics(model, test_records):
    y_true, y_pred = predict_scores_from_model(model, test_records)

    return {
        "auc": safe_auc(y_true, y_pred),
        "acc": safe_acc(y_true, y_pred),
        "f1": safe_f1(y_true, y_pred),
        "rmse": safe_rmse(y_true, y_pred),
        "mae": safe_mae(y_true, y_pred),
    }


def train_dina_with_best_epoch(model, valid_records, args, model_name="dina"):
    """
    Note:
    Here "epoch" really means one external EM step call:
        model.train(epoch=1, epsilon=args.epsilon)
    """
    best_epoch = -1
    best_valid_metrics = None
    best_state = None
    best_score = -float("inf")
    history = []

    for epoch_i in range(args.epochs):
        model.train(epoch=1, epsilon=args.epsilon)

        valid_metrics = evaluate_dina_metrics(model, valid_records)

        row = {
            "epoch": epoch_i,
            "train_loss": float("nan"),
            "train_y_loss": float("nan"),
            "train_opp_loss": 0.0,
            "train_acc_loss": 0.0,
            "train_sta_loss": 0.0,
            "valid_auc": valid_metrics["auc"],
            "valid_acc": valid_metrics["acc"],
            "valid_f1": valid_metrics["f1"],
            "valid_rmse": valid_metrics["rmse"],
            "valid_mae": valid_metrics["mae"],
        }
        history.append(row)

        print(
            f"[{model_name.upper()}][Epoch {epoch_i}] "
            f"valid_auc={row['valid_auc']:.6f} "
            f"valid_acc={row['valid_acc']:.6f} "
            f"valid_f1={row['valid_f1']:.6f} "
            f"valid_rmse={row['valid_rmse']:.6f} "
            f"valid_mae={row['valid_mae']:.6f}"
        )

        score = get_score_by_metric(valid_metrics, args.select_metric)

        if score > best_score:
            best_score = score
            best_epoch = epoch_i
            best_valid_metrics = valid_metrics
            best_state = {
                "theta": copy.deepcopy(model.theta),
                "slip": copy.deepcopy(model.slip),
                "guess": copy.deepcopy(model.guess),
            }
            if isinstance(model, DINAFluency):
                best_state.update({
                    "opportunity": copy.deepcopy(model.opportunity),
                    "acc_prior": copy.deepcopy(model.acc_prior),
                    "stability": copy.deepcopy(model.stability),
                    "gamma": copy.deepcopy(model.gamma),
                    "flu_weights": copy.deepcopy(model.flu_weights),
                    "flu_bias": copy.deepcopy(model.flu_bias),
                })

    if best_state is not None:
        model.theta = best_state["theta"]
        model.slip = best_state["slip"]
        model.guess = best_state["guess"]
        if isinstance(model, DINAFluency):
            model.opportunity = best_state["opportunity"]
            model.acc_prior = best_state["acc_prior"]
            model.stability = best_state["stability"]
            model.gamma = best_state["gamma"]
            model.flu_weights = best_state["flu_weights"]
            model.flu_bias = best_state["flu_bias"]

    return {
        "best_epoch": best_epoch,
        "best_valid_metrics": best_valid_metrics,
        "history": history,
        "best_metric_name": args.select_metric,
        "best_metric_score": best_score,
    }


def build_model(model_type, train_ds, valid_ds, test_ds, meta, args):
    stu_num = int(meta["student_n"])
    prob_num = int(meta["exer_n"])

    q_raw = extract_q_matrix(train_ds, valid_ds, test_ds, meta)
    q_m, q_info = compress_q_matrix(q_raw)
    know_num = int(q_m.shape[1])

    print(
        f"[Q-compress] original_K={q_info['orig_knowledge_n']} "
        f"-> nonzero_K={q_info['after_remove_zero']} "
        f"-> merged_K={q_info['after_merge_duplicate']}"
    )

    feasible, state_num, need = is_dina_feasible(
        stu_num=stu_num,
        prob_num=prob_num,
        know_num=know_num,
        max_gb=args.max_dina_gb,
    )
    if not feasible:
        raise ValueError(
            f"Exact DINA infeasible after Q compression: "
            f"dataset stats: stu_num={stu_num}, prob_num={prob_num}, compressed_know_num={know_num}, "
            f"state_num=2^{know_num}={state_num}, estimated_memory={human_mem_str(need)}. "
            f"Please skip this dataset for exact DINA or use an approximate CDM."
        )

    R_train = dataset_to_dense_matrix(train_ds, stu_num, prob_num, skip_value=args.skip_value)

    if model_type == "dina":
        return DINA(
            R=R_train,
            q_m=q_m,
            stu_num=stu_num,
            prob_num=prob_num,
            know_num=know_num,
            skip_value=args.skip_value,
        )

    if model_type == "fluency":
        opportunity, acc_prior, stability = extract_student_fluency_from_dataset(train_ds, stu_num)

        return DINAFluency(
            R=R_train,
            q_m=q_m,
            stu_num=stu_num,
            prob_num=prob_num,
            know_num=know_num,
            skip_value=args.skip_value,
            opportunity=opportunity,
            acc_prior=acc_prior,
            stability=stability,
            gamma=args.gamma,
            flu_weights=np.array([args.w_opp, args.w_acc, args.w_sta], dtype=np.float64),
            flu_bias=args.flu_bias,
        )

    raise ValueError(f"Unsupported model type: {model_type}")


def run_one_model_one_dataset(args, dataset_name, model_type):
    use_fluency = (model_type == "fluency")

    train_ds, valid_ds, test_ds, meta = get_dataset(
        dataset_name=dataset_name,
        data_root=args.data_root,
        seed=args.seed,
        use_fluency=use_fluency,
        alpha=args.alpha,
        beta=args.beta,
    )

    print_meta(dataset_name, model_type, meta)

    valid_records = dense_test_records_from_dataset(valid_ds)
    test_records = dense_test_records_from_dataset(test_ds)

    model = build_model(model_type, train_ds, valid_ds, test_ds, meta, args)
    train_info = train_dina_with_best_epoch(
        model=model,
        valid_records=valid_records,
        args=args,
        model_name=model_type,
    )
    test_metrics = evaluate_dina_metrics(model, test_records)

    print(f"\n[Best Epoch] dataset={dataset_name}, model={model_type}, epoch={train_info['best_epoch']}")
    print(
        f"[Best Valid by {train_info['best_metric_name']}] "
        f"AUC={train_info['best_valid_metrics']['auc']:.6f}  "
        f"ACC={train_info['best_valid_metrics']['acc']:.6f}  "
        f"F1={train_info['best_valid_metrics']['f1']:.6f}  "
        f"RMSE={train_info['best_valid_metrics']['rmse']:.6f}  "
        f"MAE={train_info['best_valid_metrics']['mae']:.6f}"
    )
    print(
        f"[Test @ Best Epoch] "
        f"AUC={test_metrics['auc']:.6f}  "
        f"ACC={test_metrics['acc']:.6f}  "
        f"F1={test_metrics['f1']:.6f}  "
        f"RMSE={test_metrics['rmse']:.6f}  "
        f"MAE={test_metrics['mae']:.6f}"
    )

    save_dir = os.path.join(args.save_root, model_type, dataset_name)
    ensure_dir(save_dir)

    model_path = os.path.join(save_dir, "best_model.pkl")
    model.save(model_path)

    save_json(os.path.join(save_dir, "history.json"), train_info["history"])

    metrics = {
        "dataset": dataset_name,
        "model": model_type,
        "student_n": int(meta["student_n"]),
        "exer_n": int(meta["exer_n"]),
        "knowledge_n_raw": int(meta["knowledge_n"]),
        "knowledge_n_used": int(model.know_num),
        "n_train": int(meta["n_train"]),
        "n_valid": int(meta["n_valid"]),
        "n_test": int(meta["n_test"]),
        "best_epoch": int(train_info["best_epoch"]),
        "select_metric": train_info["best_metric_name"],
        "best_valid_auc": float(train_info["best_valid_metrics"]["auc"]),
        "best_valid_acc": float(train_info["best_valid_metrics"]["acc"]),
        "best_valid_f1": float(train_info["best_valid_metrics"]["f1"]),
        "best_valid_rmse": float(train_info["best_valid_metrics"]["rmse"]),
        "best_valid_mae": float(train_info["best_valid_metrics"]["mae"]),
        "test_auc": float(test_metrics["auc"]),
        "test_acc": float(test_metrics["acc"]),
        "test_f1": float(test_metrics["f1"]),
        "test_rmse": float(test_metrics["rmse"]),
        "test_mae": float(test_metrics["mae"]),
        "slip_mean": float(np.mean(model.slip)),
        "slip_min": float(np.min(model.slip)),
        "slip_max": float(np.max(model.slip)),
        "guess_mean": float(np.mean(model.guess)),
        "guess_min": float(np.min(model.guess)),
        "guess_max": float(np.max(model.guess)),
        "model_path": model_path,
    }

    if isinstance(model, DINAFluency):
        flu = model._compute_student_fluency()
        metrics.update({
            "gamma": float(model.gamma),
            "flu_bias": float(model.flu_bias),
            "flu_mean": float(np.mean(flu)),
            "flu_std": float(np.std(flu)),
        })

    save_json(os.path.join(save_dir, "metrics.json"), metrics)
    return metrics


def build_comparison_table(rows):
    out = []
    grouped = {}

    for r in rows:
        if "error" in r:
            continue
        grouped.setdefault(r["dataset"], {})
        grouped[r["dataset"]][r["model"]] = r

    for dataset, dd in grouped.items():
        row = {
            "dataset": dataset,

            "dina_best_epoch": dd["dina"]["best_epoch"] if "dina" in dd else None,
            "dina_test_auc": dd["dina"]["test_auc"] if "dina" in dd else None,
            "dina_test_acc": dd["dina"]["test_acc"] if "dina" in dd else None,
            "dina_test_f1": dd["dina"]["test_f1"] if "dina" in dd else None,
            "dina_test_rmse": dd["dina"]["test_rmse"] if "dina" in dd else None,
            "dina_test_mae": dd["dina"]["test_mae"] if "dina" in dd else None,

            "fluency_best_epoch": dd["fluency"]["best_epoch"] if "fluency" in dd else None,
            "fluency_test_auc": dd["fluency"]["test_auc"] if "fluency" in dd else None,
            "fluency_test_acc": dd["fluency"]["test_acc"] if "fluency" in dd else None,
            "fluency_test_f1": dd["fluency"]["test_f1"] if "fluency" in dd else None,
            "fluency_test_rmse": dd["fluency"]["test_rmse"] if "fluency" in dd else None,
            "fluency_test_mae": dd["fluency"]["test_mae"] if "fluency" in dd else None,
        }

        if "dina" in dd and "fluency" in dd:
            row["delta_auc"] = dd["fluency"]["test_auc"] - dd["dina"]["test_auc"]
            row["delta_acc"] = dd["fluency"]["test_acc"] - dd["dina"]["test_acc"]
            row["delta_f1"] = dd["fluency"]["test_f1"] - dd["dina"]["test_f1"]
            row["delta_rmse"] = dd["fluency"]["test_rmse"] - dd["dina"]["test_rmse"]
            row["delta_mae"] = dd["fluency"]["test_mae"] - dd["dina"]["test_mae"]
        else:
            row["delta_auc"] = None
            row["delta_acc"] = None
            row["delta_f1"] = None
            row["delta_rmse"] = None
            row["delta_mae"] = None

        out.append(row)

    return out


def fmt_metric(x):
    if x is None:
        return "None"
    try:
        if np.isnan(x):
            return "nan"
    except TypeError:
        pass
    return f"{x:.6f}"


def parse_args():
    parser = argparse.ArgumentParser(description="Compare original DINA and DINAFluency")

    parser.add_argument("--datasets", type=str, default="all")
    parser.add_argument("--models", type=str, default="dina,fluency")
    parser.add_argument("--data-root", type=str, default=os.path.join(ROOT, "data"))
    parser.add_argument("--save-root", type=str, default=os.path.join(ROOT, "checkpoints", "dina_compare"))
    parser.add_argument("--seed", type=int, default=2025)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument(
        "--select-metric",
        type=str,
        default="rmse",
        choices=["auc", "acc", "f1", "rmse", "mae"],
        help="Metric used on validation set to choose the best epoch."
    )

    parser.add_argument("--skip-value", type=float, default=-1)
    parser.add_argument(
        "--max-dina-gb",
        type=float,
        default=8.0,
        help="Safety memory limit for exact exhaustive-state DINA after Q compression."
    )

    # fluency parameters
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--w-opp", type=float, default=0.2)
    parser.add_argument("--w-acc", type=float, default=1.0)
    parser.add_argument("--w-sta", type=float, default=0.5)
    parser.add_argument("--flu-bias", type=float, default=0.0)

    # proxy build
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)
    ensure_dir(args.save_root)
    save_json(os.path.join(args.save_root, "run_config.json"), vars(args))

    all_datasets = ["math1", "math2", "frcsub", "assist17", "assist09", "junyi"]
    all_models = ["dina", "fluency"]

    if args.datasets.strip().lower() == "all":
        dataset_list = all_datasets
    else:
        dataset_list = [x.strip().lower() for x in args.datasets.split(",") if x.strip()]

    if args.models.strip().lower() == "all":
        model_list = all_models
    else:
        model_list = [x.strip().lower() for x in args.models.split(",") if x.strip()]

    summary = []

    for dataset_name in dataset_list:
        for model_type in model_list:
            try:
                metrics = run_one_model_one_dataset(args, dataset_name, model_type)
                summary.append(metrics)
            except Exception as e:
                print(f"\n[ERROR] dataset={dataset_name}, model={model_type}")
                print(repr(e))
                summary.append({
                    "dataset": dataset_name,
                    "model": model_type,
                    "error": repr(e),
                })

    save_json(os.path.join(args.save_root, "summary.json"), summary)

    comparison = build_comparison_table(summary)
    save_json(os.path.join(args.save_root, "comparison.json"), comparison)

    if len(comparison) > 0:
        pd.DataFrame(comparison).to_csv(
            os.path.join(args.save_root, "comparison.csv"),
            index=False
        )

    print("\n" + "=" * 90)
    print(f"Comparison Summary (best epoch selected by valid {args.select_metric})")
    print("=" * 90)

    if len(comparison) == 0:
        print("No successful comparison rows.")
    else:
        for row in comparison:
            print(
                f'{row["dataset"]}\n'
                f'  DINA    -> epoch={row["dina_best_epoch"]}, '
                f'AUC={fmt_metric(row["dina_test_auc"])}, '
                f'ACC={fmt_metric(row["dina_test_acc"])}, '
                f'F1={fmt_metric(row["dina_test_f1"])}, '
                f'RMSE={fmt_metric(row["dina_test_rmse"])}, '
                f'MAE={fmt_metric(row["dina_test_mae"])}\n'
                f'  Fluency -> epoch={row["fluency_best_epoch"]}, '
                f'AUC={fmt_metric(row["fluency_test_auc"])}, '
                f'ACC={fmt_metric(row["fluency_test_acc"])}, '
                f'F1={fmt_metric(row["fluency_test_f1"])}, '
                f'RMSE={fmt_metric(row["fluency_test_rmse"])}, '
                f'MAE={fmt_metric(row["fluency_test_mae"])}\n'
                f'  Delta   -> '
                f'AUC={fmt_metric(row["delta_auc"])}, '
                f'ACC={fmt_metric(row["delta_acc"])}, '
                f'F1={fmt_metric(row["delta_f1"])}, '
                f'RMSE={fmt_metric(row["delta_rmse"])}, '
                f'MAE={fmt_metric(row["delta_mae"])}'
            )


if __name__ == "__main__":
    main()