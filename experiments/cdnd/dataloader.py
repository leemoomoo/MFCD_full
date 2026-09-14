import torch
from torch.utils.data import DataLoader
import params
import csv


def my_collate(batch):
    input_stu_ids, input_exer_ids, input_knowledge_embs, ys = [], [], [], []
    for log in batch:
        if log['knowledge_code'] == []:
            knowledge_emb = [1.0] * params.kn
        else:
            knowledge_emb = [0.] * params.kn
            for knowledge_code in log['knowledge_code']:
                # 1-based -> 0-based
                if 1 <= knowledge_code <= params.kn:
                    knowledge_emb[knowledge_code - 1] = 1.0

        y = log['score']
        input_stu_ids.append(log['user_id'] - 1)
        input_exer_ids.append(log['exer_id'] - 1)
        input_knowledge_embs.append(knowledge_emb)
        ys.append(y)

    return (
        torch.LongTensor(input_stu_ids),
        torch.LongTensor(input_exer_ids),
        torch.Tensor(input_knowledge_embs),
        torch.Tensor(ys)
    )


def _parse_knowledge_cell(cell: str):
    """把 knowledge 字段解析成 [k1,k2,...]（支持 1|5|9 或单个数字）"""
    cell = (cell or "").strip()
    if cell == "" or cell.lower() == "nan":
        return []
    if "|" in cell:
        return [int(x) for x in cell.split("|") if x.strip() != ""]
    if ";" in cell:
        return [int(x) for x in cell.split(";") if x.strip() != ""]
    return [int(float(cell))]


def _read_a0910_csv(path):
    logs = []

    # 先把前几行读出来（调试用）
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        preview = [next(f, "").rstrip("\n") for _ in range(3)]
    print(f"[CSV PREVIEW] {path}")
    for line in preview:
        print("  ", line)

    # 重新打开，用 csv 解析
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return []

    # 判断是否有表头：表头里通常有字母
    first_row = rows[0]
    has_header = any(any(c.isalpha() for c in col) for col in first_row)

    # 如果有表头，用 DictReader（更稳）
    if has_header:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            dr = csv.DictReader(f)
            for r in dr:
                # 兼容列名：user_id / user / uid
                u = r.get("user_id") or r.get("user") or r.get("uid")
                it = r.get("exer_id") or r.get("item_id") or r.get("exer") or r.get("item")
                k = r.get("knowledge_id") or r.get("knowledge_code") or r.get("knowledge") or r.get("skill_id")
                y = r.get("score") or r.get("label") or r.get("correct")

                if u is None or it is None or y is None:
                    continue

                logs.append({
                    "user_id": int(float(u)),
                    "exer_id": int(float(it)),
                    "knowledge_code": _parse_knowledge_cell(k) if k is not None else [],
                    "score": float(y)
                })
        return logs

    # 没表头：按前4列固定解析
    for row in rows:
        if len(row) < 4:
            continue
        u, it, k, y = row[0], row[1], row[2], row[3]
        logs.append({
            "user_id": int(float(u)),
            "exer_id": int(float(it)),
            "knowledge_code": _parse_knowledge_cell(k),
            "score": float(y)
        })

    return logs


def CD_DL():
    src_dataset = _read_a0910_csv(params.src)
    tgt_dataset = _read_a0910_csv(params.tgt)

    print("train samples =", len(src_dataset), "val samples =", len(tgt_dataset))
    if len(src_dataset) == 0:
        raise ValueError(f"读取训练集为空：{params.src}（请检查 CSV 分隔符/表头/列名/文件内容）")

    src_DL = DataLoader(dataset=src_dataset, batch_size=params.batch_size, shuffle=True, collate_fn=my_collate)
    tgt_DL = DataLoader(dataset=tgt_dataset, batch_size=params.batch_size, shuffle=False, collate_fn=my_collate)
    return src_DL, tgt_DL
