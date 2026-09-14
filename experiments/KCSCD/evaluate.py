import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score


def evaluate(model, data_loader, batch_size, device, args):
    model.eval() 
    total_preds, total_targets = [], []
    correct_count = 0
    exer_count = 0

    with torch.no_grad():
        while not data_loader.is_end():
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels, affects = data_loader.next_batch(args.isAffect)
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels, affects = input_stu_ids.to(
                device), input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device), affects.to(device)
            out_put, _, _ = model.forward(input_stu_ids, input_exer_ids, input_knowledge_embs, labels)
            # out_put = out_put.detach().cpu().numpy()
            output = out_put.view(-1)
            for i in range(len(labels)):
                if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                    correct_count += 1
            exer_count += len(labels)
            output = output.cpu().numpy()
            total_preds += output.tolist()
            total_targets.extend(labels.cpu().numpy())

    total_preds = np.array(total_preds)
    total_targets = np.array(total_targets)
    # total_targets = np.array(total_targets, dtype=object).flatten()
    
    pred_label = (total_preds > 0.5).astype(int)
    target_label = total_targets.astype(int)
    
    metrics = {
        "ACC": accuracy_score(target_label, pred_label),
        "RMSE": np.sqrt(np.mean((total_targets - total_preds) ** 2)),
        "AUC": roc_auc_score(total_targets, total_preds)
    }
    return metrics
