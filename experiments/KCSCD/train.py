import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from data_loader import DataLoader, ValTestDataLoader
from model import Net
from torch import optim
import time
import os
from datetime import datetime
import logging
from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建日志和模型保存目录
log_dir = os.path.join("logs", "junyi")
model_dir = os.path.join("model", "junyi")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# 日志文件路径
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_path = os.path.join(log_dir, f"training_{current_time}.log")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()


def train(args):
    logger.info(f"Using device: {device}")
    logger.info("Using {} dataset".format(args.dataset))

    train_data_loader = DataLoader(args)
    val_data_loader = ValTestDataLoader(args, "val")
    test_data_loader = ValTestDataLoader(args, "test")

    net = Net(args).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    affect_loss_function = nn.MSELoss()
    criterion = nn.BCELoss()

    logger.info("-------------")
    logger.info("Training!!!!")

    best_val_auc = 0
    best_val_acc = 0

    for epoch in range(args.epoch_n):
        epoch_start_time = time.time()

        net.train()
        train_data_loader.reset()
        val_data_loader.reset()
        test_data_loader.reset()

        running_loss = 0.0
        batch_count = 0
        pred_all, label_all = [], []

        while not train_data_loader.is_end():
            batch_count += 1

            input_stu_ids, input_exer_ids, input_knowledge_embs, labels, affects = train_data_loader.next_batch(args.isAffect)

            input_stu_ids = input_stu_ids.to(device)
            input_exer_ids = input_exer_ids.to(device)
            input_knowledge_embs = input_knowledge_embs.to(device)
            labels = labels.to(device)
            affects = affects.to(device)

            optimizer.zero_grad()

            output_1, affect_p, CL_loss = net.forward(
                input_stu_ids,
                input_exer_ids,
                input_knowledge_embs,
                labels
            )

            output_1 = output_1.squeeze(1)
            labels_ = labels.float()

            # 如果你后面想把 affect loss 加回来，可以用下面这一行替换
            # loss = criterion(output_1, labels_) + args.alpha * affect_loss_function(affect_p, affects) + CL_loss
            loss = criterion(output_1, labels_) + CL_loss

            loss.backward()
            optimizer.step()
            net.apply_clipper()

            running_loss += loss.item()
            pred_all += output_1.detach().cpu().tolist()
            label_all += labels.detach().cpu().tolist()

        pred_all = np.array(pred_all)
        label_all = np.array(label_all)
        pred_all_ = (pred_all > 0.5).astype(int)

        accuracy = accuracy_score(label_all, pred_all_)
        rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
        auc = roc_auc_score(label_all, pred_all)

        average_loss = running_loss / batch_count
        logger.info(f"Average train loss for epoch {epoch + 1}: {average_loss:.6f}")
        logger.info(f"Train metrics: ACC = {accuracy:.4f}, RMSE = {rmse:.4f}, AUC = {auc:.4f}")

        val_metrics = evaluate(net, val_data_loader, args.batch_size, device, args)
        logger.info(
            f"Validation metrics: ACC = {val_metrics['ACC']:.4f}, "
            f"RMSE = {val_metrics['RMSE']:.4f}, "
            f"AUC = {val_metrics['AUC']:.4f}"
        )

        if val_metrics["ACC"] > best_val_acc:
            best_val_acc = val_metrics["ACC"]
            best_val_auc = val_metrics["AUC"]

            model_name = os.path.join(model_dir, f"NIKCD_best_net_epoch{epoch + 1}.pth")
            torch.save(
                {
                    "model": net.state_dict()
                },
                model_name
            )
            logger.info(f"Saved new best model at epoch {epoch + 1} with val_acc: {best_val_acc:.4f}")
            logger.info(f"Saved new best model at epoch {epoch + 1} with val_auc: {best_val_auc:.4f}")
            logger.info(f"Model parameters saved to {model_name}!")

        test_metrics = evaluate(net, test_data_loader, args.batch_size, device, args)
        logger.info(
            f"Test metrics: ACC = {test_metrics['ACC']:.4f}, "
            f"RMSE = {test_metrics['RMSE']:.4f}, "
            f"AUC = {test_metrics['AUC']:.4f}"
        )

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        logger.info(f"Epoch {epoch + 1} duration: {epoch_duration:.2f} seconds\n")