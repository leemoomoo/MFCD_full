# coding: utf-8
# Comparison: FairWISA-main (NCDM+Fairness) vs FairWISA-fluency (Fluency+Fairness)
import os, sys, copy, random, argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)
from models.ncdm.NCDM_FluencyFair import FairNCDMFluency, LearnableMatrix
from models.ncdm.NCDM import NCDM
from experiments.ncdm.assist09_dataset import build_assist09_splits_from_folder

def set_seed(s=2025):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed(s); torch.cuda.manual_seed_all(s)
def ensure_dir(d): os.makedirs(d, exist_ok=True)
def save_json(p, o):
    with open(p, "w", encoding="utf-8") as f: json.dump(o, f, indent=2)

def safe_auc(yt, yp):
    yt, yp = np.asarray(yt), np.asarray(yp)
    return float(roc_auc_score(yt, yp)) if len(np.unique(yt)) > 1 else float("nan")
def safe_acc(yt, yp, t=0.5):
    return float(accuracy_score(np.asarray(yt).astype(int), (np.asarray(yp)>=t).astype(int)))
def safe_f1(yt, yp, t=0.5):
    return float(f1_score(np.asarray(yt).astype(int), (np.asarray(yp)>=t).astype(int), zero_division=0))
def safe_rmse(yt, yp):
    return float(np.sqrt(mean_squared_error(np.asarray(yt), np.asarray(yp))))

def load_data(dd, bs=256):
    ds = build_assist09_splits_from_folder(dd, use_fluency=True)
    meta = ds[3]; pin = torch.cuda.is_available()
    tl = DataLoader(ds[0], bs, shuffle=True, num_workers=0, pin_memory=pin)
    vl = DataLoader(ds[1], bs, shuffle=False, num_workers=0, pin_memory=pin)
    xl = DataLoader(ds[2], bs, shuffle=False, num_workers=0, pin_memory=pin)
    class SL:
        def __init__(self, l): self.l = l
        def __iter__(self):
            for b in self.l: yield b[:4]
        def __len__(self): return len(self.l)
    return meta, (tl, vl, xl), (SL(tl), SL(vl), SL(xl))

def eval_ncdm(m, loader, device):
    m.ncdm_net.to(device).eval()
    yt, yp = [], []
    with torch.no_grad():
        for i0, i1, kq, y in loader:
            p = m.ncdm_net(i0.to(device), i1.to(device), kq.to(device))
            yp.extend(p.cpu().tolist()); yt.extend(y.cpu().tolist())
    return {"auc":safe_auc(yt,yp),"acc":safe_acc(yt,yp),"f1":safe_f1(yt,yp),"rmse":safe_rmse(yt,yp)}

def eval_fluency(m, loader, device):
    m.net.to(device).eval()
    yt, yp = [], []
    with torch.no_grad():
        for uid, iid, kq, y, opp, ac, st in loader:
            uid, iid, kq = uid.to(device), iid.to(device), kq.to(device)
            opp, ac, st = opp.to(device), ac.to(device), st.to(device)
            p = m.net(uid, iid, kq, opp, ac, st, return_aux=False)
            yp.extend(p.cpu().tolist()); yt.extend(y.cpu().tolist())
    return {"auc":safe_auc(yt,yp),"acc":safe_acc(yt,yp),"f1":safe_f1(yt,yp),"rmse":safe_rmse(yt,yp)}

def discover_groups(model, train_loader, k, device, epoch=30, lr=0.05):
    mat = LearnableMatrix(model.net.student_emb.num_embeddings, k).to(device)
    opt = torch.optim.Adam(mat.parameters(), lr=lr)
    model.net.to(device).eval()
    for e in range(epoch):
        losses = []
        for batch in tqdm(train_loader, f"  GE{e}"):
            uid, iid, kq, y, opp, ac, st = [x.to(device) for x in batch]
            with torch.no_grad():
                p = model.net(uid, iid, kq, opp, ac, st, return_aux=False)
            gp = mat(uid, hard=True)
            i0, i1 = [], []
            for g in range(k):
                m = gp[:, g] > 0.5
                if m.sum() < 2: continue
                gp_, gy_ = p[m], y[m]
                z0, z1 = gy_ == 0, gy_ == 1
                if z0.sum() == 0 or z1.sum() == 0: continue
                i0.append((gp_[z0] - 0.0).mean()); i1.append((1.0 - gp_[z1]).mean())
            il = torch.stack(i0).var()+torch.stack(i1).var() if len(i0)>=2 else torch.tensor(0., device=device)
            l = -il; opt.zero_grad(); l.backward(); opt.step(); losses.append(l.item())
        print(f"  [GE{e}] v={np.mean(losses):.6f}")
    return mat

def train_main(meta, tr, va, gm, args):
    print(chr(10)+"="*65+chr(10)+"  FairWISA-main (NCDM+Fairness)"+chr(10)+"="*65)
    m = NCDM(knowledge_n=meta["knowledge_n"], exer_n=meta["exer_n"], student_n=meta["student_n"])
    m.ncdm_net.to(args.device).train()
    G = gm.get_groups(torch.arange(meta["student_n"], device=args.device))
    opt = torch.optim.Adam(m.ncdm_net.parameters(), lr=args.lr)
    bce = nn.BCELoss(); best_auc, best_sd = 0., None
    for ep in range(args.epochs_fair):
        m.ncdm_net.train(); le, li = [], []
        for batch in tr:
            i0, i1, kq, y = [x.to(args.device) for x in batch]
            p = m.ncdm_net(i0, i1, kq); g = G[i0]
            bl, i0l, i1l = [], [], []
            for gi in range(args.num_groups):
                mk = g == gi; gp, gy = p[mk], y[mk]
                if gp.numel() == 0: continue
                bl.append(bce(gp, gy))
                z0, z1 = gy == 0, gy == 1
                if z0.sum()>0 and z1.sum()>0:
                    i0l.append((gp[z0]-0.).mean()); i1l.append((1.-gp[z1]).mean())
            el = torch.stack(bl).sum()
            il = torch.stack(i0l).var()+torch.stack(i1l).var() if len(i0l)>=2 else torch.tensor(0., device=args.device)
            (el + args.fairness_weight*il).backward(); opt.step()
            le.append(el.item()); li.append(il.item())
        me = eval_ncdm(m, va, args.device)
        print(f"  [E{ep}] e={np.mean(le):.4f} i={np.mean(li):.4f} auc={me['auc']:.4f} acc={me['acc']:.4f}")
        if me["auc"] > best_auc: best_auc = me["auc"]; best_sd = copy.deepcopy(m.ncdm_net.state_dict())
    if best_sd is not None: m.ncdm_net.load_state_dict(best_sd)
    return m

def train_flu(meta, tl, vl, gm, args):
    print(chr(10)+"="*65+chr(10)+"  FairWISA-fluency"+chr(10)+"="*65)
    m = FairNCDMFluency(knowledge_n=meta["knowledge_n"], exer_n=meta["exer_n"],
        student_n=meta["student_n"], num_groups=args.num_groups,
        fairness_weight=args.fairness_weight, fluency_lambda=args.fluency_lambda,
        dropout=args.dropout, flu_latent_dim=args.flu_latent_dim)
    m.unify_weights(); m.group_matrix = gm
    m.net.to(args.device).train()
    opt = torch.optim.Adam(m.net.parameters(), lr=args.lr)
    best_auc, best_sd = 0., None
    for ep in range(args.epochs_fair):
        m.net.train(); le, li, la = [], [], []
        for batch in tl:
            uid, iid, kq, y, opp, ac, st = [x.to(args.device) for x in batch]
            out = m.net(uid, iid, kq, opp, ac, st, return_aux=True)
            p = out["p"]; g = m.group_matrix.get_groups(uid)
            bl, i0l, i1l = [], [], []
            for gi in range(args.num_groups):
                mk = g == gi; gp, gy = p[mk], y[mk]
                if gp.numel() == 0: continue
                bl.append(m.loss_y(gp, gy))
                z0, z1 = gy == 0, gy == 1
                if z0.sum()>0 and z1.sum()>0:
                    i0l.append((gp[z0]-0.).mean()); i1l.append((1.-gp[z1]).mean())
            el = torch.stack(bl).sum()
            al = m.fluency_lambda*(m.loss_opp(out["opp_hat"],out["opp_tgt"])+m.loss_acc(out["acc_hat"],out["acc_tgt"])+m.loss_sta(out["sta_hat"],out["sta_tgt"]))
            il = torch.stack(i0l).var()+torch.stack(i1l).var() if len(i0l)>=2 else torch.tensor(0.,device=args.device)
            (el + al + m.fairness_weight*il).backward(); opt.step()
            le.append(el.item()); li.append(il.item()); la.append(al.item())
        me = eval_fluency(m, vl, args.device)
        print(f"  [E{ep}] e={np.mean(le):.4f} i={np.mean(li):.4f} a={np.mean(la):.4f} auc={me['auc']:.4f} acc={me['acc']:.4f}")
        if me["auc"] > best_auc: best_auc = me["auc"]; best_sd = copy.deepcopy(m.net.state_dict())
    if best_sd is not None: m.net.load_state_dict(best_sd)
    return m

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=os.path.join(ROOT, "data", "a0910"))
    p.add_argument("--save-root", default=os.path.join(ROOT, "outputs","results","fairwisa_compare"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--num-groups", type=int, default=2)
    p.add_argument("--fairness-weight", type=float, default=10.0)
    p.add_argument("--epochs-groups", type=int, default=30)
    p.add_argument("--epochs-fair", type=int, default=20)
    p.add_argument("--fluency-lambda", type=float, default=0.5)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--flu-latent-dim", type=int, default=16)
    return p.parse_args()

def main():
    args = parse_args(); set_seed(args.seed); ensure_dir(args.save_root)
    meta, (tl,vl,xl), (t4,v4,x4) = load_data(args.data_dir, args.batch_size)
    print(f"Data: s={meta['student_n']} e={meta['exer_n']} k={meta['knowledge_n']} train={meta['n_train']} valid={meta['n_valid']} test={meta['n_test']}")
    bb = FairNCDMFluency(knowledge_n=meta["knowledge_n"], exer_n=meta["exer_n"],
        student_n=meta["student_n"], fluency_lambda=args.fluency_lambda).to(args.device)
    gm = discover_groups(bb, tl, args.num_groups, args.device, epoch=args.epochs_groups)
    mm = train_main(meta, t4, v4, gm, args)
    tm = eval_ncdm(mm, x4, args.device)
    mf = train_flu(meta, tl, vl, gm, args)
    tf = eval_fluency(mf, xl, args.device)
    print(chr(10)+"="*65+chr(10)+"  COMPARISON RESULTS"+chr(10)+"="*65)
    for n, m_ in [("FairWISA-main", tm), ("FairWISA-fluency", tf)]:
        print(f"  {n:20s}: AUC={m_['auc']:.6f} ACC={m_['acc']:.6f} F1={m_['f1']:.6f} RMSE={m_['rmse']:.6f}")
    d = {k: tf[k]-tm[k] for k in tm}
    print(f"  {'Delta':20s}: AUC={d['auc']:+.6f} ACC={d['acc']:+.6f} F1={d['f1']:+.6f} RMSE={d['rmse']:+.6f}")
    save_json(os.path.join(args.save_root,"results.json"),
        {"meta":meta,"config":vars(args),"main":tm,"fluency":tf,"delta":d})
    print(f"Results saved to {args.save_root}")

if __name__ == "__main__":
    main()
