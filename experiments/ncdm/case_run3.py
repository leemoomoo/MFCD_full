import os, sys, numpy as np, torch, pandas as pd
from torch.utils.data import DataLoader
ROOT = r"D:\12487\NCDM PLUS2"
if ROOT not in sys.path: sys.path.insert(0, ROOT)
from models.ncdm.NCDM_Fluency import NCDMFluency
from experiments.ncdm.run_NCDM_Fluency import Assist09FluencyDataset
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_ds = Assist09FluencyDataset(os.path.join(ROOT, "data", "assist09", "splits", "assist09_train.csv"))
test_ds = Assist09FluencyDataset(os.path.join(ROOT, "data", "assist09", "splits", "assist09_test.csv"),
    user_map=train_ds.user_map, item_map=train_ds.item_map, skill_map=train_ds.skill_map,
    user_oov_id=train_ds.user_oov_id)
print(f"Students={train_ds.student_n} Exers={train_ds.exer_n} Skills={train_ds.knowledge_n}")

m = NCDMFluency(knowledge_n=train_ds.knowledge_n, exer_n=train_ds.exer_n, student_n=train_ds.student_n)
ckpt = os.path.join(ROOT, "checkpoints", "best_ncdm_fluency_assist09.pt")
m.net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
m.net = m.net.to(DEVICE); m.net.eval()

def extract(batch):
    ui, ii, ke, y, op, ap, st = [t.to(DEVICE) for t in batch]
    with torch.no_grad():
        out = m.net(ui, ii, ke, op, ap, st, return_aux=True)
    se = torch.sigmoid(m.net.student_emb(ui))
    kd = torch.sigmoid(m.net.k_difficulty(ii))
    ed = torch.sigmoid(m.net.e_difficulty(ii))
    x = ed * (se - kd) * ke.float()
    x = m.net.drop_1(torch.sigmoid(m.net.prednet_full1(x)))
    x = m.net.drop_2(torch.sigmoid(m.net.prednet_full2(x)))
    ol = m.net.prednet_full3(x)
    oc = m.net._to_col(op.float()); ac = m.net._to_col(ap.float()); sc = m.net._to_col(st.float())
    fp = m.net._safe_log1p(oc); fa = torch.clamp(ac, 0.0, 1.0)
    fs = 1.0 - torch.clamp(4.0 * torch.clamp(sc, min=0.0), 0.0, 1.0)
    br = torch.cat([fp, fa, fs], dim=1)
    z = m.net.beh_encoder(br); z = z + torch.tanh(m.net.student_flu_bias(ui))
    sf = torch.sigmoid(m.net.flu_score_head(z))
    fe = torch.cat([ed, kd.mean(dim=1, keepdim=True)], dim=1)
    gi = torch.cat([z, fe], dim=1); gj = torch.sigmoid(m.net.fuse_gate(gi))
    dl = (sf - 0.5) * 4.0
    return {"o_logit": ol.item(), "core_prob": torch.sigmoid(ol).item(),
            "s_flu": sf.item(), "g_ij": gj.item(), "delta_ij": dl.item(),
            "y_pred": out["p"].item(), "y_true": y.item(),
            "beh_fp": fp.item(), "beh_fa": fa.item(), "beh_fs": fs.item()}

rec = []
ld = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
for i, b in enumerate(ld):
    if i % 5000 == 0: print(f"  {i}/{len(ld)}")
    try:
        rec.append(extract(b))
    except Exception as e:
        print(f"  err@{i}: {e}")
        if i > 100: break

df = pd.DataFrame(rec)
print(f"\nTotal: {len(df)}")
df["pc"] = (df["y_pred"] - df["core_prob"]).abs()
df["as"] = np.sign(df["y_pred"] - df["core_prob"])
c = df[(df["g_ij"] > 0.7) & (df["delta_ij"] < -0.5) & (df["as"] < 0)].sort_values("pc", ascending=False)
print(f"High-gate + neg-delta candidates: {len(c)}")
SHOW = 8
sc = c.head(SHOW) if len(c) >= SHOW else df[df["g_ij"] > 0.5].sort_values("pc", ascending=False).head(SHOW)

print("\n" + "=" * 96)
hdr = f"  {'idx':>6s} | {'o_logit':>8s} | {'sigma(o)':>8s} | {'s_flu':>6s} | {'g_ij':>6s} | {'delta':>7s} | {'y_pred':>7s} | {'y':>3s} | beh[EI EA RC]"
print(hdr)
print("=" * 96)
for idx, row in sc.iterrows():
    print(f"  {int(idx):>6d} | {row['o_logit']:+8.4f} | {row['core_prob']:8.4f} | {row['s_flu']:6.4f} | {row['g_ij']:6.4f} | {row['delta_ij']:+7.4f} | {row['y_pred']:7.4f} | {int(row['y_true']):3d} | [{row['beh_fp']:.3f} {row['beh_fa']:.3f} {row['beh_fs']:.3f}]")

out_dir = os.path.join(ROOT, "outputs", "case_study"); os.makedirs(out_dir, exist_ok=True)
df.to_csv(os.path.join(out_dir, "all_records.csv"), index=False)
sc.to_csv(os.path.join(out_dir, "showcase.csv"), index=False)
print(f"\nSaved to outputs/case_study/")
print("Done.")
