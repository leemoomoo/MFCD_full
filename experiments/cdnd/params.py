import os

# ===== 自动定位 ncdm 根目录（不依赖运行位置）=====
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
NCDM_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

# ===== a0910 数据集路径（绝对路径）=====
dataset = os.path.join(NCDM_ROOT, "data", "a0910") + os.sep

batch_size = 128
lr = 0.002
epoch = 50

src = dataset + "train.csv"
tgt = dataset + "valid.csv"
test = dataset + "test.csv"

# ===== 从 readme.txt 读取规模信息 =====
with open(os.path.join(dataset, "readme.txt"), encoding="utf-8") as f:
    lines = f.readlines()
    # 根据你 readme 的格式：
    # max user_id: 4163
    # max item_id: 17746
    # max knowledge_id: 123
    un = int(lines[0].split(":")[1].strip())
    en = int(lines[1].split(":")[1].strip())
    kn = int(lines[2].split(":")[1].strip())

latent_dim = kn
