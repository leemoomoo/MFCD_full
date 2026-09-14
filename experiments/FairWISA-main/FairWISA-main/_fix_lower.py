import os
path = r"D:\12487\NCDM PLUS2\paper\FCD_corrected.tex"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find 2nd FairWISA block (lower = math1/math2/frcsub)
count = 0
base = -1
for i, line in enumerate(lines):
    if "FairWISA" in line and "multirow" in line:
        count += 1
        if count == 2:
            base = i
            break

if base < 0:
    print("Second FairWISA block not found")
    exit()

# Data: (main_auc, main_acc, main_f1, main_rmse, flu_auc, flu_acc, flu_f1, flu_rmse)
data = {
    "math1":  (79.99,71.45,69.23,43.44,81.16,72.56,71.67,42.05),
    "math2":  (78.02,70.43,71.95,44.19,79.10,71.81,74.60,43.37),
    "frcsub": (86.39,78.17,78.01,39.10,87.63,79.85,80.85,37.91),
}
dnames = ["math1", "math2", "frcsub"]

def build(field):
    vals = []
    for dn in dnames:
        d = data[dn]
        base_i = 0 if field == "base" else 4
        for j in range(4):
            vals.append(f"{d[base_i+j]:.2f}")
    return " & ".join(vals)

def build_delta():
    vals = []
    for dn in dnames:
        d = data[dn]
        for j in range(4):
            dv = d[4+j] - d[j]
            good = dv < 0 if j == 3 else dv > 0
            cmd = "redbf" if good else "green"
            vals.append(f"\\{cmd}" + "{" + f"{dv:.2f}" + "}")
    return " & ".join(vals)

base_line = build("base")
flu_line = build("flu")
del_line = build_delta()

# Construct proper LaTeX lines using concatenation to avoid escape issues
L0 = "\\multirow{3}{*}{FairWISA} & baseline & " + base_line + " \\\\\n"
L1 = "                      & \\mymethod\\ & " + flu_line + " \\\\\n"
L2 = "                      & $\\Delta$ & " + del_line + " \\\\\n"

lines[base] = L0
lines[base+1] = L1
lines[base+2] = L2

with open(path, "w", encoding="utf-8") as f:
    f.writelines(lines)
print("OK: lines", base+1, "-", base+3)