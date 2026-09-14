import os
path = r"D:\12487\NCDM PLUS2\paper\FCD_corrected.tex"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Lower block lines 465-467 (0-indexed) - replace with Math1/Math2/FrcSub data
# Note: we already fixed this above, just fixing the escaping

DL = {
    "math1":  (79.99,71.45,69.23,43.44, 81.16,72.56,71.67,42.05),
    "math2":  (78.02,70.43,71.95,44.19, 79.10,71.81,74.60,43.37),
    "frcsub": (86.39,78.17,78.01,39.10, 87.63,79.85,80.85,37.91),
}
DNAMES = ["math1","math2","frcsub"]

def br(field):
    out = []
    for dn in DNAMES:
        d = DL[dn]
        base = 0 if field == "baseline" else 4
        for j in range(4):
            out.append(f"{d[base+j]:.2f}")
    return " & ".join(out)

def bd():
    out = []
    for dn in DNAMES:
        d = DL[dn]
        for j in range(4):
            dv = d[4+j] - d[j]
            good = dv < 0 if j == 3 else dv > 0
            cmd = "redbf" if good else "green"
            out.append(f"\letterbackslash {cmd}{{{dv:.2f}}}")
    return " & ".join(out)

BASELINE = br("baseline")
FLUENCY  = br("fluency")
DELTA    = bd()

# Replace lines 465-467
lines[465] = "\\multirow{3}{*}{FairWISA} & baseline & " + BASELINE + " \\\\\n"
lines[466] = "                      & \\mymethod\\ & " + FLUENCY + " \\\\\n"
lines[467] = "                      & $\\Delta$ & " + DELTA + " \\\\\n"

with open(path, "w", encoding="utf-8") as f:
    f.writelines(lines)
print("Done")