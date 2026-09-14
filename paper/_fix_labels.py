import re

path = r"D:\12487\NCDM PLUS2\experiments\ncdm\train_plot_losses.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

d = chr(36)  # dollar sign
bs = chr(92) # backslash

# Fix all corrupted labels
fixes = [
    (d + "{" + bs + "mathrm{Total}}" + d + " [Eq.17]", d + "L_{" + bs + "mathrm{Total}}" + d + " [Eq.17]"),
    (d + " [Eq.18]", d + "L_y" + d + " [Eq.18]"),
    (d + "{" + bs + "mathrm{ea}}" + d + " [Eq.19]", d + "L_{" + bs + "mathrm{ea}}" + d + " [Eq.19]"),
    (d + "{" + bs + "mathrm{rc}}" + d + " [Eq.20]", d + "L_{" + bs + "mathrm{rc}}" + d + " [Eq.20]"),
    (d + "{" + bs + "mathrm{ei}}" + d + " [Eq.21]", d + "L_{" + bs + "mathrm{ei}}" + d + " [Eq.21]"),
]

for old, new in fixes:
    if old in content:
        content = content.replace(old, new)
        print("Fixed:", old[:30])

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
