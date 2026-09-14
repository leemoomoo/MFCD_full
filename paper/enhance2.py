import re
with open("D:/12487/NCDM PLUS2/paper/FCD_corrected.tex","r",encoding="utf-8") as f:
    content = f.read()

# ===== Fix corrupted math first =====
content = content.replace(
    "shows the total loss {\\mathrm{Total}}$ and the primary BCE loss $.",
    "shows the total loss $L_{\\mathrm{Total}}$ and the primary BCE loss $L_y$."
)
content = content.replace(
    "{\\mathrm{ea}}$ (empirical accuracy), {\\mathrm{rc}}$ (response consistency), and {\\mathrm{ei}}$",
    "$L_{\\mathrm{ea}}$ (empirical accuracy), $L_{\\mathrm{rc}}$ (response consistency), and $L_{\\mathrm{ei}}$"
)

# ===== Rewrite Ablation Study paragraph =====
old_ablation = (
    "To evaluate the contribution of each component in \\mymethod\\, we conduct an ablation study "
    "with four model variants: Base (mastery only), L1 (Base + static fluency with fixed weight), "
    "L2 (Base + fluency with learned static fusion), and Full \\mymethod\\ (Base + fluency with dynamic fusion gate). "
    "Figure~\\ref{fig:ablation_study} reports the performance across AUC, ACC, F1 and RMSE. "
    "As shown, Full \\mymethod\\ consistently achieves the best performance across all four metrics, "
    "demonstrating that each component contributes positively to the overall framework."
)

# Try to find and replace - account for truncation
pos = content.find("To evaluate the contribution of each component in \\mymethod\\")
if pos >= 0:
    end_pos = content.find("\\begin{figure}[ht]", pos)
    if end_pos >= 0:
        before = content[pos:end_pos]
        print(f"Found ablation text at pos {pos}, len={len(before)}")
        # Replace just the text before the figure
        content = content[:pos] + new_ablation + content[end_pos:]

# Actually, let me try a simpler approach - check what's actually in the file
lines = content.split("\n")
for i, line in enumerate(lines):
    if "contribution of each component" in line:
        print(f"Ablation text at line {i+1}: {line[:100]}...")
    if "total loss" in line and "\\mathrm" in line:
        print(f"Main loss text at line {i+1}: {line[:120]}...")
    if "empirical accuracy" in line and "\\mathrm" in line:
        print(f"Aux loss text at line {i+1}: {line[:120]}...")

with open("D:/12487/NCDM PLUS2/paper/FCD_corrected.tex","w",encoding="utf-8") as f:
    f.write(content)
print("\\nDone checking")
