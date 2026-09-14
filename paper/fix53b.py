with open("D:/12487/NCDM PLUS2/paper/FCD_corrected.tex", "r", encoding="utf-8") as f:
    lines = f.readlines()

# We need to swap:
# Old: fig2 (619-624) -> lambda text (625) -> dim text (627) -> fig3 (630-635)
# New: dim text -> fig3 (h!) -> lambda text -> fig2 (h!)

# Extract the four blocks
fig2_block = lines[618:624]   # lines 619-624: fig2 figure
lambda_text = lines[624]       # line 625: lambda text
dim_text = lines[626]          # line 627: dim text
fig3_block = lines[629:635]    # lines 630-635: fig3 figure

# Fix [t] -> [h!]
fig2_block[0] = fig2_block[0].replace("[t]", "[h!]")
fig3_block[0] = fig3_block[0].replace("[t]", "[h!]")

# New order: dim_text -> fig3_block -> lambda_text -> fig2_block
new_block = [dim_text, "\n"] + fig3_block + ["\n", lambda_text, "\n"] + fig2_block

# Replace old lines 619-635 with new content
lines[618:635] = new_block

with open("D:/12487/NCDM PLUS2/paper/FCD_corrected.tex", "w", encoding="utf-8") as f:
    f.writelines(lines)
print("SUCCESS: Figures reordered and [t] -> [h!]")
