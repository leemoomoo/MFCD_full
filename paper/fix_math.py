with open("D:/12487/NCDM PLUS2/paper/FCD_corrected.tex","r",encoding="utf-8") as f:
    content = f.read()
# Fix the specific corruption on line 654  
content = content.replace(
    "with {\\mathrm{Total}}$ closely tracking $,",
    "with $L_{\\mathrm{Total}}$ closely tracking $L_y$,"
)
with open("D:/12487/NCDM PLUS2/paper/FCD_corrected.tex","w",encoding="utf-8") as f:
    f.write(content)
print("Fixed math corruption")
