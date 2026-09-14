with open("D:/12487/NCDM PLUS2/paper/FCD_corrected.tex","r",encoding="utf-8") as f:
    content = f.read()

# Fix corrupted math mode - cases where $L_ was eaten by PowerShell
content = content.replace(
    "shows the total loss {\\mathrm{Total}}$ and the primary BCE loss $.",
    "shows the total loss $L_{\\mathrm{Total}}$ and the primary BCE loss $L_y$."
)
content = content.replace(
    "{\\mathrm{ea}}$ (empirical accuracy), {\\mathrm{rc}}$ (response consistency), and {\\mathrm{ei}}$",
    "$L_{\\mathrm{ea}}$ (empirical accuracy), $L_{\\mathrm{rc}}$ (response consistency), and $L_{\\mathrm{ei}}$"
)

with open("D:/12487/NCDM PLUS2/paper/FCD_corrected.tex","w",encoding="utf-8") as f:
    f.write(content)
print("Fixed math mode corruption")
