with open('D:/12487/NCDM PLUS2/paper/FCD_corrected.tex','r',encoding='utf-8') as f:
    lines = f.readlines()
for i in range(615, 638):
    print(f'{i+1}: {lines[i].rstrip()}')
