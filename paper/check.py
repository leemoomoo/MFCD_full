import re
with open('D:/12487/NCDM PLUS2/paper/FCD_corrected.tex','r',encoding='utf-8') as f:
    lines = f.readlines()
for i,l in enumerate(lines):
    s=l.strip()
    if s and not s.startswith('%') and '\\subsection{' in s:
        nm=s.split('{')[1].split('}')[0]
        print('SEC: '+nm)
        for j in range(i+1,len(lines)):
            if lines[j].strip().startswith('\\subsection{') and not lines[j].strip().startswith('%'):
                break
            if '\\includegraphics' in lines[j] and not lines[j].strip().startswith('%'):
                m=re.search(r'{([^}]+)}',lines[j])
                if m and m.group(1).endswith(('png','jpg','pdf')):
                    print('  FIG: '+m.group(1))
