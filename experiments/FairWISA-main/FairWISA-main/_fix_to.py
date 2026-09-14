import os, sys
path = r"D:\12487\NCDM PLUS2\experiments\FairWISA-main\FairWISA-main\run_fairwisa_compare.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Fix: replace .to(device) on the model
# Change ".to(args.device)" that appears right after FairWISAFluency(...)
bad = '.to(args.device)'
good = '\n    bb.net.to(args.device)'

# Only replace the one at the bb = FairWISAFluency(...) line
content = content.replace(
    "bb = FairWISAFluency(user_num=meta[\"student_n\"], item_num=meta[\"exer_n\"], fluency_lambda=args.fluency_lambda).to(args.device)",
    "bb = FairWISAFluency(user_num=meta[\"student_n\"], item_num=meta[\"exer_n\"], fluency_lambda=args.fluency_lambda)\n    bb.net.to(args.device)"
)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Fixed _to_ issue")
print("Changed:", content.count("bb.net.to") )