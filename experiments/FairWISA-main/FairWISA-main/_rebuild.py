import os, sys
target = r"D:\12487\NCDM PLUS2\experiments\FairWISA-main\FairWISA-main\run_fairwisa_compare.py"

# Read current content (may be truncated from earlier)
try:
    with open(target, "r", encoding="utf-8") as f:
        old = f.read()
except:
    old = ""

# Check if the file is complete enough (has a main function)
if "def main():" not in old:
    print("File is truncated, needs full reconstruction")
    print("Lines in file:", len(old.splitlines()))
    sys.exit(1)

print("File looks OK, applying patches...")
print("Lines in file:", len(old.splitlines()))