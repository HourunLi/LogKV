import sys
from pathlib import Path

lines = Path(sys.argv[1] if len(sys.argv) > 1 else "gpu_usage.txt").read_text().splitlines()
usage = {}
for start in range(0, len(lines), 14):
    account = lines[start + 10].strip()
    units = int(lines[start + 8])
    usage[account] = usage.get(account, 0) + units
for account, units in sorted(usage.items(), key=lambda item: item[1]):
    print(f"{units:4d} {account}")
print(f"{sum(usage.values()):4d} all")
