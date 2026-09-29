#!/usr/bin/env bash
# Single-node, paired eval only. Extra eval.py arguments apply to BOTH runs.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$#" -lt 1 ]; then
    echo "Usage: bash unused/compare_swa_niah.sh CHECKPOINT_DIR [eval.py arguments]" >&2
    exit 1
fi
PYTHON_BIN=${PYTHON_BIN:-python}
CHECKPOINT_DIR=$("${PYTHON_BIN}" - "$1" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1]).expanduser().resolve()
steps = [p for p in root.glob('step_*') if p.name[5:].isdigit() and (p / 'lit_model.pth').is_file()]
if steps:
    root = max(steps, key=lambda p: int(p.name[5:]))
if not (root / 'lit_model.pth').is_file():
    raise SystemExit(f'Checkpoint not found: {root}/lit_model.pth')
print(root)
PY
)
shift
OUTPUT_DIR=${SWA_COMPARE_OUTPUT_DIR:-${CHECKPOINT_DIR}/swa_compare_$(date +%Y%m%d_%H%M%S)}
echo "Both modes use checkpoint: ${CHECKPOINT_DIR}"
echo "Results: ${OUTPUT_DIR}"
for mode in semantic swa; do
    "${PYTHON_BIN}" -m torch.distributed.run --standalone \
        --nproc_per_node="${GPUS_PER_NODE:-8}" eval.py \
        --config "exp/qwen1.7b-32k/compare_niah_${mode}.yaml" \
        --checkpoint_dir "${CHECKPOINT_DIR}" --output_path "${OUTPUT_DIR}/${mode}" "$@"
done
