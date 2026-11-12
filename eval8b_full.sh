#!/usr/bin/env bash
# One complete 8B evaluation: the 32K main line followed by 32K–256K length passes.
set -euo pipefail

usage() {
    echo "Usage: bash eval8b_full.sh [word_first|baseline|query|word_first_query|cpt_dense|dense]"
    echo "Default: word_first (8B CPT checkpoint). cpt_dense uses full KV on that checkpoint; dense uses the original Base."
    echo "eval.sh environment overrides remain available, for example LIMIT=2 for a smoke run."
}

if [ "$#" -gt 1 ]; then
    usage >&2
    exit 2
fi
EVAL8B_METHOD=${1:-word_first}
case "${EVAL8B_METHOD}" in
    word_first|baseline|query|word_first_query|cpt_dense|dense) ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

EVAL8B_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EVAL8B_CONFIG="${EVAL8B_ROOT}/exp/qwen8b-32k/eval_full_${EVAL8B_METHOD}.yaml"
cd "${EVAL8B_ROOT}"
echo "8B full evaluation: ${EVAL8B_METHOD} | ${EVAL8B_CONFIG}"
exec bash "${EVAL8B_ROOT}/eval.sh" "${EVAL8B_CONFIG}"
