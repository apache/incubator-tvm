---
name: tir-test
description: Run TVM's in-tree TIRx tests and tirx-kernels import and correctness checks.
---

Run the compiler suite in `3rdparty/tvm-tirx/tests/python/tirx/`, using the `tvm_tirx` package
in `3rdparty/tvm-tirx/python/`, together with the downstream correctness test in
`tests/python/tirx/test_tirx_kernels_registry_correctness.py`.

Both tirx-kernels import gates and its correctness coverage are enabled by
default. Use `TIR_TEST_WITH_KERNELS=0` only for an explicitly requested
compiler-only run, and report that reduced coverage.

## Steps

Run all commands below in the same Bash shell from the TVM repository root.

1. Point Python at this worktree and select every Blackwell GPU.
   `TIR_TEST_GPUS` may explicitly provide physical GPU IDs, and
   `TIR_TEST_NUM_GPUS` may optionally cap the automatically selected count. Otherwise,
   selection uses all Blackwell GPUs, ordered by least memory in use.
   ```bash
   export TIR_TEST_TVM_ROOT="$(pwd -P)"
   export TIR_TEST_NUM_GPUS="${TIR_TEST_NUM_GPUS:-}"
   export TIR_TEST_XDIST_WORKERS="${TIR_TEST_XDIST_WORKERS:-16}"
   export TIR_TEST_WITH_KERNELS="${TIR_TEST_WITH_KERNELS:-1}"

   [[ -d "$TIR_TEST_TVM_ROOT/3rdparty/tvm-tirx/python/tvm_tirx" &&
      -d "$TIR_TEST_TVM_ROOT/3rdparty/tvm-tirx/tests/python/tirx" ]] || {
     echo "run from the TVM repository root containing 3rdparty/tvm-tirx/" >&2
     exit 2
   }
   [[ "$TIR_TEST_WITH_KERNELS" == 0 || "$TIR_TEST_WITH_KERNELS" == 1 ]] || {
     echo "TIR_TEST_WITH_KERNELS must be 0 or 1" >&2
     exit 2
   }

   [[ -z "$TIR_TEST_NUM_GPUS" || "$TIR_TEST_NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || {
     echo "TIR_TEST_NUM_GPUS must be empty or a positive integer" >&2
     exit 2
   }
   [[ "$TIR_TEST_XDIST_WORKERS" =~ ^[1-9][0-9]*$ ]] || {
     echo "TIR_TEST_XDIST_WORKERS must be a positive integer" >&2
     exit 2
   }

   if [[ -n "${TIR_TEST_GPUS:-}" ]]; then
     IFS=, read -r -a TIR_TEST_SELECTED_GPUS <<< "${TIR_TEST_GPUS// /}"
   else
     mapfile -t TIR_TEST_SELECTED_GPUS < <(
       nvidia-smi \
         --query-gpu=index,memory.used,compute_cap \
         --format=csv,noheader,nounits \
       | awk -F, '
           {
             for (i = 1; i <= 3; ++i) {
               gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)
             }
             if (($3 + 0) >= 10) {
               print $1, $2
             }
           }' \
       | sort -k2,2n -k1,1n \
       | awk '{print $1}'
     )
     if [[ -n "$TIR_TEST_NUM_GPUS" ]] && \
        (( ${#TIR_TEST_SELECTED_GPUS[@]} > TIR_TEST_NUM_GPUS )); then
       TIR_TEST_SELECTED_GPUS=(
         "${TIR_TEST_SELECTED_GPUS[@]:0:TIR_TEST_NUM_GPUS}"
       )
     fi
   fi

   (( ${#TIR_TEST_SELECTED_GPUS[@]} > 0 )) || {
     echo "no CUDA compute capability >= 10 GPU is available" >&2
     exit 1
   }
   for gpu in "${TIR_TEST_SELECTED_GPUS[@]}"; do
     [[ "$gpu" =~ ^[0-9]+$ ]] || {
       echo "invalid physical GPU ID: $gpu" >&2
       exit 2
     }
   done
   if [[ -z "${TIR_TEST_GPUS:-}" && -n "$TIR_TEST_NUM_GPUS" ]] && \
      (( ${#TIR_TEST_SELECTED_GPUS[@]} < TIR_TEST_NUM_GPUS )); then
     echo "warning: requested $TIR_TEST_NUM_GPUS GPUs, using ${#TIR_TEST_SELECTED_GPUS[@]}"
   fi

   export CUDA_VISIBLE_DEVICES
   CUDA_VISIBLE_DEVICES="$(IFS=,; printf '%s' "${TIR_TEST_SELECTED_GPUS[*]}")"
   # Preserve reference packages and other paths supplied by branch-shell.
   export PYTHONPATH="${TIR_TEST_TVM_ROOT}/.local/python:${TIR_TEST_TVM_ROOT}/3rdparty/tvm-tirx/python:${TIR_TEST_TVM_ROOT}/python:${PYTHONPATH:-}"
   export TVM_LIBRARY_PATH="${TIR_TEST_TVM_ROOT}/build/lib"
   export TVM_TIRX_LIBRARY_PATH="${TIR_TEST_TVM_ROOT}/build/lib"

   PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"
   [[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || {
     echo "set PYTHON_BIN to the test environment's Python executable" >&2
     exit 1
   }
   export PATH="$(dirname "$PYTHON_BIN"):$PATH"
   "$PYTHON_BIN" - <<'PY'
   import os
   from pathlib import Path

   import pytest
   import torch
   import tvm_ffi
   import tvm_tirx
   import xdist
   from tvm_tirx.base import _LOADED_LIBS, _RUNTIME_ONLY

   root = Path(os.environ["TIR_TEST_TVM_ROOT"])
   assert Path(tvm_tirx.__file__).resolve().is_relative_to(root / "3rdparty/tvm-tirx/python"), (
       f"imported another TIRx worktree: {tvm_tirx.__file__}"
   )
   assert not _RUNTIME_ONLY, "build the TIRx compiler before running its tests"
   for name, library in _LOADED_LIBS.items():
       assert Path(library._name).resolve().is_relative_to(root / "build/lib"), (
           f"loaded another worktree's {name}: {library._name}"
       )
   print(f"tvm_tirx: {tvm_tirx.__file__}")
   print(f"tvm_ffi: {tvm_ffi.__file__}")
   PY
   if [[ $? -ne 0 ]]; then
     echo "PYTHON_BIN must provide pytest, xdist, torch and this worktree's TIRx build" >&2
     exit 1
   fi
   if ! command -v ninja >/dev/null; then
     NINJA_BIN_DIR="$("$PYTHON_BIN" -c 'import ninja; print(ninja.BIN_DIR)' 2>/dev/null || true)"
     if [[ -n "$NINJA_BIN_DIR" && -x "$NINJA_BIN_DIR/ninja" ]]; then
       export PATH="$NINJA_BIN_DIR:$PATH"
     fi
   fi
   command -v ninja >/dev/null || {
     echo "ninja is required by kernel JIT setup but is not on PATH" >&2
     exit 1
   }

   echo "selected physical GPUs: $CUDA_VISIBLE_DEVICES"
   for logical_id in "${!TIR_TEST_SELECTED_GPUS[@]}"; do
     echo "  cuda:$logical_id -> physical GPU ${TIR_TEST_SELECTED_GPUS[$logical_id]}"
   done
   ```

   Automatic selection uses every Blackwell device. Set
   `TIR_TEST_NUM_GPUS=4` to cap the count or `TIR_TEST_GPUS=3,4,5,6` to override
   selection with exact physical IDs.

   Use dependencies compatible with `3rdparty/tvm-tirx/pyproject.toml`. A non-editable FFI
   installation in `.local/python/` takes priority over the inherited environment.
   If native libraries are missing or stale, build this worktree with the same
   `PYTHONPATH` before testing.

2. Run downstream import gates and select the full test coverage:
   ```bash
   TIR_TEST_PATHS=(3rdparty/tvm-tirx/tests/python/tirx/)
   if [[ "$TIR_TEST_WITH_KERNELS" == 1 ]]; then
     TIR_TEST_KERNELS_ROOT="${TIR_TEST_KERNELS_ROOT:-$(dirname "$TIR_TEST_TVM_ROOT")/tirx-kernels}"
     [[ -d "$TIR_TEST_KERNELS_ROOT/tirx_kernels" ]] || {
       echo "set TIR_TEST_KERNELS_ROOT to the tirx-kernels repository" >&2
       exit 2
     }
     export PYTHONPATH="${TIR_TEST_KERNELS_ROOT}:${PYTHONPATH:-}"
     "$PYTHON_BIN" -m tirx_kernels.bench_suite --check-imports || exit 1
     "$PYTHON_BIN" -m tirx_kernels.registry --strict || exit 1
     TIR_TEST_PATHS+=(tests/python/tirx/test_tirx_kernels_registry_correctness.py)
     export TIRX_KERNEL_TEST_MIN_FREE_GB="${TIRX_KERNEL_TEST_MIN_FREE_GB:-0}"
   else
     echo "compiler-only run: tirx-kernels import and correctness checks disabled"
   fi
   ```

3. Run the selected test suites with xdist parallelism and save the full log:
   ```bash
   export TIR_TEST_LOG="${TIR_TEST_LOG:-/tmp/tir-test-$(date +%Y%m%d-%H%M%S).log}"
   set -o pipefail
   "$PYTHON_BIN" -m pytest "${TIR_TEST_PATHS[@]}" -n "$TIR_TEST_XDIST_WORKERS" 2>&1 | tee "$TIR_TEST_LOG"
   ```

   `3rdparty/tvm-tirx/tests/python/tirx/conftest.py` uses `tvm_tirx.testing._pytest` to assign
   each xdist worker's PyTorch current device round-robin across the visible
   GPUs. Tests that explicitly use `tvm_tirx.cuda(0)` still run on the first
   selected physical GPU.

   Registry correctness tests take a per-device lock to serialize large workloads
   on each GPU. MegaMoE remains skipped because it requires its dedicated
   multi-process scheduler.

4. Report results: selected physical GPUs, elapsed time, total passed, failed,
   skipped, and errors, plus the full log path. Report both downstream gates as
   passed, failed, or explicitly disabled; disabled integration checks are not
   passes.

## Failure triage rules

**CRITICAL: Never pipe test output to `tail` or `grep` when diagnosing failures.
Always capture and read full logs.**

Classify every failure into one of these categories:

- **A - Environment/import error**: Module not found, missing dependency,
  collection error. Check the source and native-library paths, FFI dependency,
  and old `tvm` versus `tvm_tirx` namespaces before assigning a cause.
- **B - Real kernel correctness regression**: Assertion failures (`cosine_sim`,
  numerical diff), `CUDA: unspecified launch failure`, or wrong results. These
  MUST be investigated and fixed if caused by current changes.
- **C - Secondary xdist crash**: `KeyError: <WorkerController gwXX>` after a
  worker abort. The KeyError itself is noise; find the underlying cause (usually
  category B in another worker).

Never dismiss a failure as pre-existing without evidence. If a test fails:

1. Check whether the test touches code you changed.
2. If unclear, verify on the parent commit before claiming it is pre-existing.
3. Fix every failure caused by current changes; do not defer it.
