---
name: tirx-ptx-dialect
description: Register, extend, or audit instructions in the table-driven T.ptx dialect (python/tvm/backend/cuda/ptx/table.py), and move the table to a newer PTX ISA version. Use when adding a PTX instruction or qualifier, widening an operand domain, fixing a ptxas certification failure, or checking the table's comments against the PTX ISA document.
---

# TIRx PTX dialect (`T.ptx`)

`T.ptx.<mnemonic>.<qualifiers>(operands...)` is not hand-written per
instruction. One data table describes every instruction family; one engine
resolves calls, renders an inline-asm helper, and registers the TVM op; three
generators derive the IDE stub, a coverage table and a helper dump from the
same table. Adding an instruction means adding **data**, then proving it
against ptxas.

| file | role |
|---|---|
| `python/tvm/backend/cuda/ptx/table.py` | the table: `InstructionEntry` / `ModifierSlot` / `OperandSlot`, check functions, variant enumeration. tvm-free. **Read its three dataclass docstrings before editing.** |
| `python/tvm/backend/cuda/ptx/render.py` | renders one variant to a `__forceinline__ __device__` helper (`asm` or `asm volatile`, according to `asm_volatile`). `CBinding` = how each TVM dtype binds a register; `BRIDGE` = three scoped cases covering the two register classes inline asm cannot bind directly (`.pred`, and the block-local `.b8` registers behind the `e2m1x2` and `st_async_b8reg` tokens) — not a general `.b8` mechanism. tvm-free. |
| `python/tvm/backend/cuda/ptx/engine.py` | `register_table()` makes each entry a TVM op `tirx.ptx.<name>` with a generic codegen; `PTXNamespace` resolves `T.ptx` attribute chains and the string form `T.ptx["..."]`; trace-time coercion and `check`/`imm_check`. |
| `python/tvm/backend/cuda/ptx/gen_stubs.py` | `python -m tvm.backend.cuda.ptx.gen_stubs -o python/tvm/script/tirx.pyi` — the checked-in stub; a test diffs it against the generator. |
| `python/tvm/backend/cuda/ptx/gen_helpers.py` | `python -m tvm.backend.cuda.ptx.gen_helpers <family>` — dump the exact helper source of every variant without compiling a kernel (it still imports `tvm`, so it needs the usual built worktree on `PYTHONPATH`). |
| `python/tvm/backend/cuda/ptx/gen_coverage.py` | markdown coverage table of the whole dialect. |
| `tests/python/tirx/codegen/test_ptx_dialect.py` | per-ISA-section dispatch tests, structural invariants, ptxas certification. |
| `tests/python/tirx/codegen/test_ptx_cvt.py` | one case per registered `cvt` syntax line (`_FORM_CASES`); coverage of every cvt entry is asserted. |
| `tests/python/tirx/codegen/test_ptx_addr.py` | `T.ptx.addr(base, byte_offset)` immediates. |

The namespace itself is installed by `python/tvm/backend/cuda/__init__.py::script_namespaces()`;
nothing there changes when the table grows.

> **DO NOT introduce a new mechanism in an `InstructionEntry` without the
> user's explicit approval.** Stop, explain why the existing entry model is
> insufficient and what codegen or validation behavior the mechanism would
> add, then wait for approval before implementing it.

## 1. Ground rules the table enforces

The first four are tested mechanically — a new entry that breaks one fails
the suite. The last one (what the comments claim) is audited by hand, §4.

- **One entry = one syntax shape.** Split into sibling entries sharing the
  `mnemonic` when the register-group shape or the result structure changes
  (scalar vs vector destination, `mul` vs `mul.wide`), or when an optional
  operand is told apart only by call arity (`{, count}`, `state` vs the `_`
  sink); the engine then selects by arity and by which tokens are written.
  An operand whose presence is fixed by one written qualifier stays in the
  entry with a 0-or-1-register `lanes` function — `{, cache_policy}` with
  `.L2::cache_hint`, `{, ctaMask}` with `.multicast::cluster` use
  `lanes=_present_lanes("<slot>")`. Variants that only add or remove dotted
  qualifiers are optional slots of one entry.
- **Single-instruction invariant**: every helper emits exactly one native PTX
  instruction. The only extra statements allowed are the framework `@p`
  wrapper and the `BRIDGE` boundary conversions. `raw_render` is the escape
  hatch and every user of it must be listed in `RAW_ENTRIES` inside
  `test_ptx_single_instruction_invariant`.
- **Every legal closed variant assembles**: `renderings()` enumerates the
  product of modifiers × operand dtypes × closed immediates × `@p` (for
  entries without a destination) and full certification assembles all of it
  through ptxas at `cert_arch`. Four things get *representative* coverage
  instead of the full product, and each says so in its docstring: an OPEN
  immediate is certified at `imm_combos`'s samples (default `"0"`), which
  proves the shape, not the caller's constant; sink masks are walked once at
  one representative modifier combination per sink domain (`renderings`);
  address immediates are a small separate axis (`_addr_offset_samples`);
  and the destination-predicated `pred=` helpers (`*_pred_undef` /
  `*_pred_keep`) are not enumerated at all — cover those with a
  production-shaped test when a call site needs them. Conversely a `check()`
  must never reject a legal form to make certification pass — narrow with
  evidence.
- **Dispatch must be unambiguous**: no two entries may accept the same call
  (`test_ptx_dispatch_unambiguous`); helper names must be unique
  (`test_ptx_all_variants_render_unique`).
- **Comments cite the ISA**: the section number, the syntax line reproduced
  verbatim, the type/qualifier domains, `sm_XX` floors, and a `NOT REGISTERED:`
  note with the reason for every syntax line or token deliberately left out.
  A fact that comes from ptxas rather than the document is marked `MEASURED`
  and quotes the ptxas message. Section and table numbers follow the ISA
  version named in the module docstring (currently **PTX ISA 9.4**, the CUDA
  13.4 developer-preview document, see §5).

## 2. Designing the entry

Open the instruction's section in the ISA document and transcribe, in this
order:

1. **Name / mnemonic / family.** `name` is the table key and must be a Python
   identifier. Dotted mnemonics are a family plus single-choice slots
   (`cvta.to.shared` → `cvta` + slots), or `mnemonic="st.bulk"` with
   `name="st_bulk"` when the dot is part of the instruction's identity.
   Several entries sharing a mnemonic (`mov`, `mbarrier`, `mma`) use
   distinct `name`s (`mov`, `mov_pack_2`, ...) and the same `mnemonic`.
   Entries that collide with an existing family on the same shape need a
   suffix (`add_int` vs `add`/`add_half`) and are told apart by their tokens.
2. **Slots** (`ModifierSlot(name, choices, optional)`), in asm render order —
   exactly the `{.qual}` positions of the syntax line, each with the ISA's
   `.qual = { ... }` list as `choices`. Tokens with `::` are written as-is
   (`"shared::cta"`); users type `shared__cta`. Python keywords get a
   trailing underscore only at the call site (`global_`).
3. **Operands** (`OperandSlot`), in PTX operand order, destinations first
   because PTX writes them first:
   - `rw`: `"w"` destination, `"rw"` accumulator (`"+"`), `"r"` input. A
     `"w"` operand binds `"="`; it does **not** block `pred=` — under a false
     predicate the value is undefined unless the caller passes
     `preserve_dst=True`, which switches the binding to `"+"` (see
     `test_ptx_predicated_destination_*`). Those two helpers are rendered on
     demand, not by `renderings()`.
   - `kind`: `"reg"` (default), `"addr"` (`[%k]`, space from `space=` or the
     entry's `space` slot; set `allow_imm_offset=True` only for an independent
     byte address), `"ptr"` (raw pointer value), `"imm"` (text operand:
     `literal=` fixed by the ISA, `choices=` closed caller set, neither = open
     immediate certified at samples; validate open ones with `imm_check`).
   - `dtype`: a `PTX_TYPE_DTYPES` key, a slot name, or a module-level pure
     function of the modifier map (`_wide_dtype`). `None` means the entry's
     `type` slot. `dtypes=` narrows/widens the TVM dtype domain independently
     (see the relaxed-carrier note in §6 before widening anything).
   - `lanes` (int or function of the modifier map) for brace-enclosed register
     groups; `vector=`, `bracket=`, `pipe=` for the few composite spellings;
     `sinkable=` only where the ISA lets a caller write `_` for a lane.
4. **`check`**: one pure module-level function `mod_map -> error | None` per
   entry with a one-line docstring (it is surfaced by the stub and the coverage
   table). Encode only what the ISA syntax block and notes state; every
   restriction that comes from ptxas instead is a separate `MEASURED` clause.
   Never a lambda: frozen dataclasses hash callables by identity.
5. **`cert_arch`**: the **maximum** `sm_XX` floor over the entry's variants
   (default is `PTX_ARCH`, `sm_90`). Certifying below a variant's floor makes
   ptxas report legal forms as illegal, and that verdict would then get baked
   into a `check()`.
6. `orders_memory=True` for fences/barriers/waits (no address operand but the
   `"memory"` clobber is needed); `asm_volatile` stays at its default.

Worked example — the whole registration of `movmatrix` (commit a29a5e97ba,
4 files, 40 lines):

```python
    # movmatrix per PTX ISA 9.7.16.5.17 -- transpose one distributed m8n8
    # matrix whose 16-bit elements are carried by one b32 register per lane.
    #
    #   movmatrix.sync.aligned.m8n8.trans.b16 d, a;
    InstructionEntry(
        name="movmatrix",
        slots=(
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
            ModifierSlot("shape", ("m8n8",)),
            ModifierSlot("trans", ("trans",)),
            ModifierSlot("type", ("b16",)),
        ),
        cert_arch="sm_75",
        operands=(
            OperandSlot("d", rw="w", dtype="b32"),
            OperandSlot("a", dtype="b32"),
        ),
    ),
```

Put the entry under its ISA chapter banner in `_ENTRIES` (`# PTX ISA 9.7.x —
...`), next to the instructions it shares a mnemonic with.

## 3. Verify — every gate, in this order

Run from the TVM repo root with the workspace `PYTHONPATH` (see `tir-test`).

1. **Table loads and enumerates.** A `check()` that filters out every
   combination, a duplicate name, or a bad `allow_imm_offset` slot fails at
   import.
   ```bash
   python -c "from tvm.backend.cuda.ptx.table import TABLE, variants; e=TABLE['<name>']; print(len(variants(e)))"
   python -m tvm.backend.cuda.ptx.gen_helpers <name> | head -60   # eyeball the asm
   ```
2. **Add tests for the new surface** (before running the suite, so one run
   settles everything):
   - a call in the dispatch test of its ISA section
     (`test_ptx_<section>_dispatch`) asserting the exact emitted line, e.g.
     `assert "mul.wide.s32 %0, %1, %2;" in src`;
   - a new `cvt` line → a `_FORM_CASES` row in `test_ptx_cvt.py`
     (`test_cvt_cases_cover_every_registered_entry` fails otherwise);
   - `raw_render` → `RAW_ENTRIES`;
   - a destination-predicated (`pred=`) call site → a production-shaped
     codegen test, since certification does not enumerate those helpers;
   - only when the instruction belongs to the MegaMoE extracted-intrinsics
     surface → its kernel and mnemonic list in
     `3rdparty/tvm-tirx/tests/python/tirx/codegen/test_cuda.py::test_megamoe_extracted_intrinsics_codegen`
     (that test covers one workload, not the dialect).
3. **Regenerate the stub** — any change to a family's tokens or shapes makes
   `test_ptx_stub_up_to_date` fail until you do:
   ```bash
   python -m tvm.backend.cuda.ptx.gen_stubs -o python/tvm/script/tirx.pyi
   ```
4. **Fast suite** (no `PTX_CERT`): structural invariants + dispatch tests +
   a seeded ptxas sample.
   ```bash
   python -m pytest tests/python/tirx/codegen/test_ptx_dialect.py tests/python/tirx/codegen/test_ptx_cvt.py tests/python/tirx/codegen/test_ptx_addr.py -q
   ```
   `test_ptx_all_variants_render_unique` ends with `assert total == <N>`;
   its failure message prints the new total — update the constant and
   re-run.
5. **Full certification** — mandatory after any table change, and the only
   gate that proves the domain. It assembles every closed variant of every
   entry, split into 32 shards; pick 8, 16 or 32 workers to taste (each one
   drives its own nvcc; `-n auto` on a many-core box mostly burns memory).
   ~3.5 min at `-n 16`, ~6.5 min at `-n 8` on a B200-class host for the
   ~760k-variant PTX ISA 9.4 table (`test_ptx_all_variants_render_unique`
   pins the exact count).
   ```bash
   PTX_CERT=1 python -m pytest -n 16 -q \
     tests/python/tirx/codegen/test_ptx_dialect.py::test_ptx_all_helpers_certify
   ```
   A failure names `<arch> batch <k>` and the ptxas message. To find the
   variant: render the failing family with `gen_helpers`, or assemble the
   batch yourself (`nvcc -arch=<arch> -ptx` then `ptxas`, map the error line
   back to the preceding `.entry`). Fix by narrowing the domain with a
   `MEASURED` clause, never by loosening the invariant.
6. `pre-commit run --files <changed files>` and commit as
   `feat(lower-tirx): support PTX <instruction>` (Conventional Commits, see
   the workspace `CLAUDE.md`).

## 4. Auditing the comments against the ISA

The comments are load-bearing (they are what a reviewer checks the data
against), so audit them the way the data is certified: download the ISA HTML
for the version the module docstring names, convert to text, and verify every
section number, quoted sentence, reproduced syntax line, type list and
`sm_XX` note. Things that have gone wrong before and are worth a targeted
pass: a `NOT REGISTERED` bullet naming something the table registers a few
hundred lines later; a section number copied from a neighbouring
instruction; a version number attached to the wrong syntax line; "the N-th
syntax line" ordinals; `Table NN` numbers, which are global and shift between
ISA versions.

## 5. Toolchain and ISA-version model

- The dialect targets the **ptxas of the installed CUDA toolkit**, not the
  latest ISA document. CUDA 13.4's ptxas implements PTX ISA 9.4. Establish
  what yours implements from ptxas itself — `nvcc -ptx` only shows the
  version the front end chose to emit, and `-ptx` never validates inline
  asm:
  ```bash
  command -v nvcc ptxas && nvcc --version | tail -1 && ptxas --version | tail -1
  printf '.version 9.4\n.target sm_107a\n.address_size 64\n.visible .entry k() { ret; }\n' \
    | ptxas -arch=sm_107a - -o /dev/null   # CUDA 13.4: accepted
  ```
  A `.version` directive above what ptxas implements is refused outright
  ("Unsupported .version ...; current version is '9.4'"), which is the
  quickest way to read a toolkit's ceiling. Then let the certification suite
  (which compiles real inline-asm helpers to cubin) be the final word.
- The table's citations follow that same version (module docstring of
  `table.py`: "Section and table numbers cite PTX ISA 9.4", URL
  `docs.nvidia.com/cuda/developer-preview/13.4/parallel-thread-execution/`;
  once CUDA 13.4 ships, the stable copy is
  `docs.nvidia.com/cuda/archive/13.4.0/parallel-thread-execution/`). The
  live `docs.nvidia.com/cuda/parallel-thread-execution/` page is whatever
  version is newest and its numbering differs; every ISA version stays
  available under `docs.nvidia.com/cuda/archive/<cuda version>/`.
- Every `MEASURED` clause in `table.py` / `render.py` names the toolkit it
  was taken on (currently CUDA 13.4). A toolkit bump re-measures all of them
  (§6 step 5); a clause that names an older toolkit is a bug.
- Certification needs only `nvcc`/`ptxas`; the on-GPU round-trip tests need
  a driver that can load a cubin from that toolkit. A newer toolkit with an
  older driver can therefore certify a table but not run it.
- `PTX_ARCH` (env, default `sm_90`) is the certification arch for entries
  without `cert_arch`.

## 6. Moving the table to a newer PTX ISA

Do this as one commit series, in this order:

1. **Toolchain first.** Install the CUDA toolkit whose ptxas implements the
   target ISA and confirm via `.version` (§5). Until then, a new form cannot
   be registered: certification would fail, and a `check()` written against
   an older ptxas would silently delete the coverage later.
2. **Build the section map structurally, from both TOCs.** Dump both HTML
   documents to text, extract the two tables of contents (`<num> <title>`),
   and derive a `rule(old) -> new` function from where the new version
   inserted sections; cross-check it by asserting every old `9.7.*` entry
   maps to an identically titled new entry. Never trust title matching
   alone (duplicate titles such as "mov" x2 or "Async Proxy" x2 collide) and
   never shift chapters only — a chapter-only shift was the exact mistake
   made the first time this table changed versions.
3. **Retarget the citations.** Update the module docstring's version
   statement and URL, then run the renumbering over `table.py`,
   `engine.py`, `render.py`, `3rdparty/tvm-tirx/tests/python/tirx/codegen/test_ptx_*.py` and
   `test_cuda.py`, skipping any block that already carries the new
   numbering (`_PTX_94_ENTRIES` was such a block for 9.3→9.4). Hand-fix the
   forms a regex cannot: ranges that span an insertion (`9.7.x.a-b`), brace
   groups (`9.7.4.{1,2,3}`), sibling shorthand (`.12`, `/ .17`), global
   `Table NN` numbers, and `:line` offsets (re-derive against the new
   section text from the quoted sentence, or drop the offset and keep the
   quote). Then regenerate the stub (§3).
4. **Register the newly legal forms.** Read the new version's release notes
   (ISA chapter 13.1) and the per-section "introduced in PTX ISA version
   X.Y" notes; each one: read the section, apply §2, then §3 including full
   certification at the right `cert_arch`. Keep the new version's additions
   grouped (the `_PTX_94_ENTRIES` list is the 9.4 group).
5. **Re-measure every `MEASURED` clause on the new toolkit** and reword it
   to name that toolkit. Registered forms are re-proven by full
   certification; excluded forms need a targeted probe each (raw PTX through
   `ptxas -arch=<a>` for grammar claims, an exact force-inlined kernel
   through `nvcc` for crash claims — `test_ptx_dialect._certification_kernel`
   builds that shape). Quote the diagnostic the new ptxas actually prints;
   texts change between releases. If a gap has closed, either widen the
   domain with certification or say so in the clause — never leave a
   sentence that attributes the restriction to a toolkit you no longer use.
   ISA §9.4.1 Tables 27/28 are an upper bound and say so ("some combinations
   may still be invalid for a particular instruction"); the `_cvt_dst_dtypes`
   / `_cvt_src_dtypes` docstrings record the measured gaps.
6. Regenerate the stub, update the variant/address counts the tests pin,
   run the full suite, and run the §4 audit against the new document before
   merging.

Worked example, PTX ISA 9.3 → 9.4:

- 9.4 inserted chapter `9.7.6 Alternate Floating-Point Instructions`, so
  every `9.7.N` with N ≥ 6 became `9.7.(N+1)` (comparison `9.7.6→9.7.7`,
  data movement `9.7.9→9.7.10`, fabric `9.7.10→9.7.11`, sync
  `9.7.14→9.7.15`, mma `9.7.15→9.7.16`, wgmma `9.7.16→9.7.17`, tcgen05
  `9.7.17→9.7.18`, misc `9.7.20→9.7.21`).
- Inside data movement, `applypriority.async.bulk[.tensor]` were inserted as
  `9.7.10.18/19`, so `9.7.9.18..27 → 9.7.10.20..29` (+2), and "Overriding
  tensor property value" as `9.7.10.28.5.2`, so
  `9.7.9.26.5.2..4 → 9.7.10.28.5.3..5`.
- Inside tcgen05, "Decompression of input matrices" became `9.7.18.10.8`,
  so `9.7.17.10.8.x → 9.7.18.10.9.x` and `9.7.17.10.9.x → 9.7.18.10.10.x`.
- Global tables: tcgen05.ld/st register-count tables `52/53 → 59/61`,
  tensormap `new_val` validity `33 → 36`; the relaxed-typing tables `27/28`
  kept their numbers.
- Two files still carried 9.2 numbering (`test_ptx_cvt.py`, `render.py`:
  cvt as `9.7.9.21`, which in 9.3 is cvta) — check every file's baseline
  before mapping it.
- Toolchain facts that changed on 13.4: the `.L2::cache_hint` diagnostics
  on `.shared`/`.local`/`.volatile`, the bare
  `clusterlaunchcontrol.query_cancel` diagnostic, `multimem.st.async`
  accepting 16/32-bit sources for its byte forms, and the bf16 atom
  bit-bucket forms compiling again (still withheld pending certification).
- 9.4-only sections worth registering later: `9.7.6.1-4` (alternate FP
  x4 arithmetic, sm_100a/sm_103a), `9.7.10.28.1.3` (report mechanisms),
  `9.7.18.10.7.2.6 / .3.6` (block16 K=128/256 scale layouts).
