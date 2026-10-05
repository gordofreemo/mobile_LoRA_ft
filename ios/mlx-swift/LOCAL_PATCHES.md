# Local patches to vendored mlx-swift

Base: mlx-swift `dc43e62`, MLX C++ `ce45c525`. See `VENDORED.md`.

Full result: `experiments/2026-08-06-mlx-nax-qmm-n-backward.md`.
Reproducers: launch `LLMEval` with `--verify-qmm-n` (numerical) or
`--benchmark-nax-ab` (paired timing).

## The problem

MLX gates Apple's neural-accelerator (NAX) quantized matmul on
`transpose == true`, so backward's `dX = dY·W` falls to a generic 32×32 kernel
even though `affine_qmm_n_nax` is compiled and `qmm_nax()` already handles
`transpose=false` end-to-end. h11 measured that generic kernel at ~61% of a full
on-device LoRA training iteration.

Relaxing the guard alone is **not** enough — the kernel behind it is broken. Two
independent upstream defects, both consistent with code that was never run
(it is unreachable through the public API, so upstream CI never exercised it):

1. **Weight addressing uses the transposed layout** — wrong results.
2. **No partial-M-tile handling** — out-of-bounds write.

Both are fixed here. Any sequence length is now supported.

## Patch 1 — kernel fix (the important one)

**File:** `Source/Cmlx/mlx-generated/quantized_nax.cpp`, in `qmm_n_nax_tgp_impl`.

⚠ **Edit the GENERATED file, not `mlx/backend/metal/kernels/quantized_nax.h`.**
These NAX kernels are **JIT-compiled at runtime** from a string literal in
`mlx-generated/quantized_nax.cpp`. The `kernels/*.h` header is not what ships —
`default.metallib` does not even contain `affine_qmm_n_nax`
(`strings default.metallib | grep -c affine_qmm_n_nax` → 0), and
`quantized_nax.metal` is not among the Metal sources the build compiles. Editing
the header changes nothing and the results reproduce to the last digit, which
looks exactly like a null result. The header is patched here too, for
consistency, but only the generated file has effect.

`qmm_n_nax_tgp_impl` uses the same `QuantizedBlockLoader<T, BK, BN, BN_padded,
0, ...>` instantiation as the known-correct generic `qmm_n_impl`
(`kernels/quantized.h:1266`) but sets its pointers up for `[N, K]`:

| | generic (correct) | NAX (as shipped) |
|---|---|---|
| weight offset | `wl += y_col * bytes/pack` | `wl += y_col * (K*bytes/pack)` |
| scales offset | `scales += y_col / group_size` | `scales += y_col * (K/group_size)` |
| leading dim | `loader_w(..., N, ...)` | `loader_w(..., K, ...)` |
| implied layout | `w = [K, N]` grouped along N ✓ | `w = [N, K]` grouped along K ✗ |

Copy-adapted from `qmm_t_nax`, weight addressing never converted. With the fix
the kernel is **bit-exact** against a dequantized reference (relative error
1.5–7826 → 0.0), and backward runs **~1.8x** faster (**~1.5x** per iteration).

**This also repairs the MoE path**: `affine_gather_qmm_n_nax`
(`kernels/quantized_nax.h:1408`) calls the same impl at `:1463`.

## Patch 2 — `MLX_ENABLE_NAX_N` dispatch flag

**Files:** `mlx/utils.h` (adds `env::enable_nax_n()`),
`mlx/backend/metal/quantized.cpp` (~line 694, in `qmm()`).

```cpp
if (metal::is_nax_available() &&
    (transpose || (env::enable_nax_n() && (M % 64 == 0))) &&
    (K % 64 == 0) && (env::enable_tf32() || x.dtype() != float32)) {
```

`enable_nax_n()` deliberately does **not** cache in a function-local static (every
sibling accessor does) — `get_var` is just getenv + atoi, nanoseconds against
millisecond GPU kernels, and reading it live lets the harness flip arms between
training iterations via `setenv` for a thermally-paired A/B.

### `N % 64 == 0` — and why M is no longer restricted

`N` is not clamped in the weight loader: the tile is `BK x BN` over `w = [K, N]`.
Every affine quantization with `group_size >= 64` gives `N % 64 == 0`
structurally (groups run along N), so this rejects only `group_size == 32` with
N not a multiple of 64, which falls back to the generic kernel as before.

**M is unconstrained** — see patch 3.

Default is **0 (OFF)**, so dispatch behaviour is identical to stock upstream MLX
unless a caller opts in.

## Patch 3 — M bounds handling (partial tiles)

**File:** `Source/Cmlx/mlx-generated/quantized_nax.cpp`, `qmm_n_nax_tgp_impl`.

The second upstream defect. The kernel did `(void)M`, had its
`num_els = min(BM, M - y_row)` line commented out, and called `Atile.load` /
`Dtile.store` unconditionally — so with a partial M tile it read x and **wrote y**
past the end of the buffer (~49 KB at M=250). Unaligned M still returned correct
values in the valid region during verification; that was luck, the overflow
landing in allocator slack.

Ported from `qmm_t_nax_tgp_impl`, which already had it:

```cpp
const short sgp_sm = min(SM, short(M - (y_row + tm)));   // rows this simdgroup owns
const bool is_unaligned_sm = (sgp_sm != SM);

dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
  ...
  if constexpr (kAlignedM.value) { Atile.load(x + kk1, K); }
  else { Atile.load_safe(x + kk1, K, short2(SK, sgp_sm)); }
  ...
  if constexpr (kAlignedM.value) { Dtile.store(y + tm * N + tn, N); }
  else { Dtile.store_safe(y + tm * N + tn, N, short2(SN, sgp_sm)); }
});
```

`dispatch_bool` is a compile-time split, so the aligned case keeps the original
unguarded loads and pays nothing; only one branch executes, so the `x += BK` walk
inside the loop is not duplicated. `sgp_sm` may go <= 0 when an entire simdgroup
sits past the end of the matrix — `load_safe`/`store_safe` handle that, as they
already do for the transposed kernel.

Verified across M ∈ {1, 2, 31, 32, 33, 63, 65, 96, 97, 100, 127, 129, 191, 193,
255, 257, 511, 513, 999, 1000, 1023, 1025} plus unaligned batched, all correct.

## Bumping upstream

Re-copy a fresh checkout (minus `.git`/`.build`) and re-apply both patches.
Remember patch 1 lives in `mlx-generated/`.
