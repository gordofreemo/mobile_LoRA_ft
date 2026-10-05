# Vendored mlx-swift (local SPM override)

Copied 2026-08-06 from `ios/mlx-swift-examples/build/SourcePackages/checkouts/mlx-swift/`
(minus `.git` and `.build`) so MLX's C++ backend can be patched locally.

## Upstream provenance

| repo | commit | subject |
|---|---|---|
| `ml-explore/mlx-swift` | `dc43e62d7055353c7f99fa071a4e71d29dfddc44` | Expose nuclear norm order in linalg API (#411) |
| `ml-explore/mlx` (C++, at `Source/Cmlx/mlx/`) | `ce45c52505c8158ea48d2a54e8caae05efd86bfe` | [CUDA] Use qmv kernel for fp quantizations (#3239) |

Git history was dropped in the copy; these SHAs are the only record of the base.

## ⚠ The directory name is load-bearing

It **must** stay exactly `mlx-swift`. SPM resolves local overrides by package
*identity*: the last URL component for remote packages, the **directory
basename** for local ones. `ios/mlx-swift-lm-local/Package.swift` depends on
mlx-swift **by URL** (`https://github.com/ml-explore/mlx-swift`), so identity
`mlx-swift` is what it looks for.

Rename this directory to anything else (`mlx-swift-local`, etc.) and SPM stops
matching, silently fetches the **unpatched remote mlx-swift** for mlx-swift-lm to
build against, and — since mlx-swift-lm contains SmolLM3 — training runs entirely
on unpatched MLX. No error, no symptom, just a null result.

After any dependency change, confirm with:

```
xcodebuild -resolvePackageDependencies   # must NOT fetch mlx-swift
```

## Local modifications

See `LOCAL_PATCHES.md`.

## Bumping upstream

Re-copy a fresh checkout (minus `.git`/`.build`) over this directory and re-apply
the patches listed in `LOCAL_PATCHES.md`. Same pattern as `mlx-swift-lm-local`.
