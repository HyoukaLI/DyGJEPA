# DyGLib source provenance

- Upstream: `https://github.com/yule-BUAA/DyGLib`
- Commit: `3aacc36b94b8d2d8293d70a74fdf6d39089b4163`
- License: MIT; the upstream license is preserved in `LICENSE`.
- Vendored modules: `MemoryModel.py` (TGN), `CAWN.py`, `TCL.py`,
  `GraphMixer.py`, `DyGFormer.py`, and `modules.py`.
- Local changes to vendored model files: package-relative imports only.
- `sampler.py` contains the upstream `NeighborSampler` class.  Dataset
  construction and evaluation are handled by `jepa_compare/dyglib_baselines.py`
  so all methods use this project's common chronological protocol.

EdgeBank's unlimited-memory rule is implemented directly in the adapter; it is
the same deterministic observed-edge lookup used by DyGLib's Wikipedia random
negative-sampling configuration.

Original-author repositories used for architecture/default cross-checks:

- TGN: `twitter-research/tgn@d55bbe678acabb9fc3879c408fd1f2e15919667c`
- CAWN: `snap-stanford/CAW@f994ff2b2c29778e6250b6a9928fd9943e0163f7`
- GraphMixer: `CongWeilin/GraphMixer@c84f1e0bee4eed848872a966b8166d741e240713`

The executable versions remain DyGLib's unified implementations because its
paper explicitly repairs checkpoint-state and equal-timestamp leakage issues
in the older TGN/CAWN reference code while retaining their model mechanisms.
