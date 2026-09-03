# Snapshot SSL baseline provenance

These baselines share the project's chronological 80/10/10 split, directed
Wikipedia queries, destination-corrupted candidates and ranking metrics. In
all three cases the encoder is pretrained only on the training prefix, frozen,
and followed by the same `PairMLP`. Consequently these results measure frozen
representation quality and must be separated from native end-to-end temporal
link-prediction results.

## CLDG

- Paper: *CLDG: Contrastive Learning on Dynamic Graphs*, ICDE 2023.
- Author repository: <https://github.com/yimingxu24/CLDG>
- Audited commit: `bdbc1eb9b46700b227690d0fb352a47baf89e866`.
- Preserved mechanisms: temporal timespan views, two-layer normalized GCN,
  normalized two-layer projection MLP, and symmetric all-view-pair InfoNCE at
  temperature 0.07.
- Local adaptation: DGL neighbor sampling is replaced by dependency-free sparse
  PyTorch propagation; the authors did not report Wikipedia, so its event bins
  are treated as discrete timespans. The original node linear classifier is
  replaced by the comparison's shared frozen link probe.

## MaskDGNN

- Paper: *MaskDGNN: Self-Supervised Dynamic Graph Neural Networks with
  Activeness-aware Temporal Masking*, IJCAI 2025.
- Paper URL: <https://www.ijcai.org/proceedings/2025/322>
- The paper points to <https://github.com/heyimingheyiming/MaskDGNN>, but that
  repository was not publicly cloneable when checked on 2026-09-02.
- This is therefore a paper-level reimplementation, not an official-code run.
  It implements equations (2)--(8) for dynamics/PageRank activeness and edge
  masking, equations (10)--(14) for normalized GCN plus learnable complex FFT
  filtering, and equation (15)'s concatenation MLP for masked-edge recovery.
- The paper's sliding-window optimizer is not used after pretraining: the
  downstream phase intentionally uses the shared frozen probe protocol.

## DVGMAE

- Paper: *DVGMAE: Self-Supervised Dynamic Variational Graph Masked
  Autoencoder*, IEEE TNNLS 36(10), 2025, DOI 10.1109/TNNLS.2025.3583045.
- No public author implementation could be verified. This is a paper-level
  reimplementation, not an official-code run.
- Preserved high-level mechanisms: temporal-aware masking that balances current
  mask probability using historical mask exposure, a variational graph encoder,
  and a globally enhanced temporal decoder trained to recover masked structure
  and snapshot features.
- Exact numerical reproduction of the TNNLS tables is not claimed; result JSON
  records `paper_reimplementation_tnnls2025` explicitly.
