# Single-pair preview follow-through

This addendum follows the user's approved all-class pipeline work and subsequent explicit instructions: two images alone, automatic common-region registration, natural filename row fallback for folders, clear local paired crops, score filters, and a Streamlit review page. Execute inline in the existing isolated branch; keep the frozen reproduction pipeline unchanged.

- [x] Reproduce small-footprint excessive zoom and metadata-only input failures. Add regression tests before changing these behaviors.
- [x] Load bare PNG inputs in honest pixel coordinates, default temporal labels 1/2 when years are absent, and naturally ordered contiguous left-to-right rows when CSV placement is absent. Malformed or explicitly missing CSVs fail. Reuse SIFT/RANSAC without invented CRS or ground resolution; preserve exact alignment between stages.
- [x] Sample native original RGB once for change crops. Keep identical before/after bounds, at least 12 original pixels of context, and at most 2x source enlargement. Preserve unavailable coverage as checkerboard.
- [x] Provide reusable filtered event rows and compact grid PNG export. Never substitute segmentation confidence for a GNN probability. Preserve every canonical event in scientific output.
- [x] Add a local Streamlit wrapper: two uploads, one native full-run command, progress log, segmentation/HTML class legend and counts, graph/edge legend, matching images, paged local crops, filters and eligible-example review/training. Use existing pipeline stages and checkpoints.
- [x] Complete an actual metadata-free run on the supplied 2013/2023 tile_00002 pair. Baseline image_pair_supervised completed segmentation, graphs, registration and change reports in one native command:284 events,63 candidates,221 season-sensitive,0 strong. External-weight inspection remains below. One pair is never independent accuracy.
- [x] Run targeted tests, full regression suite, Streamlit smoke/visual checks, and document exact commands, model support, remaining limitations and review rulings.

User-authorized extension: prioritize major construction/demolition, roads and land/vegetation changes over outline/size variation; use suitable public labeled opticalRGB examples directly without another approval. User confirmed different satellites/views, not SAR.

- [x] Conservative major-change focus/summary PNG; preserve all raw candidates.
- [x] RED-to-GREEN fixes for different-resolution registration, crop tile ownership and cautious prediction captions.
- [x] Primary-source dataset research, exact class mapping, license, official parent split;44-pair bounded LsSCD-Ex download verified by CRC/SHA.
- [x] Annotation-supervised existing-RGNN training method with full segmentation/graphs, training-only normalization, separate parent validation/test, ambiguous size changes masked, unavailable unsupported heads; independent focused review approved.
- [x] Complete real annotation training and held-out evaluation on 104 pairs; add reviewed all-class false-object negatives. Record failures honestly: four directions pass validation, with substantial held-out limitations. This is not professional target reliability.
- [x] Reuse compatible checkpoint on the supplied one-pair run, inspect native crop grids and publish source/metrics/license with supported weights.

Final evidence: fresh bare-image native run completed exit0 at21:35:28; all science/render manifests verified,284raw candidates/24pages,25scored node/tasks/0positives/0major changes. Final604tests+114subtests passed; real-run AppTest five tabs,all8classes,score/classfilters,noexceptions,external fixed-weight train button disabled. All-class professional reliability remains unachieved and explicitly documented. Earlier provisional models/tests in this history are superseded, not silently presented as current.
