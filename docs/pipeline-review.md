# Final implementation review

The independent review examined base `204d10e` through `fa5b154` and the subsequent working fixes. It found no Critical issue. All confirmed Important issues received failing regressions and a single implementation fix pass; full-suite evidence follows in pipeline-validation.md. The first reviewer stopped at its usage limit before a final report; an available model completed the one required review. No reviewer changed the checkout.

| Finding | Resolution and regression |
|---|---|
| Selected PNG metadata/source and configured old weights were omitted from fingerprints | Discovered metadata/source and resolved architecture/checkpoint bytes are bound; `test_selected_image_fingerprint_includes_discovered_placement_and_source`, `test_old_model_config_fingerprint_binds_referenced_weights` |
| GNN outputs lacked integrity verification | GNN artifact/manifest binds predictions, report, local weights/metadata; stage result includes it; `test_validation_detects_deleted_gnn_predictions` |
| An earlier tree could receive a construction target | Positives require eligible later-year building endpoints; GNN rechecks class, observation and confidence; old-event test and `test_target_csv_cannot_train_nonbuilding_nodes` |
| Saved boundary and Stage 4/5 policy overrides were ignored | Saved configuration feeds overlap, matching and change policy; explicit boundary overrides apply; custom-policy tests |
| Tree/road raster events contradicted JSON persistent statuses | Named raster regions are exported; nonbuilding node events agree and unmatched assessments remain explicit; `test_tree_gain_raster_and_json_events_agree` |
| Same-path external GNN model replacement could reuse predictions | Change-stage fingerprint includes weight/sidecar bytes; external provenance verifies referenced files; same-path and external-checkpoint tests |
| Literal historic year names appeared in new evidence | Earlier/later evidence fields; year-independent field regression |
| Context model provenance was empty | Resolved model paths, sizes, hashes and availability now persist; custom-model provenance regression |

The last two were initially graded Minor. They were regraded as required delivery details because the approved contract promises actual-year records and resolved model provenance. No deferred minor finding remains.

The reviewer explicitly declined to establish universal model accuracy/calibration, frozen study reproduction without original inputs, CUDA execution without CUDA, independent GNN human accuracy without audits, unbounded scalability, or additional unspecified features. These are evidence boundaries, not silently inferred successful checks.

## Rulings made during execution

- Work in the approved isolated feature worktree: main stays reviewable; cost is a final integration step.
- Install pytest locally under ignored tools: avoid changing the shared environment; cost is the test wrapper.
- Use a portable Windows ledger instead of repeated Bash helpers: same brief/base/red/green/commit evidence; cost is manual bookkeeping.
- Bound per-tile and comparison tensors explicitly: avoid unbounded allocations; cost is subsets or larger RAM budgets for large inputs. Upsampling adds no new physical detail.
- Withhold metric events without physical scale: preserve truthful units; cost is correspondence-only pixel comparisons.
- Test atomic downloads with local file URLs and record old hashes as verified local assets: deterministic tests and honest publisher provenance; cost is no end-to-end remote-failure test or independent publisher checksum for old weights.
- Complete the interrupted reviewer seat with an available model: the original reviewer hit its usage limit; cost is restarting that incomplete review context.
- Regrade year names and model context provenance as required details: satisfy the approved output contract; cost is generic evidence-field names and additional hash I/O.

No full original-study reproduction or scientific accuracy score is claimed for arbitrary new inputs.
