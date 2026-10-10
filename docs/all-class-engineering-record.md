# Engineering decisions and verification record

This records the decisions taken during the all-class native pipeline work, including superseded experiments. The user approved native execution, suitable public training downloads, all-class graph reasoning and local integration into main. The frozen study remains separate.

## Decisions in order

1. Use native Python/PowerShell equivalents for task-start/task-done/workspace because Git Bash fails to create its signal pipe (Win32 error 5) in this sandbox. Preserve briefs, base SHAs, command logs, and completion gates ? cost if wrong: bookkeeping format differs, not product behavior.

2. add load_annual_frame adapter to reuse owned comparison sampling for annual graph PNGs before temporal change exists — avoids duplicate coordinate logic — cost if wrong: one small wrapper API change. Tiny-caption visual defect reproduced in a failing test and fixed with independent caption cell widths/zoom while preserving aspect.

3. reuse the existing phase-guided image registration phase for mapped images, prioritize substantive coverage across source tiles, and expose an explicit registration residual tolerance while preserving the 2-unit default. Raw mutual SIFT could not align the real pair and weak cross-tile pair selection impaired strip fits. Cost if wrong: automatic landmark selections differ from the former general matcher; fitted residuals and all controls are retained and upstream fingerprints bind the new profile.

4. start the one read-only whole-branch review on committed implementation while full-suite and final image rendering finish; require the same reviewer to incorporate completion evidence before its verdict. This reduces waiting without lowering the completion gate. Cost if wrong: reviewer could miss late artifacts, so send their exact paths and checks before accepting the verdict.

5. Honor no-extra-data requirement with honest pixel units and automatic matching. Physical measurements optional. Cost if wrong: numeric pixel thresholds cannot be interpreted as metres; outputs/docs state units.

6. Sample native original RGB once using scientific tile ownership, cap enlargement2x. Cost if wrong: intrinsically blurred sources cannot become sharp, and small footprints need surrounding context.

7. Major-change display filter remains separate from scientific catalog. Exclude thin edges, small building outline/size variations and vegetation subclass swaps by default; retain all candidates. Cost if wrong: real small changes may be absent from focus; detailed review preserves them.

8. Use LsSCD-Ex because all8classes align, preserving official parent train/test split and academic/noncommercial attribution. Cost if wrong: source0.6m optical imagery differs from user's imagery; no target-image accuracy claim. Bounded subset may leave heads unsupported.

9. External targets come from annotations, not noisy model change guesses. Require current-class annotation agreement/coverage; mask ambiguous size variations. Cost if wrong: model segmentation errors reduce supported supervision; never fill missing classes with invented labels.

10. Supported external inference can score a known class with shared coverage/current confidence despite uncertain counterpart segmentation. This separate checkpoint policy does not relax pseudo training/review. Cost if wrong: domain transfer can fail; outputs stay predictions with cautious captions.

11. Bounded downloads verify official HTTP ranges, member CRC and local SHA. Do not claim full archive MD5 verification for a subset. Interrupted full archive must be completed and checked before full-archive use. Cost if wrong: corrupted data must fail before training.

12. select checkpoints by validation macro average precision, with BCE tie-breaking, because loss alone favored unchanged examples in this imbalanced problem. Keep one shared checkpoint, not a different encoder per head. Cost if wrong: validation ranking can still overfit small support; disclose support and held-out results.

13. compare both full RGNN modes to separately trained node-only and spatial-only controls on identical splits. Only full graph models can be selected. Cost if wrong: a single-seed comparison is limited evidence, not proof of universal graph superiority.

14. no deployment weights are written for edge-ablated diagnostic models. The reviewer found replaying such weights with ordinary inference would restore missing edges and invalidate the diagnostic metrics. Keep their metrics only. Cost if wrong: diagnostics need retraining to replay, preferable to silently using a different graph.

15. investigate a building-only supervised loss with full graph context because the user's priority new-building task remains unsupported in the shared-task model. Keep test outside selection and do not silently replace the all-class checkpoint. Cost if wrong: small validation support can favor a specialist that fails transfer; publish measured support and experimental status.

16. supplement the selected whole all-class baseline with complete-RGNN task experts, superseding the single-encoder restriction for the final package. The all-class comparison still selects one whole base checkpoint. The user's all-type priority and class-level validation support justify routing supported directions to the best per-class validation-AP model. Cost if wrong: extra inference cost and selection variance on small validation classes; provenance and exact owners remain explicit, tests guarantee no unrelated head changes, test data never selects experts.

17. optionally include annotation-confirmed false current-class predicted regions as negatives for every class and both directions, while excluding unknown or mixed reference regions. At least 95% observed footprint coverage and known reference coverage, and at most 5% current-class agreement, are required. Why: the earlier >=80% class-agreement gate masked these false objects during training. Cost if wrong: mostly wrong regions can include small real objects; these whole-region targets still do not recover missing objects, and the broader evaluation population is not directly comparable to the earlier conditional scores.

The one-encoder final-package restriction was superseded by the explicit complete-RGNN expert package. The original annotation-agreement gate remains the default training behavior; the current external experiment enables the reviewed false-class-negative extension. Earlier 11-head metrics are a rejected conditional baseline, not current target performance.

18. Recompute the supplied bare-image pair in a fresh output after correcting segmentation-model dependency hashing. The older fingerprint included nested GNN weights and cannot be safely reinterpreted. Cost: another native segmentation/graph pass; no sealed-context edits or integrity bypass.

19. Preserve the development worktree and local training assets after the authorized local merge. It contains the verified full dataset archive and reproducibility scripts, and shares model/tool directories through junctions. Cost: extra disk use; avoids deleting unique assets or shared checkpoints during cleanup.

## Remaining preview polish

- The full combined sheet lacks an original-image overview and a count-by-type/status header. The page supplies class counts and other stage views.
- The GNN prediction raster marks positives but does not separately mark eligible negatives and ineligible regions. Tables retain eligibility/status information.
- Unavailable external head reasons currently share a generic training/validation-support explanation; exact counts and availability gates are retained. A task can have examples but still fail the validation precision requirement.

## Scope and quality limits

Only the supplied one-pair execution is validated here; universal folder performance and every satellite/view are not established. Native PNG inspection and Streamlit AppTest are available; an interactive browser surface was unavailable. Source blur remains source blur. No labeled user-image ground truth exists, so target accuracy cannot be measured. A graph cannot recover a real object that segmentation never supplied as a node. External data/derived weights retain academic/noncommercial license provenance.

The new graph comparison trains all eight classes and 16 directions with false-class negatives. Only four directions pass validation; held-out failures and insufficient support prevent a claim of professional reliability. See the current-model table in [the guide](all-class-change-results.md).


## Final verification, 2026-10-10

- Full regression: 604 passed, 114 subtests passed, one existing Rasterio georeferencing warning, 860.18 seconds. Log: artifacts/all_class_validation/final-all-class-suite.log.
- Nested checkpoint dependency regressions: original two failures plus alias failure reproduced, then nine pipeline-state tests passed.
- Native bare-image full run on verified 2013/2023 tile_00002 copies: all three science and rendering stages complete, exit 0. Segmentation/graph/change manifests verified. Before/after graphs contain 392/364 region nodes. Buildings are present (139/168 regions), rather than the earlier synthetic zero-building example.
- Revised model: 25 scored grass-gain node/tasks, zero accepted positives. No strong major changes pass. All 284 image-evidence candidates remain across 24 readable paired-grid pages plus one complete combined PNG. None are claimed confirmed physical changes.
- PNG inspection: after segmentation/class legend, a local candidate grid page and the honest empty major-change view. Complete matching/graph PNGs retained.
- Streamlit AppTest: five tabs, all eight classes, 284 weak/seasonal candidates visible, class and GNN-score filters working, fixed-checkpoint train button disabled, zero exceptions. Interactive-browser surface unavailable, so actual browser visual QA is not claimed.
- Frozen pipeline files unchanged. The original pipeline remains available.

Quality is not complete: graph contribution is measured on a small external selected subset; all-class professional target-image reliability is not achieved. Current model/test support and limits are explicit in the guide and model metadata.
