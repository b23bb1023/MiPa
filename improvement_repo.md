# MiPa improvement plan

This plan translates the supplied 20 October 2026 repository audit into implementation work. The current `main` source was rechecked on 27 October 2026 at commit [3e63bbc11b6543bc0a01572af868aab0d8295d5d](https://github.com/b23bb1023/MiPa/commit/3e63bbc11b6543bc0a01572af868aab0d8295d5d), which still matches the audited snapshot.

**Scope:** documentation and planning only. None of the technical fixes below has been implemented by this change. Notebook JSON and code syntax were parsed, relevant source paths were inspected, and saved-result arithmetic was recalculated. No GPU run, dataset download, checkpoint loading, retraining, or benchmark validation was performed. Historical run provenance remains unresolved. The binary report and presentation were not reviewed.

## Delivery order

| Order | Audit item | Outcome required before proceeding |
| --- | --- | --- |
| 1 | MP-01 — MA schedule | Observable, tested, nonzero MA gradients with a defined scaling convention |
| 2 | MP-03 / MP-05 — scheduler and resume | Training state follows the same trajectory across uninterrupted and restored runs |
| 3 | MP-04 — evaluation split | Validation drives selection; final test data stays separate |
| 4 | MP-06 — reproducible setup | Configurable paths, tested environment, and a clean setup smoke run |
| 5 | MP-02 / MP-08 — result and resource records | Validated, traceable results and correctly scoped peak measurements |
| 6 | MP-07 — controlled study | Replicated experiments with disclosed differences and uncertainty |

Steps 2–4 can be developed separately, but all must pass before step 6. Repository packaging improvements should support these checks.

## 1. Repair the modality-adversarial schedule — MP-01, High

**Current evidence:** in both [mipa-cnn.ipynb](mipa-cnn.ipynb) and [mipa-ViT.ipynb](mipa-ViT.ipynb), code cell 3 initializes `global_step = 0` at source line 512. `get_lambda()` (523–527) reads it, with no other assignment or increment in any code cell. `forward()` applies lambda to gradient reversal and returns lambda times BCE (545–569). The resulting MA objective is zero for the committed training path.

**Implementation work**

- Specify whether schedule progress means epoch, successful optimizer update, or normalized training progress. Make the caller update it explicitly and serialize it in checkpoints.
- Decide where lambda belongs: GRL, MA objective, or both. Current double use would scale the backbone's MA contribution by lambda squared once progress advances. Implement and document the intended convention.
- Log progress, lambda, raw MA loss, weighted MA loss, and detector loss independently.
- Mark historical MA/gamma conclusions as provisional and rerun affected experiments after the training-state and split fixes.

**Acceptance checks**

- Test initial, intermediate, final, and restored progress values; lambda must match the declared schedule and become positive after its intended start.
- Backpropagate an isolated MA objective. Assert finite, nonzero classifier and unfrozen-backbone gradients, including the intended reversal sign and scale.
- Include a deliberately disabled-MA control. Check that checkpoint round-trip preserves progress and the next lambda value.

## 2. Align LR scheduling with real optimizer updates — MP-03, Medium

**Current evidence:** CNN code cell 3 computes `total_steps` and `warmup_steps` from loader minibatches (1079–1080), advances the scheduler during accumulated updates (876), and calls it again after each epoch (1133). Swin repeats these paths at 1129–1130, 926, and 1183. The configured accumulation is 2.

**Implementation work**

- Count successful optimizer updates consistently. Calculate warmup and decay lengths from accumulation windows, including partial windows and skipped empty-target batches.
- Remove the additional epoch call and advance LR only when an optimizer update actually occurs, including AMP overflow handling.
- Restore scheduler counters together with optimizer/scaler state on resume.

**Acceptance checks**

Use a tiny deterministic loader with even, odd, skipped, and partial accumulation windows. Assert a one-to-one correspondence between successful optimizer updates and scheduler advances. Verify the planned warmup endpoint and final LR; an AMP-skipped update must not advance the schedule.

## 3. Make resume explicit and complete — MP-05, Medium

**Current evidence:** `resume_from_checkpoint()` loads model weights and epoch only (CNN cell 3, 969–978; Swin 1019–1028). The runner checks specifically `epoch_1_*.pth` (CNN 1117; Swin 1167) despite saving later checkpoints. Its optimizer parameter is unused.

**Implementation work**

- Add an explicit resume option that selects a named or latest complete compatible checkpoint.
- Restore optimizer, scheduler, AMP scaler, Python/NumPy/Torch RNG states, data-order state, MA progress, next epoch/update, history, and best-model tracking.
- Validate source/config/dataset identity. Reject incompatible or incomplete checkpoints with an actionable error.
- Give weights-only warm starts a separate option and a new run identity. Load checkpoints only from trusted sources; the current code uses `torch.load(weights_only=False)`.

**Acceptance checks**

Compare a small deterministic uninterrupted run against interruption and restoration: next-batch order, parameters, optimizer state, LR, MA progress, history, and best checkpoint must match. With epoch-one and epoch-four checkpoints present, assert selection of the requested/latest complete checkpoint and rejection of incompatible metadata.

## 4. Separate selection from final evaluation — MP-04, Medium

**Current evidence:** `run_experiment()` creates `val_dataset` with `split="test"` (CNN cell 3, 1009–1015; Swin 1059–1065), selects the best checkpoint on that loader, then reports results on it. Dataset caps select a sorted filename prefix.

**Implementation work**

- Derive a declared validation split from training data, respecting scene/sequence groups where available. Record actual image IDs, group IDs, split seed, and subset selection rules.
- Use validation alone for checkpoint, distribution, and hyperparameter selection.
- Freeze the procedure before a separate final test evaluation. Label earlier results as test-set selection scores and disclose that limitation.

**Acceptance checks**

Assert train/validation/test ID disjointness and group separation when applicable. Prove the selection path cannot receive the final test loader. Test split regeneration from its manifest and validate visible/IR/annotation alignment on a tiny paired fixture. Dataset scene overlap and the magnitude of selection bias require actual data review.

## 5. Make the setup reproducible — MP-06, Medium

**Current evidence:** the two notebooks are root files. `main()` hard-codes `/kaggle/input/datasets/riyajain1040/llvip-dataset/LLVIP` and may call `os.listdir()` before checking its existence (CNN cell 3, 1556 onward; Swin 1582 onward). No dependency manifest, lock, tests, or CI workflow is committed. README paths and setup notes are corrected in this documentation change, while these runtime gaps remain.

**Implementation work**

- Expose dataset/output/experiment configuration at a single entry point, with notebook-friendly defaults and a setup-only mode.
- Validate required directories, paired filenames, annotations, and writable outputs before listing data, downloading weights, or initializing training. Return useful path-specific errors.
- Capture tested Python, PyTorch/Torchvision, NumPy, Pillow, CUDA, driver, and GPU versions; commit an environment specification with a reproducible install procedure.
- Add dataset source/version/terms and split manifests. Keep dataset assets and large checkpoints outside Git with documented retrieval/provenance.
- Record separate output/run directories and eliminate implicit continuation from existing files.

**Acceptance checks**

In a clean runtime, run setup-only checks on a tiny local paired dataset without GPU or model downloads. A missing root, missing modality, mismatched pair, malformed annotation, and unwritable output must fail predictably. Verify a documented small GPU smoke run after the core fixes.

## 6. Regenerate traceable results — MP-02, Medium

**Current evidence:** [final_results.json](final_results.json) and [final_results.csv](final_results.csv) record `swin_b2`, while the Swin model constructor accepts `resnet50` or `swin_tiny` (cell 3, 585–596). `B2_swin_variant_2` is not a current experiment definition. Its saved AVG AP50 is 86.4 although (90.25 + 86.54) / 2 = 88.395. These are confirmed artifact inconsistencies; they do not establish how the historical numbers were produced.

**Implementation work**

- Preserve the original artifacts as historical records. Investigate missing source/config provenance rather than changing a result label or average and calling it reproduced.
- Define one result schema with run ID, commit SHA, full config, validated experiment/backbone IDs, seed, environment, split IDs, checkpoint identity, raw metrics, and evaluator protocol.
- Compute aggregate values from component metrics; specify rounding and tolerance. Generate JSON, CSV, and README summaries from that one source.
- Represent failed/incomplete runs explicitly so they cannot be ranked as valid zero-valued results.

**Acceptance checks**

Reject unknown experiment/config identifiers and inconsistent averages. Rebuild all summaries deterministically from the same run artifact. Ensure every reported row links to the matching config, split manifest, checkpoint, and logs. Require corrected runs before promoting any comparative result to a supported conclusion.

## 7. Measure training and evaluation memory separately — MP-08, Low

**Current evidence:** each epoch prints then resets CUDA peak counters (CNN cell 3, 1195–1199; Swin 1245–1249). The final result reads `max_memory_allocated()` after training and evaluation (CNN 1250; Swin 1300). It therefore does not retain a run-wide maximum of preceding training epochs.

**Implementation work**

Maintain a run-level maximum across epoch peaks, or reset once before the entire measured phase. Export separate training/evaluation allocated and reserved peaks with units (GB or GiB), measurement boundaries, and device information.

**Acceptance checks**

Inject known per-epoch peaks and verify the exported maximum survives counter resets. In a GPU smoke run, compare the records against independent monitoring and document differences in what each tool measures.

## 8. Rebuild the comparative evidence — MP-07, Low

**Current evidence:** configs use one seed (42), different CNN/Swin pretraining and learning rates, and the unresolved training/evaluation paths above. The earlier README made stronger distribution and gamma claims than these controls support. This documentation change calibrates those claims; new experiments are still required.

**Implementation work**

- Declare the research questions, primary comparison, controls, failure criteria, evaluator, and analysis before the next study.
- Disclose every config difference; treat CNN/Swin as detector-configuration comparisons unless other factors are controlled.
- Run multiple paired seeds, retain failed runs, and report per-seed results, aggregate effects, and uncertainty.
- Name the custom VOC-style 11-point AP50 protocol and check agreement against a trusted evaluator on shared fixtures. Do not compare unlike metric protocols as equivalent.
- Treat U(0,1) as bounded and any explanation for observed distribution differences as a hypothesis until tested.

**Acceptance checks**

Generate methods tables from run configs, checking that every differing field is disclosed. Require corrected training-state checks, disjoint splits, traceable result records, and replicated runs before asserting a robust winner. Independent verification of original-paper claims is outside this source review.

## Repository polish after the core checks

- Extract shared config/data/training/evaluation code into importable modules while retaining notebooks as thin experiment walkthroughs; keep CNN/Swin behavior differences explicit.
- Add CPU CI for notebook JSON/syntax, split/result schema validation, paired-data fixtures, scheduler/resume tests, and README local-path checks. Add an optional small GPU job when resources permit.
- Add contribution guidance describing required artifacts and test commands once those commands exist. Require relevant checks before merging future training changes.

**Completion gate:** a new contributor can recreate the tested environment, run the tiny checks, execute a documented experiment, and trace every published metric to its exact code, data split, configuration, checkpoint, and evaluator. Complete this gate before describing the repository as reproducible or polished research software.
