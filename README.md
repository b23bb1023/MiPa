# MiPa: Visible–Infrared Patch Mixing Experiments

An exploratory study of mixed visible/infrared patches for pedestrian detection on LLVIP. The notebooks adapt the idea from **Mixed Patch Visible-Infrared Modality Agnostic Object Detection** (Medeiros et al., WACV 2025) to Faster R-CNN, comparing mixing-ratio distributions and ResNet50/Swin-Tiny detector configurations.

**Status: research prototype requiring training and evaluation corrections.** The committed MA (modality-adversarial) loss has a zero schedule throughout training, and existing result artifacts have unresolved provenance and arithmetic issues. Treat the saved outputs as historical, provisional artifacts. They do not establish a distribution winner or a successful reproduction of the original paper. The implementation tasks and acceptance checks are in [improvement_repo.md](improvement_repo.md).

## What is included

| File | Purpose |
| --- | --- |
| [mipa-cnn.ipynb](mipa-cnn.ipynb) | Faster R-CNN with ResNet50-FPN v2; five mixing-distribution configurations |
| [mipa-ViT.ipynb](mipa-ViT.ipynb) | Faster R-CNN with Swin-Tiny + FPN; three mixing-distribution configurations |
| [final_results.json](final_results.json) | Two historical result records and serialized configurations |
| [final_results.csv](final_results.csv) | Tabular version of those historical results |
| `DL REPORT.docx`, `DL PROJECT.pptx` | Project report and presentation |
| [improvement_repo.md](improvement_repo.md) | Prioritized technical remediation and verification plan |
| [LICENSE](LICENSE) | MIT license for this repository |

The notebooks are at the repository root. There is no committed `notebooks/` or `results/` directory, installable package, dependency lock, automated test suite, or CI workflow. Dataset images, annotations, and model checkpoints are not included.

## Research scope

The patch mixer constructs one image from spatially paired RGB and IR patches. The intended MA module uses a modality classifier and gradient reversal to encourage modality-invariant features.

The CNN notebook defines uniform U(0,1), Beta(2,2), Beta(0.5,0.5), truncated uniform U(0.15,0.85), and clipped Gaussian mixing ratios. The Swin notebook defines uniform, Beta(2,2), and clipped Gaussian variants. U(0,1) is bounded; Beta(2,2) reduces the probability of extreme ratios but does not exclude them.

The detector and pixel-space mixing implementation differ from the original project's setup. Equivalence to token-level mixing has not been established here. The two detector configurations also differ in pretraining and learning rate, so their comparison does not isolate the backbone alone.

### Intended configuration in `define_experiments()`

These values describe the committed configuration, rather than a validated training procedure.

| Setting | CNN notebook | Swin notebook |
| --- | --- | --- |
| Detector | Faster R-CNN | Faster R-CNN |
| Backbone | ResNet50-FPN v2 | Swin-Tiny + FPN |
| Pretraining path | COCO detector weights | ImageNet backbone weights |
| Learning rate | 2e-4 | 1e-4 |
| Training / evaluation subset caps | 4,000 / 800 pairs | 4,000 / 800 pairs |
| Image height × width | 512 × 640 | 512 × 640 |
| Epochs / seed | 6 / 42 | 6 / 42 |
| Batch / gradient accumulation | 8 / 2 | 8 / 2 |
| Patch size | 32 × 32 | 32 × 32 |
| MA setting | Enabled, gamma 0.05; currently zero-weighted | Enabled, gamma 0.05; currently zero-weighted |

## Known limitations and results

The source at commit `3e63bbc11b6543bc0a01572af868aab0d8295d5d` has the following unresolved issues:

- `global_step` is initialized to zero and never advanced. Consequently, MA lambda and the weighted MA objective remain zero. Claims about MA or gamma effects require corrected experiments.
- Scheduler lengths count minibatches, while scheduler calls occur at accumulated optimizer updates and once more per epoch. The intended warmup/decay schedule is not implemented consistently.
- The official test subset is used for repeated evaluation and checkpoint selection. Those scores are selection scores, rather than an untouched final test estimate.
- Automatic resume selects an epoch-one checkpoint and restores model weights without the full training state.
- Both serialized result configs specify `swin_b2`, while the Swin notebook supports `resnet50` and `swin_tiny`. The second experiment identifier is also absent from the current definitions.
- The second result stores IR AP50 90.25, RGB AP50 86.54, and AVG AP50 86.4. Their arithmetic mean is 88.395 (approximately 88.40). This discrepancy needs provenance investigation and regeneration; it has not been silently corrected.
- Exported GPU peak memory is read after per-epoch peak counters have been reset, so it may not represent the training peak.

The custom evaluator uses VOC-style 11-point interpolated AP at IoU 0.5. State this protocol when reporting metrics, and validate it against a trusted implementation before comparing results from other evaluators. Historical notebook output is not evidence that the current source produced a particular saved number.

## Opening and preparing the notebooks

The intended environment is a Kaggle notebook with a T4-class GPU. The source imports NumPy, PyTorch, Torchvision, and Pillow, and pretrained weights may require an initial network download. No tested package versions or complete environment specification are committed yet.

1. Open either [mipa-cnn.ipynb](mipa-cnn.ipynb) or [mipa-ViT.ipynb](mipa-ViT.ipynb) in a separate Kaggle session.
2. Attach an LLVIP dataset you are permitted to use. The dataset root must contain `visible/train`, `visible/test`, `infrared/train`, `infrared/test`, and the XML annotation directory (`Annotations` in the documented layout). Paired visible/IR files must share filenames.
3. Before executing the third code cell, inspect `main()`. It currently assigns `dataset_root = "/kaggle/input/datasets/riyajain1040/llvip-dataset/LLVIP"`. Update that assignment in your working notebook to the actual mounted directory. Changing only the dataclass default will not override `main()`. A missing mount can raise `FileNotFoundError` before a helpful validation message.
4. Check the output directory in `main()`: `/kaggle/working/mipa_results`, with a `./mipa_results` fallback outside Kaggle. Use separate, empty output directories for independent runs to avoid the existing automatic resume behavior.
5. Review `Config`, `define_experiments()`, and the [technical plan](improvement_repo.md) before starting training. Resolve the MA, scheduler, resume, and split-selection issues before producing new benchmark claims.

These steps identify the current entry points and required manual setup; a clean-runtime training reproduction has not been verified. GPU memory and elapsed time depend on the environment and should be measured again after the fixes.

## Contributing

Start with the ordered tasks in [improvement_repo.md](improvement_repo.md). Keep fixes small, include the specified regression checks, and retain the original raw results with their provenance. Publish new metrics only with the source commit, complete config, environment, split IDs, seed, evaluator protocol, and run artifacts.

## References and attribution

- Original project: [heitorrapela/MiPa](https://github.com/heitorrapela/MiPa)
- Original paper: [Medeiros et al., WACV 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Medeiros_Mixed_Patch_Visible-Infrared_Modality_Agnostic_Object_Detection_WACV_2025_paper.pdf)
- Dataset: [LLVIP](https://bupt-ai-cz.github.io/LLVIP/) — dataset rights and terms are separate from this repository's MIT license.
- Frameworks: PyTorch and Torchvision; intended hosted runtime: Kaggle.

If your work builds on the method, cite the original paper:

```bibtex
@inproceedings{medeiros2025mipa,
  title={Mixed Patch Visible-Infrared Modality Agnostic Object Detection},
  author={Medeiros, Heitor R. and Latortue, David and Granger, Eric and Pedersoli, Marco},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages={9005--9014},
  year={2025}
}
```
