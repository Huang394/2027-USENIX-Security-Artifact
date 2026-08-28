# Licenses And Provenance

This artifact contains project code, released logs, and derived summaries for review. It also refers to third-party assets that are not redistributed here.

## Included Code

- `code/main_experiments/` preserves the main purification/evaluation implementation. This component is built on the public ZIP repository, `https://github.com/sycny/ZIP`, which accompanies the NeurIPS 2023 paper `Black-box Backdoor Defense via Zero-shot Image Purification` by Yucheng Shi, Mengnan Du, Xuansheng Wu, Zihan Guan, Jin Sun, and Ninghao Liu. The copied upstream license is preserved in `code/main_experiments/LICENSE`.
- `code/analysis_tool/` preserves the mechanism-analysis implementation used to generate released mechanism evidence.

## Third-Party Or External Assets

The following assets are documented for full reproduction but are not bundled as full datasets or checkpoints:

| Asset | Role |
| --- | --- |
| Imagenette2 | Main evaluation dataset. |
| CIFAR-10 | External-validity dataset. |
| Backdoored victim classifiers | Required for full CA/ASR/PA reruns. |
| ZIP/guided-diffusion components | Required for full I2I-Diffusion reruns. |
| ConvIR checkpoints | Required for full dehazing/deblurring restoration reruns. |
| Lite-BD/SwinIR | Third-party baseline; source code is not redistributed in this artifact. |

## Lite-BD Policy

Lite-BD is included only as audit evidence through logs and parsed rows. This artifact does not redistribute Lite-BD source code. Full Lite-BD reruns should use the official Lite-BD release and its license terms.

## Logs And Derived Results

The files under `logs/` and `results/` are released to audit paper-reported measurements. They should be interpreted together with `README.md` and `REPRODUCIBILITY.md`, especially the distinction between full-split paper-level metrics and sampled mechanism diagnostics.
