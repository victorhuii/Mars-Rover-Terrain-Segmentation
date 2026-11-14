# Third-party notices

The compact model architecture uses TorchVision's MobileNetV3-small implementation and PyTorch. Their bundled license texts are included under `docs/licenses/`. The scratch candidates are initialized randomly and trained on AI4Mars; they do not incorporate ImageNet pretrained checkpoint weights. TorchVision's documentation notes that pretrained weights can have separate terms, so the ImageNet-pretrained comparison remains local and its redistribution terms are not established here.

AI4Mars is credited to Swan et al. and NASA/JPL. The [official Zenodo record](https://zenodo.org/records/15995036) reports CC BY 4.0; the [Curiosity mirror](https://huggingface.co/datasets/hassanjbara/AI4MARS) declares CC0, while the NASA catalogue leaves its license unspecified. See [the dataset card](dataset_card.md) for source and revision. No NASA affiliation or endorsement is claimed.

The web interface uses ONNX Runtime Web (MIT) and the DM Sans and Space Grotesk font families (SIL Open Font License). Copies of their notices are in `docs/licenses/`. The application serves fonts and runtime assets from its own origin rather than a third-party font CDN.
