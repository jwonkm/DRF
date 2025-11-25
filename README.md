# Dual Recursive Feedback on Generation and Appearance Latents for Pose-Robust Text-to-Image Diffusion (ICCV 2025)

<p align="center">
  <a href="https://arxiv.org/abs/2508.09575">
    <img src="https://img.shields.io/badge/arXiv-Paper-E1523D.svg?labelColor=555555&logo=arxiv&logoColor=white" alt="arXiv Paper" />
  </a>
  <a href="https://jwonkm.github.io/DRF/">
    <img src="https://img.shields.io/badge/Project-Page-FFC857.svg?labelColor=555555" alt="Project Page" />
  </a>
  <a href="https://github.com/jwonkm/DRF">
    <img src="https://img.shields.io/github/stars/jwonkm/DRF?style=social" alt="GitHub stars" />
  </a>
</p>


<p align="center">
  <a href="https://scholar.google.com/citations?user=LvFTDwwAAAAJ&hl">Jiwon Kim</a><sup>1</sup>,
  Pureum Kim<sup>1</sup>,
  <a href="https://scholar.google.com/citations?user=RE9ZWDwAAAAJ&hl=ko&oi=sra">SeonHwa Kim</a><sup>1</sup>,
  Soobin Park<sup>2</sup>,
  <a href="https://scholar.google.com/citations?user=mqNGNqEAAAAJ&hl=ko">Eunju Cha</a><sup>2</sup>,
  <a href="https://scholar.google.com/citations?user=aLYNnyoAAAAJ&hl=ko&oi=sra">Kyong Hwan Jin</a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup> Korea University &nbsp;&nbsp;
  <sup>2</sup> Sookmyung Women's University
</p>

![Algorithm](https://github.com/jwonkm/DRF/blob/main/docs/asset/main_qual2.png)
## Environment Setup
To install the environment, please run the following.
```
conda env create -f environment.yaml
conda activate drf
```
## Run
To run DRF, use a [notebook](https://github.com/jwonkm/DRF/blob/main/DRF_demo.ipynb) or run the below code.

We use a single NVIDIA RTX 3090 GPU for our experiments.
```bash
python run.py \
    --structure_image dataset/structure/person_mesh.jpg \
    --appearance_image dataset/appearance/tiger.jpg \
    --prompt "a photo of a tiger standing on the snow field" \
    --structure_prompt "a mesh of a standing human" \
    --appearance_prompt "a photo of a tiger walking on the snow field"
```

#### Optional arguments

- `disable_refiner`: If enabled, disables the refiner (and does not load it), reducing memory usage and inference time.
- `model` (`str`): When provided a `.safetensors` checkpoint path, loads the checkpoint as the base model instead of the default one.
- `benchmark`: If enabled, reports the inference time and peak memory usage for the current run.
- `structure_schedule` (`float`, default `0.6`): Ratio of diffusion steps during which **structure control** is active.  
  For example, with 50 sampling steps:
  - `0.6` → control is used for the first 60% (first 30 steps), then turned off for the remaining 40%.  
  - `0.7` → control is used for the first 70% (first 35 steps), then turned off for the remaining 30%.
- `appearance_schedule` (`float`, default `0.6`): Same as `structure_schedule`, but for **appearance control**.  
  e.g., `0.6` with 50 steps → appearance control is applied for the first 30 steps and disabled for the last 20.
- `seed` (`int`, default `90095`): Random seed for sampling. Use the same value to reproduce results across runs; change it to obtain different random outputs.


## Reference
If you find our work useful for your research, please cite our paper.
```
@InProceedings{Kim_2025_ICCV,
    author    = {Kim, Jiwon and Kim, Pureum and Kim, SeonHwa and Park, Soobin and Cha, Eunju and Jin, Kyong Hwan},
    title     = {Dual Recursive Feedback on Generation and Appearance Latents for Pose-Robust Text-to-Image Diffusion},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {15491-15500}
}
```
## Acknowledgements
Our code is based on [Ctrl-X](https://github.com/genforce/ctrl-x). We thank the authors for sharing their works.
