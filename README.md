# Perceptual Colour Models for 3D Gaussian Splatting 
Our implementation builds on Safeguard Gaussian Splatting's open-source code, which includes comparative models like LightGaussian, Mini-Splatting, and RadSplat. Inspired by SafeguardGS’s simple yet effective integration of score functions into a CUDA-differentiable 3DGS rasterizer, we extended this framework by exploring new score functions. We conducted training using a single NVIDIA A100 80GB, maintaining consistent iteration parameters, learning rates, and densification methods from the original 3DGS. For fair comparison, we standardized the loss function across all models, using a hybrid of Huber loss and D-SSIM.

### Dataset
Download the link below to download the dataset.
- [MipNeRF360](https://jonbarron.info/mipnerf360/)

### Implementation guide

1. Clone this repository

2. Set up the conda environment.
```shell
conda env create -f environment.yml
conda activate ColourGS
```

3. Train and evaluate the model for the bicycle scene in mipnerf360. 
```shell
python train.py -s data/mipnerf360/bicycle -m output/mipnerf360/bicycle --prune_method safeguard_gs --eval # pruning method: safeguard_gs
```

## Acknowledgments
Huge thanks to the authors of SafeguardGS who generously made their code available in public.

[ Reference of SafeguardGS ]
<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@misc{lee2024safeguardgs3dgaussianprimitive,
      title={SafeguardGS: 3D Gaussian Primitive Pruning While Avoiding Catastrophic Scene Destruction}, 
      author={Yongjae Lee and Zhaoliang Zhang and Deliang Fan},
      year={2024},
      eprint={2405.17793},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2405.17793}, 
}</code></pre>
  </div>
</section>
