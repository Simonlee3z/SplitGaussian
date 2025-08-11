<p align="center">
  
  <h2 align="center"><strong>SplitGaussian: Reconstructing Dynamic Scenes via Visual Geometry Decomposition</strong></h2>

  <p align="center">
  <span>
    Jiahui Li<sup>1</sup>,
    Shengeng Tang<sup>1</sup>,
    Jingxuan He<sup>2</sup>,
    Gang Huang<sup>1</sup>,
    Zhangye Wang<sup>1</sup>,
    Yantao Pan<sup>3</sup>,
    <a href="https://scholar.google.com/citations?user=PKFAv-cAAAAJ&hl=en">Lechao Cheng</a><sup>2,3✉</sup>
  </span>
    <br>
  <span>
    <!-- <sup>*</sup>Equal contribution. -->
    <sup>✉</sup>Corresponding author.
    <br>
    <sup>1</sup>Zhejiang University,
    <sup>2</sup>Hefei University of Technology,
    <sup>3</sup>KAIYANG Laboratory, Chery
  </span>
</p>

<div align="center">

<!-- <a href='https://arxiv.org/abs/2404.08966'><img src='https://img.shields.io/badge/arXiv-2404.08966-b31b1b.svg'></a> -->

<a href='https://simonlee3z.github.io/SplitGaussian-Page/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>

</div>

## Getting Started

```Shell
git clone https://github.com/Simonlee3z/SplitGaussian.git
cd SplitGaussian
conda env create -f splitgs.yaml

```

## Dataset

We follow the data organization of ["GauFRe: Gaussian Deformation Fields for Real-time Dynamic Novel View Synthesis"](https://github.com/brownvc/GauFRe/). which could be downloaded [here](https://1drv.ms/f/c/4dd35d8ee847a247/EpmindtZTxxBiSjYVuaaiuUBr7w3nOzEl6GjrWjmVPuBFw?e=cW5gg1)

## Train

**NeRF-DS:**

```shell
python train.py -s path/to/your/NeRF-DS/dataset -m output/exp-name --eval --configpath path/to/NeRF-DS/config
```

## Render

```shell
python render.py -m output/exp-name --mode render --configpath path/to/NeRF-DS/config
```

## Acknowledgments

We thank [3DGS](https://github.com/graphdeco-inria/gaussian-splatting)、[Deformable 3DGS](https://github.com/ingra14m/Deformable-3D-Gaussians)、[GauFre](https://github.com/brownvc/GauFRe)、[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), and [Track Anything](https://github.com/gaomingqi/Track-Anything) for source code reference.

## BibTex
