# SplitGaussian: Reconstructing Dynamic Scenes via Visual Geometry Decomposition

## [Project page](https://simonlee3z.github.io/SplitGaussian-Page/)

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
python train.py -s path/to/your/NeRF-DS/dataset -m output/exp-name --eval --configpath path/to/NeRF-DS/config

## Render

python render.py -m output/exp-name --mode render --configpath path/to/NeRF-DS/config

## Acknowledgments

We thank [3DGS](https://github.com/graphdeco-inria/gaussian-splatting)、[Deformable 3DGS](https://github.com/ingra14m/Deformable-3D-Gaussians)、[GauFre](https://github.com/brownvc/GauFRe) for source code reference.

## BibTex
