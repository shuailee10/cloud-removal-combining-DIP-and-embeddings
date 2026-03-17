# cloud-removal-combining-DIP-and-embeddings
This repository provides the scripts and code accompanying the DIP-based study for cloud removal. 

## :package: Key Python Packages for the Environment Setup
```bash
pytorch-lightning==1.2.0
torch==1.8.1
numpy=1.19.2
rasterio=1.0.21
```

## :card_index_dividers: Supporting Dataset
The dataset used in the experiments presented in the study can be found here:
https://ieee-dataport.org/documents/paired-sentinel-1-and-sentinel-2-images-and-google-satellite-embeddings-3-locations

## :memo: Citation
Please cite the following paper if you use any part of this repository:
```bibtex
@article{10.1016/j.srs.2026.100404,
  author = {Li, S. and He, Y. and Yang, D. and Li, Y. and Liu, W.},
  title = {Incorporating learned geospatial embeddings to deep image prior for inpainting cloud areas in remotely sensed images},
  journal = {Science of Remote Sensing},
  volume = {13},
  year = {2026},
  url = {https://www.sciencedirect.com/science/article/pii/S2666017226000428},
  doi = {10.1016/j.srs.2026.100404}
}
