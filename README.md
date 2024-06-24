# MRI Phase Unwrapping

[![DOI](https://zenodo.org/badge/551228762.svg)](https://zenodo.org/badge/latestdoi/551228762)

A simple, laplacian-based unwrapping pipeline for MRI phase images.

If you use this code in your work, please cite:
```
Blake E. Dewey. (2022). Laplacian-based Phase Unwrapping in Python. Zenodo. [https://doi.org/10.5281/zenodo.7198990](https://doi.org/10.5281/zenodo.7198991)
```

## Install with pip
```bash
pip install phase_unwrap
```

## Basic Usage
```bash
unwrap-phase /path/to/phase_image.nii.gz
```
This will produce an unwrapped phase image in the same directory as the input image with `_unwrapped` appended to the filename.

**NOTE:** Some have reported the best results when the phase image is reoriented to the axial plane before unwrapping.
This can be done with the `--orientation` option:
```bash
unwrap-phase /path/to/phase_image.nii.gz --orientation RAI
```
If you would like the unwrapped image returned in the original orientation, use the `--undo-reorient` option.

### CLI Options
| Option            | Description                                                 | Default  |
|-------------------|-------------------------------------------------------------|----------|
|                   | Path to phase image                                         | Required |
| `-o`/`--output`   | Output path                                                 | Optional |
| `--orientation`   | Reorient to this before unwrapping (`RAI`, `RSA`, or `ASR`) | Optional |
| `--undo-reorient` | Return image to orientation after unwrapping                | `False`  |

Using the `--output` option will save the unwrapped image to that path. 
If not provided, the unwrapped image will be saved to the same directory as the input image with `_unwrapped` appended to the filename (before the `.nii.gz`).

`--orientation` can be used to reorient the image before unwrapping.
This script uses 2D functions for unwrapping and will slice the data according to the slice direction of the volume (last dimension).
If the slice direction is not the desired orientation, use this option to reorient the image before unwrapping.
`RAI` will give you an axial image. `RSA` will give you a coronal image. `ASR` will give you a sagittal image.
If you would like the unwrapped image returned in the original orientation, use the `--undo-reorient` option.

## Docker Usage
The docker file in this package is used to build a container with the necessary dependencies to run the unwrapping script.
You can pull it directly from Docker Hub with:
```bash
docker pull blakedewey/phase_unwrap:v2.0.0
```

After pulling the image, you can run the unwrapping script with:
```bash
docker run -it --rm -v /path/to/data:/data blakedewey/phase_unwrap:v2.0.0 /data/phase_image.nii.gz
```

All of the same CLI options will work with the Docker container as well.
Remember to mount the directory containing the data to a place in the container (`/data` in the example).
You will also have to specify the path to the image relative to the mounted directory.

## Upgrading from v1.0.0
The CLI options have changed slightly from v1.0.0 to v2.0.0:
 - The `-p`/`--phase-image` option has been replaced with a positional argument for the path to the phase image.
 - In v1.0.0, image paths were assumed to be in `/data` for use in Docker. This is no longer the case. You must specify the full path to the image or output, even in the Docker container.
 - In v1.0.0, the default for `--orientation` was `RAI`. This has been removed. If you want to reorient the image, you must specify the orientation. Use `--orientation RAI` to get the same behavior as v1.0.0.
 - The `--undo-reorient` option has been added to return the image to the original orientation after unwrapping.

## Works Using This Code
This processing has been used in a number of published manuscripts detailing phase-rim lesions in multiple sclerosis.

1. Absinta et al. "Persistent 7-tesla phase rim predicts poor outcome in new multiple sclerosis patient lesions" *Journal of Clinical Investigation* (doi: [10.1172/JCI86198](https://doi.org/10.1172%2FJCI86198))
