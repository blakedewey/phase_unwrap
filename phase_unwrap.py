import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

ORIENT_DICT = {"R": "L", "A": "P", "I": "S", "L": "R", "P": "A", "S": "I"}
GAUSS_STDEV = 10.0


def check_3d(obj: nib.Nifti1Image) -> nib.Nifti1Image:
    if len(obj.shape) > 3:
        obj_list = nib.four_to_three(obj)
        obj = obj_list[1]  # Assume phase image is 2nd volume
    return obj


def reorient(obj: nib.Nifti1Image, orientation: str) -> nib.Nifti1Image:
    if orientation is None:
        return obj
    target_orient = [ORIENT_DICT[char] for char in orientation]
    if nib.aff2axcodes(obj.affine) != tuple(target_orient):
        orig_ornt = nib.orientations.io_orientation(obj.affine)
        targ_ornt = nib.orientations.axcodes2ornt(target_orient)
        ornt_xfm = nib.orientations.ornt_transform(orig_ornt, targ_ornt)

        affine = obj.affine.dot(nib.orientations.inv_ornt_aff(ornt_xfm, obj.shape))
        data = nib.orientations.apply_orientation(obj.dataobj, ornt_xfm)
        obj = nib.Nifti1Image(data, affine, obj.header)
    return obj


def unwrap_phase(phase_obj: nib.Nifti1Image) -> nib.Nifti1Image:
    phase_data = phase_obj.get_fdata().astype(np.float32)
    if phase_data.max() > 3.15:
        if phase_data.min() >= 0:
            norm_phase = ((phase_data / phase_data.max()) * 2 * np.pi) - np.pi
        else:
            norm_phase = (phase_data / phase_data.max()) * np.pi
    else:
        norm_phase = phase_data

    dim = norm_phase.shape
    tmp = np.array(
        np.array(range(int(np.floor(-dim[1] / 2)), int(np.floor(dim[1] / 2)))) / float(dim[1])
    )
    tmp = tmp.reshape((1, dim[1]))
    uu = np.ones((1, dim[0]))
    xx = np.dot(tmp.conj().T, uu).conj().T
    tmp = np.array(
        np.array(range(int(np.floor(-dim[0] / 2)), int(np.floor(dim[0] / 2)))) / float(dim[0])
    )
    tmp = tmp.reshape((1, dim[0]))
    uu = np.ones((dim[1], 1))
    yy = np.dot(uu, tmp).conj().T
    kk2 = xx**2 + yy**2
    hp1 = gauss_filter(dim[0], GAUSS_STDEV, dim[1], GAUSS_STDEV)

    filter_phase = np.zeros_like(norm_phase)
    with np.errstate(divide="ignore"):
        for i in range(dim[2]):
            z_slice = norm_phase[:, :, i]
            lap_sin = -4.0 * (np.pi**2) * icfft(kk2 * cfft(np.sin(z_slice)))
            lap_cos = -4.0 * (np.pi**2) * icfft(kk2 * cfft(np.cos(z_slice)))
            lap_theta = np.cos(z_slice) * lap_sin - np.sin(z_slice) * lap_cos
            tmp = np.array(-cfft(lap_theta) / (4.0 * (np.pi**2) * kk2))
            tmp[np.isnan(tmp)] = 1.0
            tmp[np.isinf(tmp)] = 1.0
            kx2 = tmp * (1 - hp1)
            filter_phase[:, :, i] = np.real(icfft(kx2))

    filter_phase[filter_phase > np.pi] = np.pi
    filter_phase[filter_phase < -np.pi] = -np.pi
    filter_phase *= -1.0

    filter_obj = nib.Nifti1Image(filter_phase, phase_obj.affine, phase_obj.header)
    filter_obj.set_data_dtype(np.float32)
    return filter_obj


def cfft(img_array: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img_array)))


def icfft(freq_array: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(freq_array)))


def gauss_filter(dimx: int, stdevx: float, dimy: int, stdevy: float) -> np.ndarray:
    if dimx % 2 == 0:
        centerx = (dimx / 2.0) + 1
    else:
        centerx = (dimx + 1) / 2.0
    if dimy % 2 == 0:
        centery = (dimy / 2.0) + 1
    else:
        centery = (dimy + 1) / 2.0
    kki = np.array(range(1, dimy + 1)).reshape((1, dimy)) - centery
    kkj = np.array(range(1, dimx + 1)).reshape((1, dimx)) - centerx

    h = gauss(kkj, stdevy).conj().T * gauss(kki, stdevx)
    h /= h.sum()
    h /= h.max()
    return h


def gauss(r: np.ndarray, std0: float) -> np.ndarray:
    return np.exp(-(r**2) / (2 * (std0**2))) / (std0 * np.sqrt(2 * np.pi))


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("phase_image", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--orientation", type=str)
    parsed = parser.parse_args(args)

    parsed.phase_image = parsed.phase_image.resolve()
    if parsed.output is None:
        parsed.output = parsed.phase_image.parent / parsed.phase_image.name.replace(
            ".nii.gz", "_unwrapped.nii.gz"
        )
    else:
        parsed.output = parsed.output.resolve()

    obj = reorient(
        check_3d(nib.Nifti1Image.load(parsed.phase_image)),
        parsed.orientation,
    )
    filter_obj = unwrap_phase(obj)
    filter_obj.to_filename(parsed.output)


if __name__ == "__main__":
    main()
