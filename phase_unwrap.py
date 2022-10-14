import os
import shutil
import argparse

import numpy as np
import nibabel as nib

from nipype import Workflow, Node
from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec, TraitedSpec, traits, File, 
                                    isdefined)
from nipype.utils.filemanip import split_filename

np.seterr(all='ignore')


class Check3DInputSpec(BaseInterfaceInputSpec):
    input_image = File(exists=True, desc='image to check', mandatory=True)
    volume_num = traits.Int(default_value=0, desc='3D volume to extract (0 count)', usedefault=True)


class Check3DOutputSpec(TraitedSpec):
    out_file = File(desc='3d image')


class Check3D(BaseInterface):
    input_spec = Check3DInputSpec
    output_spec = Check3DOutputSpec
    
    def _run_interface(self, runtime):
        obj = nib.load(self.inputs.input_image)
        if len(obj.shape) > 3:
            obj_list = nib.four_to_three(obj)
            obj_list[self.inputs.volume_num].to_filename(split_filename(self.inputs.input_image)[1] + '_3d.nii.gz')
        else:
            obj.to_filename(split_filename(self.inputs.input_image)[1] + '_3d.nii.gz')
        
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = os.path.abspath(split_filename(self.inputs.input_image)[1] + '_3d.nii.gz')
        return outputs


class ReorientInputSpec(BaseInterfaceInputSpec):
    input_image = File(exists=True, desc='image to check', mandatory=True)
    orientation = traits.String('RAI', desc='orientation string', mandatory=True, usedefault=True)


class ReorientOutputSpec(TraitedSpec):
    out_file = File(desc='reoriented image')


class Reorient(BaseInterface):
    input_spec = ReorientInputSpec
    output_spec = ReorientOutputSpec
    
    def _run_interface(self, runtime):
        orient_dict = {'R': 'L', 'A': 'P', 'I': 'S', 'L': 'R', 'P': 'A', 'S': 'I'}
        
        obj = nib.load(self.inputs.input_image)
        target_orient = [orient_dict[char] for char in self.inputs.orientation]
        if nib.aff2axcodes(obj.affine) != tuple(target_orient):
            orig_ornt = nib.orientations.io_orientation(obj.affine)
            targ_ornt = nib.orientations.axcodes2ornt(target_orient)
            ornt_xfm = nib.orientations.ornt_transform(orig_ornt, targ_ornt)
            
            affine = obj.affine.dot(nib.orientations.inv_ornt_aff(ornt_xfm, obj.shape))
            data = nib.orientations.apply_orientation(obj.dataobj, ornt_xfm)
            obj_new = nib.Nifti1Image(data, affine, obj.header)
            obj_new.to_filename(split_filename(self.inputs.input_image)[1] + '_reorient.nii.gz')
        else:
            obj.to_filename(split_filename(self.inputs.input_image)[1] + '_reorient.nii.gz')
        
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = os.path.abspath(split_filename(self.inputs.input_image)[1] + '_reorient.nii.gz')
        return outputs


class UnwrapPhaseInputSpec(BaseInterfaceInputSpec):
    phase = File(exists=True, desc='T2* Phase Image', mandatory=True)
    gauss_stdev = traits.Int(10, desc='Std Dev of Gaussian for HP Filter', mandatory=True, usedefault=True)
    scaled_phase = File(desc='Scaled Phase Image [-pi, pi]')
    unwrapped_phase = File(desc='Unwrapped Phase Image')


class UnwrapPhaseOutputSpec(TraitedSpec):
    scaled_phase = File(exists=True, desc='Scaled Phase Image [-pi, pi]')
    unwrapped_phase = File(exists=True, desc='Unwrapped Phase Image')


class UnwrapPhase(BaseInterface):
    input_spec = UnwrapPhaseInputSpec
    output_spec = UnwrapPhaseOutputSpec

    def _run_interface(self, runtime):
        phase_obj = nib.load(self.inputs.phase)
        phase_data = phase_obj.get_fdata().astype(np.float32)
        if phase_data.max() > 3.15:
            if phase_data.min() >= 0:
                norm_phase = ((phase_data / phase_data.max()) * 2 * np.pi) - np.pi
            else:
                norm_phase = (phase_data / phase_data.max()) * np.pi
        else:
            norm_phase = phase_data

        dim = norm_phase.shape
        tmp = np.array(np.array(range(int(np.floor(-dim[1] / 2)), int(np.floor(dim[1] / 2)))) / float(dim[1]))
        tmp = tmp.reshape((1, dim[1]))
        uu = np.ones((1, dim[0]))
        xx = np.dot(tmp.conj().T, uu).conj().T
        tmp = np.array(np.array(range(int(np.floor(-dim[0] / 2)), int(np.floor(dim[0] / 2)))) / float(dim[0]))
        tmp = tmp.reshape((1, dim[0]))
        uu = np.ones((dim[1], 1))
        yy = np.dot(uu, tmp).conj().T
        kk2 = xx ** 2 + yy ** 2
        hp1 = gauss_filter(dim[0], self.inputs.gauss_stdev, dim[1], self.inputs.gauss_stdev)

        filter_phase = np.zeros_like(norm_phase)
        for i in range(dim[2]):
            z_slice = norm_phase[:, :, i]
            lap_sin = -4.0 * (np.pi ** 2) * icfft(kk2 * cfft(np.sin(z_slice)))
            lap_cos = -4.0 * (np.pi ** 2) * icfft(kk2 * cfft(np.cos(z_slice)))
            lap_theta = np.cos(z_slice) * lap_sin - np.sin(z_slice) * lap_cos
            tmp = np.array(-cfft(lap_theta) / (4.0 * (np.pi ** 2) * kk2))
            tmp[np.isnan(tmp)] = 1.0
            tmp[np.isinf(tmp)] = 1.0
            kx2 = (tmp * (1 - hp1))
            filter_phase[:, :, i] = np.real(icfft(kx2))

        filter_phase[filter_phase > np.pi] = np.pi
        filter_phase[filter_phase < -np.pi] = -np.pi
        filter_phase *= -1.0

        scaled_obj = nib.Nifti1Image(norm_phase, None, phase_obj.header)
        scaled_obj.set_data_dtype(np.float32)
        if isdefined(self.inputs.scaled_phase):
            scaled_obj.to_filename(self.inputs.scaled_phase)
        else:
            scaled_obj.to_filename(split_filename(self.inputs.phase)[1] + '_scaled.nii.gz')

        filter_obj = nib.Nifti1Image(filter_phase, None, phase_obj.header)
        filter_obj.set_data_dtype(np.float32)
        if isdefined(self.inputs.unwrapped_phase):
            filter_obj.to_filename(self.inputs.unwrapped_phase)
        else:
            filter_obj.to_filename(split_filename(self.inputs.phase)[1] + '_unwrapped.nii.gz')

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        if isdefined(self.inputs.scaled_phase):
            outputs['scaled_phase'] = self.inputs.scaled_phase
        else:
            outputs['scaled_phase'] = os.path.abspath(split_filename(self.inputs.phase)[1] + '_scaled.nii.gz')
        if isdefined(self.inputs.unwrapped_phase):
            outputs['unwrapped_phase'] = self.inputs.unwrapped_phase
        else:
            outputs['unwrapped_phase'] = os.path.abspath(split_filename(self.inputs.phase)[1] + '_unwrapped.nii.gz')
        return outputs


def cfft(img_array):
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img_array)))


def icfft(freq_array):
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(freq_array)))


def gauss_filter(dimx, stdevx, dimy, stdevy):
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


def gauss(r, std0):
    return np.exp(-(r ** 2) / (2 * (std0 ** 2))) / (std0 * np.sqrt(2 * np.pi))


class MoveResultFileInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc='input file to be renamed', mandatory=True)
    output_name = traits.String(desc='output name string')


class MoveResultFileOutputSpec(TraitedSpec):
    out_file = File(desc='path of moved file')


class MoveResultFile(BaseInterface):
    input_spec = MoveResultFileInputSpec
    output_spec = MoveResultFileOutputSpec

    def _run_interface(self, runtime):
        shutil.copyfile(self.inputs.in_file, self.inputs.output_name)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self.inputs.output_name
        return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--phase-image', type=str, default='T2STAR_PHA.nii.gz')
    parser.add_argument('-o', '--output-prefix', type=str, default='T2STAR_PHA')
    parser.add_argument('--orientation', type=str, default='RAI')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    
    for argname in ['phase_image', 'output_prefix']:
        setattr(args, argname, os.path.join('/data', getattr(args, argname)))
        
    wf = Workflow('qsm')

    # Check3D (choose 2nd volume)
    check3d_phase = Node(Check3D(volume_num=1), 'check3d_phase')
    check3d_phase.inputs.input_image = args.phase_image
    check3d_phase.inputs.volume_num = 1

    # Reorient phase image
    reorient_phase = Node(Reorient(), 'reorient_phase')
    reorient_phase.inputs.orientation = args.orientation
    wf.connect([(check3d_phase, reorient_phase, [('out_file', 'input_image')])])

    # Unwrap phase image
    unwrap_phase = Node(UnwrapPhase(), 'unwrap_phase')
    wf.connect([(reorient_phase, unwrap_phase, [('out_file', 'phase')])])

    # Move unwrapped phase image to output
    move_unwrapped_phase = Node(MoveResultFile(), 'move_unwrapped_phase')
    move_unwrapped_phase.inputs.output_name = args.output_prefix + '_UNWRAPPED.nii.gz'
    wf.connect([(unwrap_phase, move_unwrapped_phase, [('unwrapped_phase', 'in_file')])])
    
    wf.run()
