import os
import h5py
import numpy as np
import tomllib
import pickle
from pyscf import lib
from pyscf.lib import logger
from pyscf.data.nist import BOHR
from pyscf.pbc.tools import pyscf_ase
from iwf.iwf import IWF
from iwf.plot import BandPlotter


class IWFWorkflow(lib.StreamObject):
    def __init__(self, kmf, kmesh, verbose=4):
        self.kmf = kmf
        self.kmesh = kmesh
        self.ase_obj = pyscf_ase.pyscf_to_ase_atoms(kmf.cell)
        self.iwf_obj = None
        self.verbose = verbose

        self.band_kwargs = {}
        self.downfold_kwargs = {}

        self.log = logger.new_logger(self, verbose=self.verbose)


    def run_mf(self):
        if self.band_kwargs.get('kpts_abs') is None:
            self.init_band_path_object()
        kmf = self.kmf
        dm0 = kmf.get_init_guess(key='chk') if os.path.exists('./kmf.chk') else None
        kmf.kernel(dm0=dm0)
        with h5py.File('./mean_field.h5', 'w') as f:
            for ds, method in [('hcore_ao', 'get_hcore'), ('ovlp_ao', 'get_ovlp'),
                            ('rdm1_ao', 'make_rdm1'), ('fock_ao', 'get_fock')]:
                f.create_dataset(ds, data=getattr(kmf, method)())
            # NOTE: Add mo_energy/mo_coeff/mo_occ.
            f['mo_energy'] = np.array(kmf.mo_energy)
            f['mo_coeff'] = np.array(kmf.mo_coeff)
            f['mo_occ'] = np.array(kmf.mo_occ)
        self.log.info('Mean-field calculation done and dumped to mean_field.h5')
        e_band, c_band = map(np.array,
                kmf.get_bands(kpts_band=self.band_kwargs['kpts_abs']))
        e_band -= kmf.get_fermi()
        with h5py.File('./bands.h5', 'w') as f:
            f['kpts_rel'] = self.band_kwargs['kpts_rel']
            f['kpts_abs'] = self.band_kwargs['kpts_abs']
            f['e_band']   = e_band
            f['c_band']   = c_band
        self.log.info('Band structure calculation done and dumped to bands.h5')


    def load_from_config(self, config):
        if isinstance(config, str):
            with open(config, 'rb') as f:
                config_dict = tomllib.load(f)
        elif isinstance(config, dict):
            config_dict = config
        self.band_kwargs.update(config_dict.get('band', {}))
        self.downfold_kwargs.update(config_dict.get('downfold', {}))


    def init_band_path_object(self):
        ase_obj = self.ase_obj
        bp = ase_obj.cell.bandpath(
            self.band_kwargs['kpath'],
            npoints=self.band_kwargs['npoints']
        )
        if self.band_kwargs.get('special_points') is None:
            self.band_kwargs['special_points'] = bp.special_points
        if self.band_kwargs.get('kpts_rel') is None:
            self.band_kwargs['kpts_rel'] = bp.kpts
        if self.band_kwargs.get('kpts_abs') is None:
            # The unit of resulting kpts_abs is Bohr
            # const: BOHR = 0.52917721092 Ang * (Bohr^{-1})
            self.band_kwargs['kpts_abs'] = self.band_kwargs['kpts_rel'] @ (
                ase_obj.cell.reciprocal().array * 2 * np.pi * BOHR)
        return self


    def init_iwf_object(self):
        kmf = self.kmf
        _keys_iwf = {'minao', 'core', 'val', 'ref_method', 'sprd_param'}
        _keys_general = {'downfold_labels', 'erange'}
        iwf_kwargs = {k: v for k, v in self.downfold_kwargs.items() if k in _keys_general}
        iwf_kwargs.update({k: v for k, v in self.downfold_kwargs.get('iwf', {}).items() if k in _keys_iwf})

        itp = self.downfold_kwargs.get('interpolate', True)

        if itp:
            if os.path.exists("./kmf.chk"):
                kmf.update_from_chk()
            else:
                with h5py.File("./mean_field.h5", "r") as f:
                    for k in ('mo_energy', 'mo_coeff', 'mo_occ'):
                        assert k in f.keys(), f"{k} not found in mean_field.h5"
                        kmf.__dict__[k] = f[k][:]
            e_fermi = kmf.get_fermi()
            mo_energy = np.array(kmf.mo_energy) - e_fermi   # Fermi-aligned, Hartree
            mo_coeff  = np.array(kmf.mo_coeff)
            mylo = IWF(cell=self.kmf.cell,
                        kpts_abs_or_kmesh=self.kmesh,
                        mo_energy=mo_energy, mo_coeff=mo_coeff,
                        verbose=self.verbose, **iwf_kwargs)
        else:
            with h5py.File('./bands.h5', 'r') as f:
                kpts_band_abs = f['kpts_abs'][:]
                mo_energy_band = f['e_band'][:]
                mo_coeff_band = f['c_band'][:]
            mylo = IWF(self.kmf.cell, kpts_abs_or_kmesh=kpts_band_abs,
                        mo_energy=mo_energy_band,
                        mo_coeff=mo_coeff_band,
                        verbose=self.verbose, **iwf_kwargs)
        self.iwf_obj = mylo
        return self


    def run_lo(self):
        if self.iwf_obj is None:
            self.init_iwf_object()
        mylo = self.iwf_obj
        lo_method = self.downfold_kwargs.get('lo_method', 'iwf')
        path = self.downfold_kwargs.get('path', f"./downfold/{lo_method}")
        write_cube = self.downfold_kwargs.get('write_cube', False)
        cube_kwargs = self.downfold_kwargs.get('cube', dict(nx=80, ny=80, nz=80))
        os.makedirs(path, exist_ok=True)

        if lo_method == "iwf":
            mylo.get_LO_ref()
            mylo.kernel()
            C_ao_lo = mylo.iwf
            with open(f"{path}/iwf.pkl", 'wb') as f:
                pickle.dump(mylo.anal_data, f)
        elif lo_method == "w90":
            w90_kwargs = self.downfold_kwargs.get('w90', {})
            if 'num_wann' in w90_kwargs:
                w90_kwargs.pop('num_wann')
            C_ao_lo = mylo.run_w90(w90_kwargs)
        elif lo_method == "scdm":
            scdm_kwargs = self.downfold_kwargs.get('scdm', {})
            C_ao_lo = mylo.run_scdm(scdm_kwargs)

        if write_cube:
            mylo.dump_cube(path=f"{path}/cube/", C_ao_lo=C_ao_lo, **cube_kwargs)
        np.save(f"./{path}/C_ao_lo.npy", C_ao_lo)
        return C_ao_lo


    def get_band_plotter(self, energy_range=[-10, 10], nrow=1, ncol=1, subplot_titles=None):
        return BandPlotter(
            ase_obj=self.ase_obj,
            kpath=self.band_kwargs['kpath'],
            kpts_rel=self.band_kwargs['kpts_rel'],
            kpts_abs=self.band_kwargs['kpts_abs'],
            npoints=self.band_kwargs['npoints'],
            special_points=self.band_kwargs['special_points'],
            energy_range=energy_range,
            nrow=nrow, ncol=ncol,
            subplot_titles=subplot_titles,
        )
