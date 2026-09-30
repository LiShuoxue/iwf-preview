#!/usr/bin/env python

"""
wannier90 (v4.x) interface with pyscf.

Same user interface as ``pywannier90.W90`` (the v3 ``wannier_setup_`` /
``wannier_run_`` ctypes interface), but driven through the f90wrap python
wrapper (``wan90``) of the wannier90 v4 library. Compared with v3, the v4
library additionally supports e.g. the projectability-disentangled Wannier
functions (PDWF, ``dis_froz_proj``, ``dis_proj_min``, ``dis_proj_max``).

The wrapper directory (containing ``wan90.py`` and
``_wan90.cpython-*.so``) is taken from the environment variable ``W904PATH``.

Differences w.r.t. the v3 interface that matter to the callers:
    1. In v4, ``num_bands`` in the .win file is the number of bands *after*
       ``exclude_bands`` (v3 took the total number), handled in ``setup``.
    2. With projectability disentanglement the outer window is no longer a
       contiguous energy range. As in v3, the rows of ``U_matrix_opt`` are
       indexed by the bands inside the window, so use ``lwindow`` to select
       the bands: C_lo[k] = C_mo[k][:, lwindow[:, k]] @ U_opt[:ndimwin[k], :, k] @ U[:, :, k]

Shuoxue NOTE (Sep 29, 2026): the M/A/eigenvalue construction, the .win file
and the UNK/AME export are inherited from ``pywannier90.W90``.
"""

import os
import sys
import numpy as np

from pyscf.lib import logger

from iwf.backend import pywannier90
from iwf.backend.pywannier90 import (get_A_mat_from_lo, atomic_init_guess)

W904PATH = os.environ.get("W904PATH", None)
if W904PATH is not None and W904PATH not in sys.path:
    sys.path.insert(0, W904PATH)

try:
    import wan90
except ImportError as err:
    print("wannier90 v4 python wrapper (wan90) not found, check W904PATH. (%s)" % err)
    wan90 = None

# Fortran units of the opened output files. Opening one file on two
# different units is a gfortran runtime error, so reuse the unit.
_FORTRAN_UNITS = {}

def _get_fortran_unit(fname):
    fname = os.path.abspath(fname)
    if fname not in _FORTRAN_UNITS:
        _FORTRAN_UNITS[fname] = wan90.w90_library.w90_get_fortran_file(fname)
    return _FORTRAN_UNITS[fname]


class W90(pywannier90.W90):
    def __init__(self, kmf, kmesh, num_wann, gamma_only=False, spinors=False,
                 restricted=True, spin_up=None, other_keywords=None):
        """
        Main class to hold the input and output parameters for Wannier90 (v4).
        """
        pywannier90.W90.__init__(self, kmf, kmesh, num_wann, gamma_only=gamma_only,
                                 spinors=spinors, restricted=restricted,
                                 spin_up=spin_up, other_keywords=other_keywords)
        self.seed_name = "wannier90"
        # library object and the arrays pointed to by it
        self.w90_data = None
        self._w90_arrays = None
        # output of disentanglement
        self.ndimwin = None
        self.nfirstwin = None

    def make_win(self, fname=None, num_bands=None):
        """
        Make a basic *.win file for wannier90.
        num_bands: number of bands written to the .win (after exclude_bands in v4).
        """
        if num_bands is None:
            return pywannier90.W90.make_win(self, fname=fname)
        num_bands_tot = self.num_bands_tot
        self.num_bands_tot = num_bands
        try:
            pywannier90.W90.make_win(self, fname=fname)
        finally:
            self.num_bands_tot = num_bands_tot

    def _check(self, ierr, name):
        if ierr > 0:
            raise RuntimeError("Wannier90: %s failed (ierr = %d), see %s.wout / %s.werr"
                               % (name, ierr, self.seed_name, self.seed_name))
        elif ierr < 0:
            logger.warn(self, "Wannier90: %s returns warning (ierr = %d).", name, ierr)

    def _read_input(self):
        """
        Create the library object and read the .win file.
        """
        lib, lib_extra = wan90.w90_library, wan90.w90_library_extra
        data = lib.lib_common_type()
        self._check(lib_extra.input_reader_special(data, self.seed_name, self.istdout,
                                                   self.istderr), "input_reader_special")
        self._check(lib.w90_input_reader(data, self.istdout, self.istderr),
                    "w90_input_reader")
        return data

    def setup(self):
        """
        Read the .win file, setup the kmesh, and get the projections and
        excluded bands from the wannier90 v4 library.
        """
        logger.info(self, "Wannier90: setup start.")
        if wan90 is None:
            raise ImportError("wannier90 v4 python wrapper (wan90) not found, check W904PATH.")
        lib, lib_extra = wan90.w90_library, wan90.w90_library_extra
        self.istdout = _get_fortran_unit(self.seed_name + ".wout")
        self.istderr = _get_fortran_unit(self.seed_name + ".werr")

        # NOTE: v4 takes num_bands after exclusion, rewrite the .win if needed.
        data = self._read_input()
        num_excl = lib.w90_get_num_excl_bands(data)
        # NOTE: copy, the wrapper returns a view of the Fortran array, which is
        # freed with data when the .win is re-read below.
        exclude_bands = np.array(data.exclude_bands, dtype=np.int32, copy=True) \
                if num_excl > 0 else np.zeros(0, dtype=np.int32)
        num_bands = self.num_bands_tot - num_excl
        if data.num_bands != num_bands:
            self.make_win(num_bands=num_bands)
            data = self._read_input()
        assert data.num_bands == num_bands
        assert data.num_wann == self.num_wann
        assert data.num_kpts == self.num_kpts

        # all kpoints on rank 0
        lib_extra.set_kpoint_distribution(data, np.zeros(self.num_kpts, dtype=np.int32),
                                          self.istdout, self.istderr)
        self._check(lib.w90_create_kmesh(data, self.istdout, self.istderr),
                    "w90_create_kmesh")

        kmesh_info = data.kmesh_info
        nntot = int(kmesh_info.nntot)
        nnlist = np.zeros((self.num_kpts, self.num_nnmax), dtype=np.int32, order='F')
        nncell = np.zeros((3, self.num_kpts, self.num_nnmax), dtype=np.int32, order='F')
        nnlist[:, :nntot] = np.asarray(kmesh_info.nnlist)[:, :nntot]
        nncell[:, :, :nntot] = np.asarray(kmesh_info.nncell)[:, :, :nntot]

        # projections
        num_proj = max(int(data.num_proj), 1)
        proj_site = np.zeros((3, num_proj), dtype=np.double, order='F')
        proj_l = np.zeros((num_proj), dtype=np.int32, order='F')
        proj_m = np.zeros((num_proj), dtype=np.int32, order='F')
        proj_s = np.zeros((num_proj), dtype=np.int32, order='F')
        proj_radial = np.zeros((num_proj), dtype=np.int32, order='F')
        proj_x = np.zeros((3, num_proj), dtype=np.double, order='F')
        proj_z = np.zeros((3, num_proj), dtype=np.double, order='F')
        proj_s_qaxis = np.zeros((3, num_proj), dtype=np.double, order='F')
        proj_zona = np.zeros((num_proj), dtype=np.double, order='F')
        if data.num_proj > 0:
            self._check(lib.w90_get_proj(data, np.zeros((), dtype=np.int32), proj_site,
                                         proj_l, proj_m, proj_s, proj_radial, proj_x,
                                         proj_z, proj_s_qaxis, proj_zona, self.istdout,
                                         self.istderr), "w90_get_proj")

        # save to self
        self.w90_data = data
        self.nntot = nntot
        self.nnlist = nnlist
        self.nncell = nncell
        self.num_bands = num_bands
        self.proj_site = proj_site
        self.proj_l = proj_l
        self.proj_m = proj_m
        self.proj_radial = proj_radial
        self.proj_z = proj_z
        self.proj_x = proj_x
        self.proj_zona = proj_zona
        self.exclude_bands = exclude_bands
        self.band_included_list = [i for i in range(self.num_bands_tot)
                                   if (i + 1) not in self.exclude_bands] # not standard
        self.proj_s = proj_s
        self.proj_s_qaxis = proj_s_qaxis
        logger.info(self, "Wannier90: setup complete.")

    def get_projectability(self):
        r"""
        Projectability of each included band, p_{nk} = \sum_i |A_{n,i}^{k}|^2,
        as used by the projectability disentanglement. 2D-(n, k).
        """
        return np.einsum('nik, nik -> nk', self.A_matrix.conj(), self.A_matrix).real

    def run(self):
        """
        Execute the wannier90 v4 run:
        disentangle (num_bands > num_wann) or project_overlap, then wannierise.
        """
        logger.info(self, "Wannier90: run start.")
        assert self.w90_data is not None, "call setup() first."
        assert isinstance(self.M_matrix, np.ndarray)
        assert isinstance(self.A_matrix, np.ndarray)
        assert isinstance(self.eigenvalues, np.ndarray)
        lib = wan90.w90_library
        data = self.w90_data
        num_bands, num_wann, num_kpts = self.num_bands, self.num_wann, self.num_kpts

        # NOTE: the library holds pointers to these arrays and modifies them
        # in place (M is slimmed/rotated, u_opt initially holds A),
        # so pass copies and keep them alive.
        m_matrix = np.array(self.M_matrix, dtype=np.complex128, order='F', copy=True)
        u_matrix_opt = np.array(self.A_matrix, dtype=np.complex128, order='F', copy=True)
        u_matrix = np.zeros((num_wann, num_wann, num_kpts), dtype=np.complex128, order='F')
        eigval = np.array(self.eigenvalues, dtype=np.double, order='F', copy=True)
        assert m_matrix.shape == (num_bands, num_bands, self.nntot, num_kpts)
        assert u_matrix_opt.shape == (num_bands, num_wann, num_kpts)
        assert eigval.shape == (num_bands, num_kpts)
        self._w90_arrays = (m_matrix, u_matrix_opt, u_matrix, eigval)

        lib.w90_set_m_local(data, m_matrix)
        lib.w90_set_u_opt(data, u_matrix_opt)
        lib.w90_set_u_matrix(data, u_matrix)
        lib.w90_set_eigval(data, eigval)

        if num_bands > num_wann:
            dis = data.dis_manifold
            if dis.frozen_proj:
                proj = self.get_projectability()
                logger.info(self, "Wannier90: projectability disentanglement, "
                            "projectability in [%.4f, %.4f]", proj.min(), proj.max())
            self._check(lib.w90_disentangle(data, self.istdout, self.istderr),
                        "w90_disentangle")
            lwindow = np.array(dis.lwindow, dtype=bool)
            self.ndimwin = np.array(dis.ndimwin, dtype=np.int32)
            self.nfirstwin = np.array(dis.nfirstwin, dtype=np.int32)
        else:
            self._check(lib.w90_project_overlap(data, self.istdout, self.istderr),
                        "w90_project_overlap")
            lwindow = np.ones((num_bands, num_kpts), dtype=bool)
            self.ndimwin = np.full(num_kpts, num_bands, dtype=np.int32)
            self.nfirstwin = np.ones(num_kpts, dtype=np.int32)
        self._check(lib.w90_wannierise(data, self.istdout, self.istderr), "w90_wannierise")

        wann_centres = np.zeros((3, num_wann), dtype=np.double, order='F')
        wann_spreads = np.zeros((num_wann), dtype=np.double, order='F')
        lib.w90_get_centres(data, wann_centres)
        lib.w90_get_spreads(data, wann_spreads)
        omega = data.omega
        spread = np.array([omega.total, omega.invariant, omega.tilde])

        """
        post processing
        """
        self.U_matrix = np.array(u_matrix, order='F')
        self.U_matrix_opt = np.array(u_matrix_opt, order='F')
        self.lwindow = lwindow
        self.wann_centres = wann_centres
        self.wann_spreads = wann_spreads
        self.spread = spread
        logger.info(self, "Wannier90: run complete.")

    def eval_lo_spread(self, C_mo_lo, M_matrix=None):
        r"""
        Wannier centres and spread functional of the given orbitals (no optimisation).
        Requires setup() (and M_matrix, or it is computed here).

        Args:
            C_mo_lo: (nkpts, num_bands, num_wann), the orbitals in the basis of the
                     included bands, |w_nk> = \sum_m |psi_mk> C_mo_lo[k, m, n].

        Returns:
            dict: centres (num_wann, 3) and spreads (num_wann) in Angstrom (^2),
                  omega_total, omega_I, omega_D, omega_OD, omega_tilde in Angstrom^2.
        """
        assert self.w90_data is not None, "call setup() first."
        lib = wan90.w90_library
        num_wann, num_kpts, nntot = self.num_wann, self.num_kpts, self.nntot
        C_mo_lo = np.asarray(C_mo_lo, dtype=np.complex128)
        assert C_mo_lo.shape == (num_kpts, self.num_bands, num_wann)
        dev = max(abs(C_mo_lo[k].conj().T @ C_mo_lo[k] - np.eye(num_wann)).max()
                  for k in range(num_kpts))
        if dev > 1e-6:
            logger.warn(self, "Wannier90: orbitals are not orthonormal within the included "
                        "bands (max dev = %.2e), the spreads are not well defined.", dev)
        if M_matrix is None:
            M_matrix = self.get_M_mat()

        # M in the subspace: M_w(k, b) = C(k)^H M(k, b) C(k+b)
        nnlist = np.array(self.nnlist[:, :nntot] - 1)
        m_matrix = np.zeros((num_wann, num_wann, nntot, num_kpts), dtype=np.complex128,
                            order='F')
        for b in range(nntot):
            Mb = C_mo_lo.conj().transpose(0, 2, 1) @ M_matrix[:, :, b, :].transpose(2, 0, 1) \
                    @ C_mo_lo[nnlist[:, b]]
            m_matrix[:, :, b, :] = Mb.transpose(1, 2, 0)

        # a second library object with num_bands = num_wann and num_iter = 0,
        # so that wannierise only evaluates the spread of the input gauge.
        _skip = ('num_iter', 'exclude_bands', 'dis_')
        lines = [ll for ll in (self.keywords or '').splitlines()
                 if not ll.strip().lower().startswith(_skip)]
        keywords, num_bands_tot = self.keywords, self.num_bands_tot
        self.keywords = '\n'.join(lines + ['num_iter = 0', ''])
        self.num_bands_tot = num_wann
        try:
            self.make_win()
            data = self._read_input()
        finally:
            self.keywords, self.num_bands_tot = keywords, num_bands_tot
        assert data.num_bands == num_wann
        wan90.w90_library_extra.set_kpoint_distribution(
            data, np.zeros(num_kpts, dtype=np.int32), self.istdout, self.istderr)
        self._check(lib.w90_create_kmesh(data, self.istdout, self.istderr), "w90_create_kmesh")
        assert (np.asarray(data.kmesh_info.nnlist)[:, :nntot] == self.nnlist[:, :nntot]).all()

        u_matrix_opt = np.zeros((num_wann, num_wann, num_kpts), dtype=np.complex128, order='F')
        u_matrix_opt[:] = np.eye(num_wann)[:, :, None]
        u_matrix = np.zeros((num_wann, num_wann, num_kpts), dtype=np.complex128, order='F')
        arrays = (m_matrix.copy(order='F'), u_matrix_opt, u_matrix)
        lib.w90_set_m_local(data, arrays[0])
        lib.w90_set_u_opt(data, u_matrix_opt)
        lib.w90_set_u_matrix(data, u_matrix)
        self._check(lib.w90_project_overlap(data, self.istdout, self.istderr),
                    "w90_project_overlap")
        self._check(lib.w90_wannierise(data, self.istdout, self.istderr), "w90_wannierise")

        centres = np.zeros((3, num_wann), dtype=np.double, order='F')
        spreads = np.zeros((num_wann), dtype=np.double, order='F')
        lib.w90_get_centres(data, centres)
        lib.w90_get_spreads(data, spreads)
        omega = data.omega
        # Omega_OD = 1/N_k \sum_{k,b} w_b \sum_{m != n} |M_mn(k, b)|^2
        wb = np.asarray(data.kmesh_info.wb)[:nntot]
        m2 = abs(m_matrix)**2
        omega_od = np.einsum('b,mnbk->', wb, m2 - np.einsum('mmbk->mbk', m2)[:, None]
                             * np.eye(num_wann)[:, :, None, None]) / num_kpts
        return dict(centres=np.array(centres.T), spreads=np.array(spreads),
                    omega_total=omega.total, omega_I=omega.invariant,
                    omega_tilde=omega.tilde, omega_OD=omega_od,
                    omega_D=omega.tilde - omega_od)

    def get_C_mo_lo(self):
        """
        U matrices from the included bands to the Wannier orbitals,
        (nkpts, num_bands, num_wann), taking care of lwindow.
        """
        C_mo_lo = np.zeros((self.num_kpts, self.num_bands, self.num_wann),
                           dtype=np.complex128)
        for k in range(self.num_kpts):
            ndim = self.ndimwin[k]
            C_mo_lo[k][self.lwindow[:, k]] = \
                    self.U_matrix_opt[:ndim, :, k] @ self.U_matrix[:, :, k]
        return C_mo_lo


if __name__ == '__main__':
    from pyscf.pbc import gto as pgto
    from pyscf.pbc import scf as pscf

    cell = pgto.Cell()
    cell.atom = '''
    Si  0.0000  0.0000  0.0000
    Si  1.3575  1.3575  1.3575
    '''
    cell.a = [[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]
    cell.basis = 'gth-dzvp'
    cell.pseudo = 'gth-pade'
    cell.verbose = 4
    cell.build()

    kmesh = [3, 3, 3]
    kmf = pscf.KRKS(cell, cell.make_kpts(kmesh)).density_fit()
    kmf.xc = 'lda'
    kmf.kernel()

    # Projectability-disentangled sp3 orbitals
    num_wann = 8
    keywords = """
    num_iter = 200
    dis_num_iter = 200
    begin projections
    Si:sp3
    end projections
    dis_froz_proj = .true.
    dis_proj_min = 0.01
    dis_proj_max = 0.95
    """
    w90 = W90(kmf, kmesh, num_wann, other_keywords=keywords)
    w90.kernel()
    print(w90.wann_centres.T)
    print(w90.wann_spreads)
