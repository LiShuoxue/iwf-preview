"""
Implementation of the IWF method.
"""

import os
import h5py
import numpy as np
from scipy import linalg as la
from scipy.optimize import root_scalar

from pyscf import gto, lib, lo
from pyscf.data.nist import HARTREE2EV
from pyscf.lib import logger
from pyscf.pbc import gto as pgto
from pyscf.pbc.tools import pyscf_ase
from pyscf.tools import cubegen

from iwf.utils import get_w90_projection_keywords
from iwf.backend import iao, orth, scdm, pywannier90


def inv1pexp(x):
    return np.where(x < 0., 1. / (1. + np.exp(x)), np.exp(-x) / (1. + np.exp(-x)))


def fsmear(x, mu, sigma):
    fd = inv1pexp((-x + mu) / sigma)
    f1 = inv1pexp((-1 + mu) / sigma)
    f0 = inv1pexp(mu / sigma)
    return (fd - f0) / (f1 - f0)


def get_scaled_character_matrix(C_ao_lo, ovlp, mo_energy, mo_coeff, e0=0.05):
    # sigma: eV
    C_iao_mo = C_ao_lo.conj().swapaxes(-2, -1) @ ovlp @ mo_coeff
    char_mat = np.einsum("...mi,...mj->...ij", C_iao_mo.conj(), C_iao_mo)
    dE = np.abs(mo_energy[..., None] - mo_energy[..., None, :]) * HARTREE2EV
    scale_mat = np.exp(-dE**2 / (2 * e0**2))
    return char_mat * scale_mat


def get_character(C_ao_lo, ovlp, ew, ev, e0=None):
    if e0 is None or np.allclose(e0, 0.0):    # diagonal
        C_iao_mo = C_ao_lo.conj().swapaxes(-2, -1) @ ovlp @ ev
        char_mo = np.sum(C_iao_mo * C_iao_mo.conj(), axis=-2).real
        return dict(char=char_mo, ew=ew, ev=ev)
    else:
        fock = ovlp @ ev @ np.diag(ew) @ ev.conj().T @ ovlp
        char_mat = get_scaled_character_matrix(C_ao_lo, ovlp, ew, ev, e0=e0)
        char_mo, rot_coef = la.eigh(char_mat)
        evrot = ev @ rot_coef
        ewrot = np.diag(evrot.conj().T @ fock @ evrot).real
        return dict(char=char_mo, ew=ewrot, ev=evrot)


def init_cell_ref(cell, kpts, minao, core, val):
    prep_res = iao.pre_step_iao(cell, minao, core, val, None, None)
    pmol_core, pmol_val = prep_res[:2]
    lls_cell, lls_core, lls_val = (x.ao_labels() if x is not None else []
                                   for x in (cell, pmol_core, pmol_val))
    idx_core = np.array([lls_cell.index(l) for l in lls_core])
    idx_val = np.array([lls_cell.index(l) for l in lls_val])

    if isinstance(cell, pgto.Cell):
        s2v = pmol_val.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts) \
            if pmol_val is not None else None
        s12v = pgto.cell.intor_cross('int1e_ovlp', cell, pmol_val, kpts=kpts)

    else:
        s2v = pmol_val.intor('int1e_ovlp', hermi=1) if pmol_val is not None else None
        s12v = gto.mole.intor_cross('int1e_ovlp', cell, pmol_val)
        s2v, s12v = s2v[None], s12v[None]
    return (pmol_val, idx_core, idx_val, s2v, s12v)


def get_smear_occs(char_mo, nocc, sigma=0.05):
    """
    Also used in the character-based SCDM method.
    """
    def _fn(x): return np.sum(fsmear(char_mo, x, sigma)) - nocc
    mu = root_scalar(_fn, bracket=[0., 1.0], method='brentq', xtol=1e-8, x0=0.5).root
    occ = fsmear(char_mo, mu, sigma)
    return occ


def kernel_per_kpt(C_ref, mo_energy, mo_coeff, ovlp_dict, nocc,
        idx_ref=None, sigma=0.05, e0=None, tol=1e-12):
    s1, s2, s12 = map(ovlp_dict.get, ('s1', 's2', 's12'))
    res = get_character(C_ref, s1, mo_energy, mo_coeff, e0=e0)
    occ = get_smear_occs(res['char'], nocc, sigma) * 2.
    _s2 = s2 if idx_ref is None else s2[idx_ref][:, idx_ref]
    _s12 = s12 if idx_ref is None else s12[:, idx_ref]
    C_ao_lo = iao.get_iao(s1, _s2, _s12, mo_coeff, occ, proj_B1=mo_coeff, tol=tol)
    res.update(C_ao_lo=C_ao_lo, occ=occ)
    return res


def _get_MO_mask(ew, energy_window, ncore=0):
    ew_min, ew_max = energy_window[0] / HARTREE2EV, energy_window[1] / HARTREE2EV
    mask = (ew >= ew_min) & (ew <= ew_max)
    if ncore > 0:
        mask[:ncore] = False
    return mask

class IWF(lib.StreamObject):
    def __init__(self,
                 cell: gto.Mole | pgto.Cell,
                 mo_energy,
                 mo_coeff,
                 kpts_abs_or_kmesh=np.array([1, 1, 1]),
                 minao='scf', core={}, val={},
                 ref_method='iao',
                 downfold_labels=[],
                 erange=[-10, 10],  # eV
                 sigma=0.05,
                 e0=None, # eV, for scaled character matrix method
                 verbose=4,
                 ):

        self.verbose = verbose
        self.log = logger.new_logger(self)
        self.cell = cell

        self.kmesh = None
        if kpts_abs_or_kmesh.ndim == 2:
            self.kpts_abs = kpts_abs_or_kmesh
        elif kpts_abs_or_kmesh.ndim == 1:
            self.kmesh = kpts_abs_or_kmesh
            self.kpts_abs = self.cell.make_kpts(kpts_abs_or_kmesh)

        self.ase_obj = pyscf_ase.pyscf_to_ase_atoms(self.cell)
        self.cell_ref = None
        self.idx_core = None
        self.idx_val = None
        self.mo_coeff = mo_coeff
        self.mo_energy = mo_energy
        self._fock = None

        s1 = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=self.kpts_abs)
        self.ovlp_dict = {'s1': s1}
        self.ref_method = ref_method
        self.minao = minao
        self.core = core
        self.val = val
        self.sigma = sigma
        self.e0 = e0
        self.downfold_labels = downfold_labels
        self.idx_model_B1 = []
        self.idx_model_B2 = []
        self.C_ref = None
        self.erange = erange
        self.iwf = None
        self.anal_data = []

    @property
    def mf(self):
        from pyscf.pbc.scf.khf import KRHF
        class MFPseudo(KRHF):
            def __init__(cls):
                cls.cell = self.cell
                cls.mo_energy = self.mo_energy
                cls.mo_coeff = self.mo_coeff
                cls.verbose = self.verbose
            @property
            def kpts(cls):
                return self.kpts_abs
            def get_ovlp(cls): return self.ovlp_dict['s1']
        return MFPseudo()

    @property
    def nao(self): return self.cell.nao_nr()

    @property
    def nwann(self):
        return len(self.downfold_labels)

    @property
    def nkpts(self): return len(self.kpts_abs)

    @property
    def ncore(self): return len(self.idx_core)

    def get_fock(self):
        s1, ew, ev = self.ovlp_dict['s1'], self.mo_energy, self.mo_coeff
        fock_ao = s1 @ ev @ np.array([np.diag(e) for e in ew]) @ ev.conj().swapaxes(-2, -1) @ s1
        return fock_ao

    @property
    def fock(self):
        if self._fock is None: self._fock = self.get_fock()
        return self._fock

    def init_cell_ref(self, minao='scf', core=None, val=None):
        self.cell_ref, self.idx_core, self.idx_val, s2, s12 = init_cell_ref(
            self.cell, self.kpts_abs, minao, core, val)
        self.ovlp_dict.update(s2=s2, s12=s12)
        cell_lbs = list(map(str.strip, self.cell.ao_labels()))
        ref_lbs = list(map(str.strip, self.cell_ref.ao_labels()))
        self.idx_model_B1 = np.array([cell_lbs.index(ll) for ll in self.downfold_labels])
        self.idx_model_B2 = np.array([ref_lbs.index(ll) for ll in self.downfold_labels])

    def get_LO_ref(self):
        self.init_cell_ref(self.minao, self.core, self.val)

        if self.ref_method.lower() in ('meta_lowdin', 'lowdin', 'NAO'):
            mf = self.mf
            s1 = self.ovlp_dict['s1']
            C_ref = np.array([lo.orth_ao(mf, method=self.ref_method.lower(),
                       s=s1[k], pre_orth_ao=lo.orth.REF_BASIS)
                       for k in range(self.nkpts)])
            self.idx_core = np.array([])
            C_ref = C_ref[..., self.idx_model_B1]

        elif self.ref_method.lower() == "b2":
            s1, s12 = map(self.ovlp_dict.get, ('s1', 's12'))
            lls = [x.strip() for x in self.cell_ref.ao_labels()]
            idxs = np.array([lls.index(ll) for ll in self.downfold_labels])
            C_ref = []
            for k in range(self.nkpts):
                s1cd = la.cho_factor(s1[k])
                coef_orth = orth.vec_lowdin(la.cho_solve(s1cd, s12[k][..., idxs]), s1[k])
                C_ref.append(coef_orth)
            C_ref = np.array(C_ref)
            self.idx_core = np.array([])

        self.C_ref = C_ref
        return C_ref

    def kernel(self):
        C = []
        for ik in range(self.nkpts):
            mmk = _get_MO_mask(self.mo_energy[ik], self.erange, ncore=self.ncore)
            ewk, evk = self.mo_energy[ik][mmk], self.mo_coeff[ik][:, mmk]
            res = kernel_per_kpt(
                self.C_ref[ik], ewk, evk,
                {key: val[ik] for key, val in self.ovlp_dict.items()},
                self.nwann, self.idx_model_B2, self.sigma, self.e0, tol=1e-12
            )
            C.append(res['C_ao_lo'])
            self.anal_data.append({k: res[k] for k in ('char', 'ew', 'occ')})
        self.iwf = np.array(C)
        return self.iwf, self.anal_data

    def dump_cube(self, path, C_ao_lo=None, **kwargs):
        C_ao_lo = self.iwf if C_ao_lo is None else C_ao_lo
        os.makedirs(path, exist_ok=True)
        C_R0 = C_ao_lo.mean(axis=0).real
        labels_all = self.downfold_labels
        for i, lbl in enumerate(labels_all):
            fname = path + "/" + "_".join(lbl.split()) + ".cube"
            cubegen.orbital(self.cell, fname, C_R0[..., i], **kwargs)

    # Benchmark
    def get_w90_obj(self):
        assert self.kmesh is not None, "kmesh must be provided to get the Wannier90 object"
        other_kws = get_w90_projection_keywords(self.cell, self.downfold_labels)
        myw90 = pywannier90.W90(self.mf, self.kmesh,
                                num_wann=self.nwann, other_keywords=other_kws)
        return myw90


    def run_w90(self, w90_kwargs: dict={}):
        """
        extra_keywords: List of additional Wannier90 keywords to include in the input file (e.g. for disentanglement or preconditioning).
            If dis_win_min and dis_win_max are not provided, they will be automatically set to the minimum and maximum of the energy_window, respectively.
        """
        kmf = self.mf
        myw90 = self.get_w90_obj()
        labels = self.downfold_labels

        dis_win_min, dis_win_max = map(w90_kwargs.get, ('dis_win_min', 'dis_win_max'))
        if dis_win_min is None: dis_win_min = self.erange[0]
        if dis_win_max is None: dis_win_max = self.erange[1]
        w90_kwargs.update(dis_win_min=dis_win_min, dis_win_max=dis_win_max)

        myw90.keywords = get_w90_projection_keywords(
            self.cell, labels, extra_keywords=w90_kwargs)

        A_matrix, M_matrix = None, None
        if os.path.exists("w90.h5"):
            with h5py.File("w90.h5", "r") as f:
                A_matrix = f['A_matrix'][:]
                M_matrix = f['M_matrix'][:]
        myw90.kernel(A_matrix=A_matrix, M_matrix=M_matrix)

        C_ao_mo = np.array(myw90.mo_coeff)[..., myw90.band_included_list]
        ew_eV = self.mo_energy * HARTREE2EV
        dis_win_idxs = [np.where((ew_eV[k] >= dis_win_min) & (ew_eV[k] <= dis_win_max))[0]
                        for k in range(len(kmf.kpts))]
        ndiswin = [len(idxs) for idxs in dis_win_idxs]
        C_ao_lo = np.array([
            np.einsum('mi,iI,IJ->mJ',
                    C_ao_mo[k][:, dis_win_idxs[k]],
                    myw90.U_matrix_opt[:ndiswin[k], :, k],
                    myw90.U_matrix[:, :, k])
            for k in range(len(kmf.kpts))
        ])

        # Dump to the w90.h5 file
        if not os.path.exists("w90.h5"):
            with h5py.File("w90.h5", "w") as f:
                f['C_ao_mo'], f['C_ao_lo'] = C_ao_mo, C_ao_lo
                for k in ('A_matrix', 'M_matrix', 'U_matrix', 'U_matrix_opt'):
                    f.create_dataset(k, data=getattr(myw90, k))
        return C_ao_lo


    def run_scdm(self, scdm_kwargs: dict={}):
        ew, ev = self.mo_energy, self.mo_coeff
        band_include_list = scdm_kwargs.get('band_include_list', None)
        use_iwf_smear = scdm_kwargs.get('use_iwf_smear', False)
        if band_include_list is not None:
            ew = ew[..., band_include_list]
            ev = ev[..., band_include_list]

        if use_iwf_smear:
            self.get_LO_ref()
            smear_func = []
            for k in range(self.nkpts):
                char_mo = get_character(self.C_ref[k], self.ovlp_dict['s1'][k], mo_coeff=ev[k])
                _sf = get_smear_occs(char_mo, self.nwann, sigma=self.sigma)
                smear_func.append(_sf)
            smear_func = np.array(smear_func)[np.newaxis]
        else:
            smear_func = scdm.smear_func(
                ew,
                mu=scdm_kwargs.get('mu', 0.0) / HARTREE2EV,
                sigma=scdm_kwargs.get('sigma', 0.05) / HARTREE2EV,
            )
        C_ao_lo = scdm.scdm_k(
            self.cell, np.array([ev]), self.kpts_abs,
            grid=scdm_kwargs.get('grid', 'becke'),
            use_gamma_perm=scdm_kwargs.get('use_gamma_perm', True),
            nlo=self.nwann,
            mesh=scdm_kwargs.get('mesh', None),
            level=scdm_kwargs.get('level', 5),
            smear_func=smear_func
        )[0]
        return C_ao_lo
