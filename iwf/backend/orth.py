"""
Tools for orbital orthogonalization in pyscf.
"""

import numpy as np
from scipy import linalg as la
from pyscf import lo


def lowdin_k(mf, method='meta_lowdin', s=None, pre_orth_ao=lo.orth.REF_BASIS):
    """
    Meta Lowdin orbitals with k point sampling.
    Copied from libdmet.lo.lowdin.lowdin_k

    Args:
        kmf: kscf
        method : str
            One of
            | lowdin : Symmetric orthogonalization
            | meta-lowdin : Lowdin orth within core, valence, virtual space
                            separately (JCTC, 10, 3784)
            | NAO:
        s: overlap matrix, shape (nkpts, nao, nao)
        pre_orth_ao: if None, use ANO or MINAO as reference basis

    Returns:
        C_ao_lo: shape (nkpts, nao, nlo)
    """
    from pyscf import scf
    from pyscf.pbc.scf import khf, kuhf

    if isinstance(mf, scf.hf.SCF):  # mol
        if not isinstance(mf, khf.KSCF):
            C_lowdin = lo.orth_ao(mf, method, s=s, pre_orth_ao=pre_orth_ao)
            if isinstance(mf, scf.uhf.UHF):  # unrestricted
                C_lowdin = (C_lowdin, C_lowdin)
            return np.asarray(C_lowdin)
    else:
        raise ValueError("Unknown mf object: %s" % type(mf))

    if s is None:
        s1e = mf.get_ovlp()
    else:
        s1e = s
    nkpts = len(mf.kpts)
    assert nkpts == len(s1e)
    C_lowdin = [lo.orth_ao(mf, method, s=s1e[k], pre_orth_ao=pre_orth_ao)
                for k in range(nkpts)]
    is_uhf = isinstance(mf, scf.uhf.UHF) or isinstance(mf, kuhf.KUHF)
    if is_uhf:
        C_lowdin = (C_lowdin, C_lowdin)
    return np.asarray(C_lowdin)


def _lowdin(s, tol=1e-18):
    ''' new basis is |mu> c^{lowdin}_{mu i} '''
    e, v = la.eigh(s)
    idx = e > tol
    return np.dot(v[:,idx]/np.sqrt(e[idx]), v[:,idx].conj().T)


def _cano(s, tol=1e-18):
    e, v = la.eigh(s)
    idx = e > tol
    return v[:, idx] / np.sqrt(e[idx])


def vec_lowdin(c, s, tol=1e-18):
    ''' lowdin orth for the metric c.T*s*c and get x, then c*x'''
    #u, w, vh = numpy.linalg.svd(c)
    #return numpy.dot(u, vh)
    # svd is slower than eigh
    return np.dot(c, _lowdin(c.conj().T @ s @ c, tol=tol))


def orth_cano(c, s, tol=1e-18):
    return np.dot(c, _cano(c.conj().T @ s @ c, tol=tol))
