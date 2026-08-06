#! /usr/bin/env python

"""
Selected Columns of Density Matrix (SCDM) for localization.
Ref: Multiscale Model Simul., 16, 1392, 2018.
"""
# ZHC TODO add prune using dm.
# ZHC TODO: different kpts / spin has different number of MO?

import numpy as np
import scipy as sp
import scipy.linalg as la

from pyscf import lib
from pyscf import lo
from pyscf import dft
from pyscf.pbc import gto as pgto
from pyscf.pbc import dft as pdft
from pyscf.pbc.lib import kpts_helper
from pyscf.tools import mo_mapping
from pyscf.lib import logger

IMAG_DISCARD_TOL = 1e-7


def inv1pexp(x):
    return np.where(x < 0., 1. / (1. + np.exp(x)), np.exp(-x) / (1. + np.exp(-x)))


def max_abs(x):
    """
    Equivalent to np.max(np.abs(x)), but faster.
    """
    if np.iscomplexobj(x):
        res = np.abs(x).max()
    else:
        res = max(np.max(x), abs(np.min(x)))
    return res


def cartesian_prod(arrays, out=None, order='C'):
    """
    Generate a cartesian product of input arrays.
    Support different order.
    """
    arrays = [np.asarray(x) for x in arrays]
    dtype = np.result_type(*arrays)
    nd = len(arrays)
    dims = [nd] + [len(x) for x in arrays]
    out = np.ndarray(dims, dtype, buffer=out)

    shape = [-1] + [1] * nd
    for i, arr in enumerate(arrays):
        out[i] = arr.reshape(shape[:nd-i])
    return out.reshape((nd, -1), order=order).T



def get_grid_uniform_mol(mol, mesh=[201, 201, 201], box=np.eye(3)*20.0,
                         origin=None, **kwargs):
    """
    Generate a cubic grid for molecule.
    """
    logger.debug(mol, "uniform grid mesh: %s", mesh)
    if origin is None:
        origin = -np.array(box) * 0.5
    origin = origin.sum(axis=0)
    mesh = np.asarray(mesh)
    ngrids = np.prod(mesh)
    qv = lib.cartesian_prod([np.arange(x) for x in mesh])
    a_frac = np.einsum('i, ij -> ij', 1./mesh, box)
    coords = np.dot(qv, a_frac) + origin
    weights = np.empty(ngrids)
    weights[:] = abs(np.linalg.det(box)) / float(ngrids)
    return coords, weights

def get_grid_becke_mol(mol, level=5, **kwargs):
    """
    Generate a becke grid for molecule. level is the same as pyscf.
    """
    logger.debug(mol, "becke grid level: %s", level)
    grids = dft.gen_grid.Grids(mol)
    grids.level = level
    grids.build(with_non0tab=True)
    return grids.coords, grids.weights

def get_uniform_grids(cell, mesh=None, order='C', **kwargs):
    """
    Generate a uniform real-space grid consistent w/ samp thm; see MH (3.19).
    Support different order.
    """
    if mesh is None: mesh = cell.mesh
    if 'gs' in kwargs:
        logger.warn(cell, 'cell.gs is deprecated.  It is replaced by cell.mesh,'
                    'the number of PWs (=2*gs+1) along each direction.')
        mesh = [2*n+1 for n in kwargs['gs']]
    mesh = np.asarray(mesh, dtype=np.double)
    qv = cartesian_prod([np.arange(x) for x in mesh], order=order)
    a_frac = np.multiply((1.0 / mesh)[:, None], cell.lattice_vectors())
    coords = np.dot(qv, a_frac)
    return coords

def get_uniform_weights(cell, mesh):
    ngrids = np.prod(mesh)
    weights = np.empty(ngrids)
    weights[:] = cell.vol / ngrids
    return weights

def get_grid_uniform_cell(cell, mesh=None, order='C', **kwargs):
    """
    Generate a uniform grid for cell.
    """
    if order == 'C':
        # ZHC FIXME prune the grids based on density,
        # the current grids are very expensive!
        logger.debug(cell, "uniform grid mesh: %s", mesh)
        if mesh is None:
            grids = pdft.gen_grid.UniformGrids(cell)
        else:
            pcell = cell.copy()
            pcell.mesh = mesh
            grids = pdft.gen_grid.UniformGrids(pcell)
        # ZHC NOTE with_non0tab should be True?
        grids.build(with_non0tab=True)
        return grids.coords, grids.weights
    elif order == 'F':
        assert mesh is not None
        coords = get_uniform_grids(cell, mesh=mesh, order=order, **kwargs)
        weights = get_uniform_weights(cell, mesh)
        return coords, weights
    else:
        raise ValueError

def get_grid_becke_cell(cell, level=5, **kwargs):
    """
    Generate a becke grid for cell. level is the same as pyscf.
    """
    logger.debug(cell, "becke grid level: %s", level)
    grids = pdft.gen_grid.BeckeGrids(cell)
    grids.level = level
    grids.build(with_non0tab=True)
    return grids.coords, grids.weights

def scdm_model(mo_coeff, return_C_mo_lo=False, **kwargs):
    """
    SCDM for model (orbitals are already in site basis).
    """
    mo_coeff = np.asarray(mo_coeff)
    if mo_coeff.ndim == 2:
        mo_coeff = mo_coeff[np.newaxis]
    spin, nao, nmo = mo_coeff.shape

    # SCDM
    nlo = nmo
    C_mo_lo = np.zeros((spin, nmo, nlo), dtype=mo_coeff.dtype)
    C_ao_lo = np.zeros((spin, nao, nlo), dtype=mo_coeff.dtype)
    for s in range(spin):
        mo_g = mo_coeff[s]
        #logger.debug(1, "SCDM: ovlp of mo_g (spin %s):\n%s",
        #          s, mo_g.conj().T.dot(mo_g))
        psiT = mo_g.conj().T
        Q, R, perm = la.qr(psiT, pivoting=True)
        if kwargs.get("cholesky", False): # Cholesky-QR
            C_mo_lo[s] = Q
        else: # Lowdin
            C_mo_lo[s] = lo.vec_lowdin(psiT[:, perm[:nlo]])

        sorted_idx = mo_mapping.mo_1to1map(C_mo_lo[s])
        C_mo_lo[s] = C_mo_lo[s][:, sorted_idx]

        C_ao_lo[s] = mo_coeff[s].dot(C_mo_lo[s])

    if max_abs(C_ao_lo.imag) < IMAG_DISCARD_TOL:
        C_ao_lo = C_ao_lo.real
    if return_C_mo_lo:
        return C_ao_lo, C_mo_lo
    else:
        return C_ao_lo

def scdm_mol(mol, mo_coeff, grid='becke', return_C_mo_lo=False, **kwargs):
    """
    SCDM for molecule.
    """
    # AO on grids
    if grid.strip().startswith(('b', 'B')):
        coords, weights = get_grid_becke_mol(mol, **kwargs)
    else:
        coords, weights = get_grid_uniform_mol(mol, **kwargs)
    ao_g = mol.eval_gto('GTOval_sph', coords) * \
           np.sqrt(weights[:, None].astype(np.complex128))

    # MO on grids
    mo_coeff = np.asarray(mo_coeff)
    if mo_coeff.ndim == 2:
        mo_coeff = mo_coeff[np.newaxis]
    spin, nao, nmo = mo_coeff.shape

    # SCDM
    nlo = nmo
    C_mo_lo = np.zeros((spin, nmo, nlo), dtype=ao_g.dtype)
    C_ao_lo = np.zeros((spin, nao, nlo), dtype=ao_g.dtype)
    for s in range(spin):
        mo_g = np.dot(ao_g, mo_coeff[s])
        logger.debug(mol, "SCDM: ovlp of mo_g (spin %s):\n%s",
                  s, mo_g.conj().T.dot(mo_g))
        psiT = mo_g.conj().T
        Q, R, perm = la.qr(psiT, pivoting=True)
        if kwargs.get("cholesky", False): # Cholesky-QR
            C_mo_lo[s] = Q
        else: # Lowdin
            C_mo_lo[s] = lo.vec_lowdin(psiT[:, perm[:nlo]])
        C_ao_lo[s] = mo_coeff[s].dot(C_mo_lo[s])

    if max_abs(C_ao_lo.imag) < IMAG_DISCARD_TOL:
        C_ao_lo = C_ao_lo.real

    if max_abs(C_mo_lo.imag) < IMAG_DISCARD_TOL:
        C_mo_lo = C_mo_lo.real

    if return_C_mo_lo:
        return C_ao_lo, C_mo_lo
    else:
        return C_ao_lo


# add lowdin_avoid_
def lowdin_tight(s):
    from scipy.linalg import eigh
    ''' new basis is |mu> c^{lowdin}_{mu i} '''
    e, v = eigh(s)
    idx = e > 1e-50
    return np.dot(v[:,idx]/np.sqrt(e[idx]), v[:,idx].conj().T)


def vec_lowdin_tight(c, s=1):
    ''' lowdin orth for the metric c.T*s*c and get x, then c*x'''
    #u, w, vh = numpy.linalg.svd(c)
    #return numpy.dot(u, vh)
    # svd is slower than eigh
    from functools import reduce
    return np.dot(c, lowdin_tight(reduce(np.dot, (c.conj().T,s,c))))


def scdm_k(cell, mo_coeff, kpts, grid='becke', return_C_mo_lo=False,
           use_gamma_perm=True, nlo=None, smear_func=None, **kwargs):
    """
    SCDM for k-MO.
    """
    # grids
    if grid.strip().startswith(('b', 'B')):
        coords, weights = get_grid_becke_cell(cell, **kwargs)
    else:
        coords, weights = get_grid_uniform_cell(cell, **kwargs)
    weights_factor = np.sqrt(weights[:, None].astype(np.complex128))

    mo_coeff = np.asarray(mo_coeff)
    if mo_coeff.ndim == 3:
        mo_coeff = mo_coeff[np.newaxis]
    spin, nkpts, nao, nmo = mo_coeff.shape

    # SCDM
    if nlo is None:
        nlo = nmo
    C_mo_lo = np.zeros((spin, nkpts, nmo, nlo), dtype=np.complex128)
    C_ao_lo = np.zeros((spin, nkpts, nao, nlo), dtype=np.complex128)

    perm_spin = []
    if use_gamma_perm:
        print('Calculate the permutation of the Gamma point and use it for all kpts.')
        # find the kpts that is closest to Gamma point, and calculate the permutation.
        k_gamma = np.argmin(np.linalg.norm(kpts, axis=1))
        ao_g = pdft.numint.eval_ao(cell, coords, kpt=kpts[k_gamma], deriv=0) * weights_factor
        for s in range(spin):
            mo_g = np.dot(ao_g, mo_coeff[s, k_gamma])
            if smear_func is None:
                psiT = mo_g.conj().T
            else:
                psiT = smear_func[s, k_gamma][:, None] * (mo_g.conj().T)
            Q, R, perm = la.qr(psiT, pivoting=True)
        perm_spin.append(perm)


    for k in range(nkpts):
        # AO on grids
        # (ngrid, nao)
        ao_g = pdft.numint.eval_ao(cell, coords, kpt=kpts[k], deriv=0) * weights_factor
        for s in range(spin):
            mo_g = np.dot(ao_g, mo_coeff[s, k])
            logger.debug(cell, "SCDM: ovlp of mo_g (spin %s, kpts: %s):\n%s",
                      s, k,  mo_g.conj().T.dot(mo_g))
            if smear_func is None:
                psiT = mo_g.conj().T
            else:
                psiT = smear_func[s, k][:, None] * (mo_g.conj().T)
            if use_gamma_perm:
                perm = perm_spin[s]
                if k == 0:
                    logger.info(cell, "SCDM: use_gamma_perm = True")
                    if not kpts_helper.gamma_point(kpts[k]):
                        raise ValueError("use_gamma_perm requires kpts[0] = G")
                    Q, R, perm = la.qr(psiT, pivoting=True)
                    perm_spin.append(perm)
                else:
                    perm = perm_spin[s]
            else:
                Q, R, perm = la.qr(psiT, pivoting=True)

            # C_mo_lo[s, k] = lo.vec_lowdin(psiT[:, perm[:nlo]])
            C_mo_lo[s, k] = vec_lowdin_tight(psiT[:, perm[:nlo]])
            C_ao_lo[s, k] = mo_coeff[s, k].dot(C_mo_lo[s, k])

    if max_abs(C_ao_lo.imag) < IMAG_DISCARD_TOL:
        C_ao_lo = C_ao_lo.real
    if return_C_mo_lo:
        return C_ao_lo, C_mo_lo
    else:
        return C_ao_lo

def smear_func(mo_energy, mu, sigma, method='erfc'):
    """
    Smear function for disentanglement.
    """
    mo_energy = np.asarray(mo_energy)
    if mo_energy.ndim == 2:
        mo_energy = mo_energy[np.newaxis]
    if method == "erfc":
        return smear_func_erfc(mo_energy, mu, sigma)
    elif method == "erf":
        return smear_func_erf(mo_energy, mu, sigma)
    elif method == "gaussian":
        return smear_func_gaussian(mo_energy, mu, sigma)
    elif method == "fermi":
        return smear_func_fermi(mo_energy, mu, sigma)
    else:
        raise ValueError

def smear_func_erfc(mo_energy, mu, sigma):
    return 0.5 * sp.special.erfc((mo_energy-mu) / sigma)

def smear_func_erf(mo_energy, mu, sigma):
    return 0.5 * sp.special.erf((mo_energy-mu) / sigma)

def smear_func_gaussian(mo_energy, mu, sigma):
    return np.exp(-((mo_energy - mu) / sigma)**2)

def smear_func_fermi(mo_energy, mu, sigma):
    return inv1pexp((mo_energy - mu) / sigma)


if __name__ == '__main__':
    from pyscf import gto, scf, tools
    from pyscf.tools import molden
    np.set_printoptions(3, linewidth=1000)
    atom_coords = np.array([[3.17500000, 3.17500000, 3.17500000],
                            [2.54626556, 2.54626556, 2.54626556],
                            [3.80373444, 3.80373444, 2.54626556],
                            [2.54626556, 3.80373444, 3.80373444],
                            [3.80373444, 2.54626556, 3.80373444]]) \
                 - np.array([3.17500000, 3.17500000, 3.17500000])
    mol = gto.Mole()
    mol.build(
        atom=[['C', atom_coords[0]],
                ['H', atom_coords[1]],
                ['H', atom_coords[2]],
                ['H', atom_coords[3]],
                ['H', atom_coords[4]]],
        basis='ccpvdz')

    mf = scf.RHF(mol)
    mf.kernel()

    mo_coeff = mf.mo_coeff
    orb_list = range(1, 5)
    mo = mo_coeff[:, orb_list]

    # SCDM
    C_ao_lo = scdm_mol(mol, mo, grid='B', level=5)[0]
    C_mo_lo = mo.conj().T.dot(mf.get_ovlp()).dot(C_ao_lo)
    molden.from_mo(mol, 'CH4_SCDM.molden', C_ao_lo)

    # Boys and compare cf values
    loc = lo.Boys(mol, mo)
    print("Dipole cf values:")
    print("cf (MO): %s" % loc.cost_function())
    print("cf (SCDM): %s" % loc.cost_function(u=C_mo_lo))

    loc_orb = loc.kernel()
    print("cf (Boys): %s" % loc.cost_function())
    molden.from_mo(mol, 'CH4_Boys.molden', loc_orb)
