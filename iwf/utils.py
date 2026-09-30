"""
Various helper functions for IWF and other benchamarks.
"""

import numpy as np
from itertools import pairwise
from pyscf.pbc import gto as pgto
from pyscf.data.nist import BOHR


def read_poscar(fname: str = "POSCAR") -> pgto.Cell:
    """
    Read cell structure from a VASP POSCAR file.

    Args:
        fname: file name.

    Returns:
        cell: cell, without build, unit in A.
    """
    with open(fname, "r") as f:
        lines = f.readlines()

        # 1 line scale factor
        line = lines[1].split()
        assert len(line) == 1
        factor = float(line[0])

        # 2-4 line, lattice vector
        a = np.array([np.fromstring(lines[i], dtype=np.double, sep=" ")
                      for i in range(2, 5)]) * factor

        # 5, 6 line, species names and numbers
        sp_names = lines[5].split()
        if all(name.isdigit() for name in sp_names):
            # 5th line can be number of atoms not names.
            sp_nums = np.fromstring(lines[5], dtype=int, sep=" ")
            sp_names = ["X" for i in range(len(sp_nums))]
            line_no = 6
        else:
            sp_nums = np.fromstring(lines[6], dtype=int, sep=" ")
            line_no = 7

        # 7 cartisian or fraction or direct
        line = lines[line_no].split()
        if line[0].startswith(("S", "s")):
            # selective dynamics
            line_no += 1
            line = lines[line_no].split()
        use_cart = line[0].startswith(("C", "K", "c", "k"))
        line_no += 1

        # 8-end, coords
        atom_col = []
        for sp_name, sp_num in zip(sp_names, sp_nums):
            for i in range(sp_num):
                # there may be 4th element for comments or fixation.
                coord = np.array(list(map(float, lines[line_no].split()[:3])))
                if use_cart:
                    coord *= factor
                else:
                    coord = np.dot(coord, a)  # frac2real(a, coord)
                atom_col.append((sp_name, coord))
                line_no += 1

        return pgto.Cell().set(a=a, unit='A', atom=atom_col)


def get_wigner_seitz_supercell(
        lattice_vector, kmesh,
        ws_search_size=[2, 2, 2], ws_distance_tol=1e-6):
    real_metric = (lattice_vector.T @ lattice_vector) * BOHR ** 2
    dist_dim = np.prod(2 * (np.asarray(ws_search_size) + 1) + 1)
    ndegen, irvec = [], []
    mp_grid = np.asarray(kmesh)
    x, y, z = np.meshgrid(*(np.arange(-s * g, s * g + 1) for s, g in zip(ws_search_size, mp_grid)))
    n_list = np.vstack([z.flatten('F'), x.flatten('F'), y.flatten('F')]).T
    x, y, z = np.meshgrid(*(np.arange(-s - 1, s + 2) for s in ws_search_size))
    i_list = np.vstack([z.flatten('F'), x.flatten('F'), y.flatten('F')]).T
    nrpts = 0
    for n in n_list:
        ndiff = n - i_list * mp_grid
        dist = (ndiff @ real_metric @ ndiff.T).diagonal()
        dist_min = dist.min()
        if abs(dist[(dist_dim + 1) // 2 - 1] - dist_min) < ws_distance_tol ** 2:
            temp = sum(1 for i in range(dist_dim) if abs(dist[i] - dist_min) < ws_distance_tol ** 2)
            ndegen.append(temp)
            irvec.append(n.tolist())
            if (n ** 2).sum() < 1.e-10: rpt_origin = nrpts
            nrpts += 1
    irvec, ndegen = np.asarray(irvec), np.asarray(ndegen)
    assert np.sum(1 / ndegen) - np.prod(mp_grid) < 1e-8, "Error in finding Wigner-Seitz points!!!"
    return ndegen, irvec, rpt_origin


def get_hR(hk, kpts_rel, Rs):
    nkpts = len(kpts_rel)
    center = np.where((np.asarray(Rs) ** 2).sum(axis=1) < 1e-10)[0]
    center = center[0] if center.shape[0] == 1 else 0
    phase = np.exp(-1j * 2 * np.pi * np.dot(Rs, kpts_rel.T))
    return np.einsum('k,kst,Rk->Rst', phase[center], hk, phase.conj()) / nkpts


def get_interpolate_hk(hk, cell, kmesh, kpts_abs_band):
    """
    Interpolation method extracted from the following link for more details:
    https://github.com/ZhuGroup-Yale/fcdmft/tree/main/examples/interpolation
    """
    ndegen, irvec, _ = get_wigner_seitz_supercell(cell.lattice_vectors(), kmesh)
    hR = get_hR(hk, cell.get_scaled_kpts(cell.make_kpts(kmesh)), irvec)
    phase = np.exp(-1j * 2 * np.pi * np.dot(irvec, cell.get_scaled_kpts(kpts_abs_band).T))
    return np.einsum("R,Rst,Rk->kst", 1. / ndegen, hR, phase)


def get_w90_projection_keywords(
        cell, labels, special_kpoints=None,
        kpath=None, extra_keywords: dict={}
        ):
    """
    Generate Wannier90 projection keywords based on the given labels and energy window.
    """
    # NOTE: Map the PySCF labels to Wannier90 labels
    LM_W90_MAP = {'dz^2': 'dz2',
                'f-3': 'fy(3x2-y2)',
                'f-2': 'fxyz',
                'f-1': 'fyz2',
                'f0': 'fz3',
                'f1': 'fxz2',
                'f2': 'fz(x2-3y2)',
                'f3': 'fx(x2-3y2)'
                }

    frac_coords = cell.atom_coords() @ np.linalg.inv(cell.lattice_vectors())
    proj_strs = []

    for label in labels:
        atom_idx_s, _symbol, tag = label.split()
        atom_idx = int(atom_idx_s)
        lm = tag.split(":")[1] if ":" in tag else tag[1:]
        lm_w90 = LM_W90_MAP.get(lm, lm)
        frac = frac_coords[atom_idx]
        proj_strs.append('f=%.8f,%.8f,%.8f:%s' % (frac[0], frac[1], frac[2], lm_w90))
    kws = ['begin projections', *proj_strs, 'end projections', '']

    if special_kpoints is not None:
        assert kpath is not None, "kpath must be provided if special_kpoints is given"
        kws.append("begin kpoint_path")
        for ll1, ll2 in pairwise(kpath):
            kws.append("  %s %.8f %.8f %.8f  %s %.8f %.8f %.8f" % (
                ll1, *special_kpoints[ll1], ll2, *special_kpoints[ll2]))
        kws.append("end kpoint_path")
        kws.append("")

    for k, v in extra_keywords.items():
        if isinstance(v, list):
            kws.append(f"begin {k}")
            for vv in v:
                kws.append(str(vv))
            kws.append(f"end {k}")
        elif isinstance(v, bool):
            kws.append(f"{k} = {'.true.' if v else '.false.'}")
        else:
            kws.append(f"{k} = {v}")
    return '\n'.join(kws) + '\n'
