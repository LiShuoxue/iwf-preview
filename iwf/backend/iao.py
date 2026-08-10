#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Paul J. Robinson <pjrobinson@ucla.edu>
#
# Modified for k-points sampling, IAO virtuals, cores, smearing by
#         Zhi-Hao Cui <zhcui0408@gmail.com>

"""
Intrinsic Atomic Orbitals (IAOs) and projected atomic orbitals (PAOs).
Ref. Knizia, J. Chem. Theory Comput. 2013, 9, 11, 4834.
     Cui et al. J. Chem. Theory Comput. 2020, 16, 1, 119.
"""

import os
import sys
import copy
from functools import reduce
import collections
from collections.abc import Iterable
import numpy as np
from scipy import linalg as la

from pyscf import gto
from pyscf import lib
from pyscf import __config__
from pyscf.lo.iao import is_ghost_atom
from pyscf.gto.basis import parse_nwchem
from pyscf.lib import logger

from iwf.backend import atom_scf_basis, orth


def mdot(*args):
    """
    Reduced matrix dot.
    """
    return reduce(np.dot, args)


def kdot(a, b):
    """
    Matrix dot with kpoints.
    """
    ka, s1_a, _ = a.shape
    kb, _, s2_b = b.shape
    assert ka == kb
    res = np.zeros((ka, s1_a, s2_b), dtype=np.result_type(a.dtype, b.dtype))
    for k in range(ka):
        np.dot(a[k], b[k], out=res[k])
    return res


# Alternately, use ANO for minao
# orthogonalize iao by orth.lowdin(c.T*mol.intor(ovlp)*c)
MINAO = getattr(__config__, 'lo_iao_minao', 'minao')

LNAMES = {0: "s",
          1: "p",
          2: "d",
          3: "f",
          4: "g",
          5: "h",
          6: "i",
          7: "j"}


def get_labels(cell, minao=MINAO, full_virt=False, B2_labels=None,
               core_labels=None):
    """
    Get labels of all, val and virt.
    """
    mol = cell
    if core_labels is None:
        core_labels = []

    B1_labels = mol.ao_labels()
    if B2_labels is None:
        if full_virt:
            B2_labels = []
        else:
            pmol = reference_mol(mol, minao)
            B2_labels = pmol.ao_labels()

    virt_labels = [label for idx, label in enumerate(B1_labels)
                   if ((label not in B2_labels) and (label not in core_labels))]
    nB1 = len(B1_labels)
    nB2 = len(B2_labels)
    nvirt = len(virt_labels)
    ncore = len(core_labels)
    if nB2 + nvirt + ncore != nB1:
        raise ValueError("nB2 (%s) + nvirt (%s) + ncore (%s) != nB1 (%s)"
                         %(nB2, nvirt, ncore, nB1))

    labels = B2_labels + virt_labels
    return labels, B2_labels, virt_labels


def get_idx_each(cell=None, minao=MINAO, full_virt=False, labels=None,
                 B2_labels=None, core_labels=None, kind='atom', symbols=None):
    """
    Get orbital index for all atom / orbital in the cell.

    Args:
        cell: cell
        minao: IAO projection reference
        full_virt: whether to exclude the orbitals with the same names
        B2_labels  : if give, will use as IAO reference labels
        core_labels: if give, will use as core labels
        kind: since the label is formatted as 'id + 'atom' + 'nl' + 'lz',
              support:
              'id atom': id + atom name
              'atom': atom name
              'atom nl': atom name + nl
              'atom nl lz' or 'atom nlm': atom name + nl + lz
              'id atom nl': id + atom name + nl
              'atom l': atom name + l
              'id atom l': id + atom name + l
              'all': keys includes all information

    Returns:
        dic: {atom name: index}. OrderedDict, not include core indices.
    """
    kind = kind.lower()
    if labels is None:
        labels = get_labels(cell, minao=minao, full_virt=full_virt,
                            B2_labels=B2_labels, core_labels=core_labels)[0]
    if symbols is None:
        if kind == 'id atom':
            symbols = [" ".join(label.split()[:2]) for label in labels]
        elif kind == 'atom':
            symbols = [label.split()[1] for label in labels]
        elif kind == 'atom nl':
            symbols = []
            for label in labels:
                ele = label.split()
                for i, x in enumerate(ele[2], start=1):
                    if not x.isdigit():
                        symbols.append(ele[1] + " " + ele[2][:i])
                        break
        elif kind == 'atom nl lz' or kind == 'atom nlm':
            symbols = [" ".join(label.split()[1:]) for label in labels]
        elif kind == 'id atom nl':
            symbols = []
            for label in labels:
                ele = label.split()
                for i, x in enumerate(ele[2], start=1):
                    if not x.isdigit():
                        symbols.append(ele[0] + " " + ele[1] + " " + ele[2][:i])
                        break
        elif kind == 'atom l':
            symbols = []
            for label in labels:
                ele = label.split()
                for x in ele[2]:
                    if not x.isdigit():
                        symbols.append(ele[1] + " " + x)
                        break
        elif kind == 'atom lm':
            symbols = []
            for label in labels:
                ele = label.split()
                for i, x in enumerate(ele[2]):
                    if not x.isdigit():
                        symbols.append(ele[1] + " " + ele[2][i:])
                        break
        elif kind == 'id atom l':
            symbols = []
            for label in labels:
                ele = label.split()
                for x in ele[2]:
                    if not x.isdigit():
                        symbols.append(ele[0] + " " + ele[1] + " " + x)
                        break
        elif kind == 'all':
            symbols = labels
        else:
            raise ValueError

    dic = collections.OrderedDict()
    for i, lab in enumerate(symbols):
        if lab in dic:
            dic[lab].append(i)
        else:
            dic[lab] = [i]
    return dic


def get_idx_each_atom(cell=None, minao=MINAO, full_virt=False, B2_labels=None,
                      core_labels=None, kind='atom'):
    return get_idx_each(cell=cell, minao=minao, full_virt=full_virt,
                        B2_labels=B2_labels, core_labels=core_labels, kind=kind)


def get_idx_each_orbital(cell=None, minao=MINAO, full_virt=False, B2_labels=None,
                         core_labels=None, kind='atom nl'):
    return get_idx_each(cell=cell, minao=minao, full_virt=full_virt,
                        B2_labels=B2_labels, core_labels=core_labels, kind=kind)


def get_idx_to_ao_labels(cell, minao=MINAO, labels=None):
    if labels is None:
        labels = get_labels(cell, minao)[0]
    #atom_ids = [int(lab.split()[0]) for lab in labels]
    #idx = np.argsort(atom_ids, kind='mergesort')
    ao_labels = cell.ao_labels()
    vals = [ao_labels.index(lab) for lab in labels]
    idx  = np.argsort(vals, kind='mergesort')
    return idx


def get_idx(labels, atom_num, offset=0):
    """
    Get orbital index for a given atom_num.
    atom_num: a list of atoms, e.g. atom_num=[1, 2] (the 2nd and 3rd atom).
    """
    if not isinstance(atom_num, Iterable):
        atom_num = [atom_num]
    atom_num = [str(atom) for atom in atom_num]
    idx = [i+offset for i, label in enumerate(labels) if label.split()[0] in atom_num]
    #if log.__verbose() >= log.Level["DEBUG1"]:
    #    log.debug(1, "Get idx for atom_num = %s", atom_num)
    #    log.debug(1, "idx   label")
    #    for i in idx:
    #        log.debug(1, " %s    %s", i, labels[i-offset])
    return idx


def reference_mol(mol, minao=MINAO):
    """
    Create a molecule which uses reference minimal basis.
    """
    pmol = mol.copy()
    atoms = gto.format_atom(pmol.atom, unit=1)
    # remove ghost atoms
    pmol.atom = [atom for atom in atoms if not is_ghost_atom(atom[0])]
    if len(pmol.atom) != len(atoms):
        logger.info(mol, 'Ghost atoms found in system. '
                    'Current IAO does not support ghost atoms. '
                    'They are removed from IAO reference basis.')
    if getattr(pmol, 'rcut', None) is not None:
        pmol.rcut = None
    with lib.temporary_env(sys, stderr=open(os.devnull, "w")):
        pmol.build(False, False, basis=minao)
    return pmol


def get_core_shells(pmol_core):
    """
    Get number of core shells for each element and angular momentum.

    Args:
        pmol_core: reference core cell.

    Returns
        core_shells: a dict:
        key: element. value: a dict (key: anglular momentum, value: number of shells).
    """
    core_shells = {}
    for ele in pmol_core._basis.keys():
        for i, atm in enumerate(pmol_core._atom):
            if atm[0] == ele:
                dic = {}
                for b in pmol_core._bas:
                    if b[0] == i:
                        if LNAMES[b[1]] in dic:
                            dic[LNAMES[b[1]]] += b[3]
                        else:
                            dic[LNAMES[b[1]]] = b[3]
                break
        else:
            raise ValueError
        core_shells[ele] = dic
    return core_shells


def build_pmol_core_val(cell, basis_core, basis_val):
    """
    Build reference cell (pmol) for core and valence basis.
    and adjust the atomic labels for pmol_val.

    Args:
        cell: cell object.
        basis_core: basis for core orbitals.
        basis_val: basis for valence orbitals.

    Returns:
        pmol_core: core reference cell.
        pmol_val: valence reference cell.
    """
    def remove_unexist_basis(pmol, basis):
        # ZHC NOTE FIXME if no basis is found, need to filter out
        basis_new = copy.deepcopy(pmol._basis)
        for ele, bas in pmol._basis.items():
            string = parse_nwchem.search_seg(basis, ele)
            if len(string) == 0:
                del basis_new[ele]
        pmol.basis = basis_new
        with lib.temporary_env(sys, stderr=open(os.devnull, "w")):
            pmol.build(dump_input=False)
        return pmol

    pmol_val = reference_mol(cell, minao=basis_val)
    if isinstance(basis_val, str) and os.path.exists(basis_val):
        pmol_val = remove_unexist_basis(pmol_val, basis_val)

    if basis_core is None:
        pmol_core = None
    else:
        pmol_core = reference_mol(cell, minao=basis_core)
        # ZHC NOTE FIXME if no basis is found, need to filter out
        if isinstance(basis_core, str) and os.path.exists(basis_core):
            pmol_core = remove_unexist_basis(pmol_core, basis_core)

        # rewrite the valence cell ao labels.
        core_shells = get_core_shells(pmol_core)
        val_labels = pmol_val.ao_labels()
        val_labels_2 = pmol_val.ao_labels(fmt=False)
        for i in range(len(val_labels)):
            lab_2 = list(val_labels_2[i])
            lab = val_labels[i]
            lab_sp = lab.split()

            nspace = len(lab) - len(" ".join(lab_sp))
            # special treatment for 4f 0
            idx, ele, orb = lab_sp[:3]
            for j, x in enumerate(orb):
                if not x.isdigit():
                    old_n = int(orb[:j])
                    lm = orb[j:]
                    break
            old_n_2 = int(lab_2[2][:-1])
            l = lab_2[2][-1]
            assert old_n_2 == old_n

            if ele in core_shells:
                lab_sp[2] = str(old_n + core_shells[ele].get(lm[0],  0)) + lm
                lab_2[2] = str(old_n + core_shells[ele].get(lm[0],  0)) + l
            else:
                lab_sp[2] = str(old_n) + lm
            val_labels[i] = (" ".join(lab_sp)) + (" " * nspace)
            val_labels_2[i] = tuple(lab_2)

        def ao_labels(fmt=True):
            if fmt:
                res = val_labels
            else:
                res = val_labels_2
            return res
        pmol_val.ao_labels = ao_labels
    return pmol_core, pmol_val


def pre_step_iao(cell, minao, core_dic, val_dic, pmol_core, pmol_val, save_minao=None):
    """
    Pre set up of iao references.
    The essential task is to determine the pmol_core and pmol_val.
    """
    # ZHC NOTE no need to check minao, core_dic, val_dic if pmol are given.
    if pmol_val is None and pmol_core is None:
        if minao is None:
            if cell.has_ecp():
                minao_pre = "gth-szv-mol-opt-sr"
            else:
                minao_pre = 'minao'
        elif isinstance(minao, str):
            if minao.lower() == 'scf':
                if cell.has_ecp():
                    minao_pre = "gth-szv-mol-opt-sr"
                else:
                    minao_pre = 'minao'
            else:
                minao_pre = minao
        else:
            minao_pre = minao

        mol_minao = reference_mol(cell, minao_pre)

        # define core_dic and val_dic
        if val_dic is None and core_dic is None:
            labs_shell_minao = atom_scf_basis.get_labels_shell(mol_minao)
            val_dic = labs_shell_minao
            core_dic = {}
        elif (val_dic is not None) and (core_dic is not None):
            pass
        else:
            raise ValueError("val_dic and core_dic should be specified together.")

        # define basis_core, basis_val
        if minao == 'scf':
            basis_scf = atom_scf_basis.get_atomic_hf_basis(cell)
            basis_core, basis_val = atom_scf_basis.split_basis(cell,
                                                               basis_scf,
                                                               core_dic,
                                                               val_dic)
        else:
            basis_core, basis_val = atom_scf_basis.split_basis(mol_minao,
                                                               mol_minao._basis,
                                                               core_dic,
                                                               val_dic)
        # save the IAO reference
        if save_minao:
            atom_scf_basis.write_nwchem(save_minao + "-core.dat", basis_core)
            atom_scf_basis.write_nwchem(save_minao + "-val.dat", basis_val)
            atom_scf_basis.write_nwchem(save_minao + ".dat",
                                        atom_scf_basis.merge_basis(basis_core, basis_val))
    else:
        logger.info(cell, "Use customized pmol_core and pmol_val...")

    # build pmol
    labs_ao = cell.ao_labels()
    if pmol_core is None:
        if pmol_val is None:
            if any(basis_core.values()):
                pmol_core, pmol_val = build_pmol_core_val(cell, basis_core, basis_val)
                labs_core = pmol_core.ao_labels()
                idx_core = [labs_ao.index(lab) for lab in labs_core]
            else:
                labs_core = []
                idx_core = []
                pmol_val = reference_mol(cell, basis_val)
        else: # pmol_val is not None
            labs_core = []
            idx_core = []
    else:
        assert type(pmol_val) is type(pmol_core)
        # ZHC NOTE if pmol_core / pmol_val are basis
        if not hasattr(pmol_core, "ao_labels"):
            pmol_core, pmol_val = build_pmol_core_val(cell, pmol_core, pmol_val)
        labs_core = pmol_core.ao_labels()
        idx_core = [labs_ao.index(lab) for lab in labs_core]

    if not hasattr(pmol_val, "ao_labels"):
        pmol_val = reference_mol(cell, pmol_val)
    labs_val = pmol_val.ao_labels()
    idx_val = [labs_ao.index(lab) for lab in labs_val]

    # ZHC NOTE explicitly convert idx_* to np.array
    idx_core = np.asarray(idx_core, dtype=int)
    idx_val  = np.asarray(idx_val, dtype=int)

    return (pmol_core, pmol_val, idx_core, idx_val, labs_core, labs_val)


def get_iao(s1, s2, s12, mo_coeff, mo_occ, proj_B1=None, tol=1e-18):
    """
    Return the independent IAO construction for each k-point.
    without virtual orbital PAOs.
    """
    mop = mo_coeff if proj_B1 is None else proj_B1.conj().T @ s1 @ mo_coeff
    s1p = np.array(s1) if proj_B1 is None else proj_B1.conj().T @ s1 @ proj_B1
    s12p = np.array(s12) if proj_B1 is None else proj_B1.conj().T @ s12
    s21p = s12p.conj().T
    s1cd, s2cd = map(la.cho_factor, (s1p, s2))
    p12 = la.cho_solve(s1cd, s12p)
    A = np.array(p12)
    if mop.size != 0:
        ccs1 = (mop * mo_occ) @ mop.conj().T @ s1p
        # cocc = mop[:, mo_occ > (.5 - tol)]
        cocc = mop[:, mo_occ > tol]     # NOTE needs the case of some 'smearing'.
        ctild = la.cho_solve(s2cd, s21p @ cocc)
        ctild = la.cho_solve(s1cd, s12p @ ctild)
        ctild = orth.orth_cano(ctild, s1p, tol=tol)
        ccs2 = ctild @ ctild.conj().T @ s1p
        A += (ccs1 @ ccs2 * 2. - ccs1 - ccs2) @ p12
        # virt = ((ccs1 @ ccs2 - ccs1 - ccs2) @ p12 + p12)
        # print(f"Virt contribution = {np.linalg.norm(virt)} / {np.linalg.norm(A)}")
    A = A if proj_B1 is None else proj_B1 @ A
    A2 = orth.vec_lowdin(A, s1, tol=tol)
    return A2
