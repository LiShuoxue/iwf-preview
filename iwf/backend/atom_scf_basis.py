#!/usr/bin/env python

"""
Generate basis set from atomic HF.
"""

import os
import sys
import copy
import numpy as np

from pyscf.data import elements
from pyscf.scf.atom_hf import (AtomSphAverageRHF, AtomHF1e)
from pyscf.gto.basis import parse_nwchem
from pyscf.gto.basis.parse_nwchem import SPDF
from pyscf.data.elements import _std_symbol
from pyscf import lib

from iwf.backend import atom_hf_pp
from pyscf.lib import logger


def get_atomic_hf_basis(mol, atomic_configuration=elements.NRSRHF_CONFIGURATION,
                        tol=1e-7, save_nwchem=None):
    """
    Get atomic HF basis set.

    Args:
        mol: Mole / Cell.
        atomic_configuration: atomic conifg.
        tol: for check non-zero mo coefficients.
        return_nwchem: return a string?

    Returns:
        string: NWChem format basis set for every element in mol
                if return_nwchem == True
                else return a dict
    """
    verbose = max(mol.verbose - 2, 2)
    mol = expand_shared_exp(mol)
    mol.verbose = verbose

    string = ""
    basis_all = {}
    labs_all  = {}
    ele_dic = set()
    ele_dic_std = set()

    elements_all = set(a[0] for a in mol._atom)
    logger.info(mol, 'Spherically averaged atomic HF for %s', elements_all)

    atm_template = copy.copy(mol)
    if hasattr(mol, "a"):
        atm_template.a = copy.copy(mol.a)
    atm_template.charge = 0
    atm_template.symmetry = False  # TODO: enable SO3 symmetry here
    atm_template.atom = atm_template._atom = []
    atm_template.cart = False  # AtomSphAverageRHF does not support cartensian basis

    atm_scf_result = {}
    for ia, a in enumerate(mol._atom):
        element = a[0]
        if element in atm_scf_result:
            continue

        atm = atm_template
        atm._atom = [a]
        atm._atm = mol._atm[ia:ia+1]
        atm._bas = mol._bas[mol._bas[:, 0] == ia].copy()
        atm._ecpbas = mol._ecpbas[mol._ecpbas[:, 0] == ia].copy()
        # Point to the only atom
        atm._bas[:, 0] = 0
        atm._ecpbas[:, 0] = 0

        if element in mol._pseudo:
            atm._pseudo = {element: mol._pseudo.get(element)}
        atm.spin = atm.nelectron % 2

        nao = atm.nao
        # nao == 0 for the case that no basis was assigned to an atom
        if nao == 0 or atm.nelectron == 0:  # GHOST
            mo_occ = mo_energy = np.zeros(nao)
            mo_coeff = np.zeros((nao, nao))
            atm_scf_result[element] = (0, mo_energy, mo_coeff, mo_occ)
        elif atm._pseudo:
            atm.a = None
            if atm.nelectron == 1:
                atm_hf = atom_hf_pp.AtomHF1ePP(atm)
            else:
                atm_hf = atom_hf_pp.AtomSCFPP(atm)
                atm_hf.atomic_configuration = atomic_configuration

            atm_hf.verbose = verbose
            atm_hf.conv_tol = 1e-20
            atm_hf.conv_tol_grad = 1e-20
            atm_hf.run()
            atm_scf_result[element] = (atm_hf.e_tot, atm_hf.mo_energy,
                                       atm_hf.mo_coeff, atm_hf.mo_occ)
        else:
            if atm.nelectron == 1:
                atm_hf = AtomHF1e(atm)
            else:
                atm_hf = AtomSphAverageRHF(atm)
                atm_hf.atomic_configuration = atomic_configuration

            atm_hf.verbose = verbose
            atm_hf.conv_tol = 1e-20
            atm_hf.conv_tol_grad = 1e-20
            atm_hf.run()
            atm_scf_result[element] = (atm_hf.e_tot, atm_hf.mo_energy,
                                       atm_hf.mo_coeff, atm_hf.mo_occ)

        labels_2 = atm.ao_labels(fmt=False)

        mo_coeff = atm_hf.mo_coeff
        bas = atm._basis[element]

        idx_nl = [] # list of first idx of each nl shell, first 3d, 4s, ...
        exists = set()
        for i, lab in enumerate(labels_2):
            if lab[2] in exists:
                continue
            else:
                idx_nl.append(i)
                exists.add(lab[2])
        exists = None

        basis = []
        labs_ele = []
        for j in idx_nl:
            new_bas = []
            mo = mo_coeff[idx_nl, j]
            for i, bas_cur in enumerate(bas):
                l = bas_cur[0]
                exp_list = bas_cur[1:]
                c_list = atm.bas_ctr_coeff(i)
                assert c_list.shape[-1] == 1
                c_list = c_list.ravel()

                for (exp, _), coeff in zip(exp_list, c_list):
                    if abs(coeff * mo[i]) > tol:
                        new_bas.append([l, [exp, coeff * mo[i]]])
            l_list, tmp = list(zip(*new_bas))
            assert len(np.unique(l_list)) == 1
            basis.append([l_list[0], *tmp])
            labs_ele.append(labels_2[j][2])

        basis = parse_nwchem.to_general_contraction(basis)

        # ZHC NOTE nwchem need stardard element symbols
        if _std_symbol(element) not in ele_dic_std:
            string += convert_basis_to_nwchem(element, basis)
            string += "\n"
            ele_dic_std.add(_std_symbol(element))

        if element not in ele_dic:
            basis_all[element] = basis
            labs_all[element] = labs_ele
            ele_dic.add(element)

    if save_nwchem is not None:
        fname = save_nwchem + "-atom-scf.dat"
        with open(fname, 'w') as f:
            f.write(string)

    return basis_all


def get_labels_shell(mol):
    """
    Get labs of shell in mol.

    Args:
        mol: Mole/Cell object.

    Returns:
        labs_shell: dict {ele: shell}
                    e.g., 'Cu': ['3s', '4s', '3p', '3d']
    """
    labs = mol.ao_labels(fmt=False)
    labs_shell = {}
    for lab in labs:
        if lab[1] not in labs_shell:
            labs_shell[lab[1]] = [lab[2]]
        else:
            if lab[2] not in labs_shell[lab[1]]:
                labs_shell[lab[1]].append(lab[2])
    return labs_shell


def _find_lab_and_pop(lbs, l):
    for i, lb in enumerate(lbs):
        if lb[-1] == l.lower():
            lb = lbs.pop(i)
            break
    else:
        lb = None
    return lb, lbs


def _convert_shell_dic_to_list(labs_shell, core_dic):
    core_dic_new = {}
    labs_shell_xcore = {}
    for ele in labs_shell.keys():
        lbs_xcore = copy.deepcopy(labs_shell[ele])
        cores_new = []

        if ele in core_dic or _std_symbol(ele) in core_dic:
            if ele in core_dic:
                cores = core_dic[ele]
            elif _std_symbol(ele) in core_dic:
                # it is possible that the symbol in core_dic is not standard.
                cores = core_dic[_std_symbol(ele)]

            if isinstance(cores, dict):
                for l, num_l in cores.items():
                    for j in range(num_l):
                        lb, lbs_xcore = _find_lab_and_pop(lbs_xcore, l)
                        if lb is None:
                            logger.warn(sys, "%s 's %s th %s shell not found, "
                                        "check your core_dic / val_dic",
                                        ele, j+1, l)
                        else:
                            cores_new.append(lb)
            else:
                cores_new = [v.lower() for v in cores]
                for lb in cores_new:
                    if lb in lbs_xcore:
                        lbs_xcore.remove(lb)

        labs_shell_xcore[ele] = lbs_xcore
        core_dic_new[ele] = cores_new
    return core_dic_new, labs_shell_xcore


def split_basis(mol, basis_dic, core_dic, val_dic):
    """
    Split the basis according to the core and valence dictionary.

    Args:
        mol: Mole object that contains basis information.
        basis_dic: a dict, {ele: basis}
        core_dic: {ele: cores}, cores can be
                  the number of core of shells, like {'s': 1, 'p': 1}
                  or a list ["3s", "3p"].
        val_dic:  {ele: vals}, vals can be dic of numbers, or list.

    Returns:
        basis_core: dict, core basis.
        basis_val:  dict, val  basis.
    """
    mol = expand_shared_exp(mol)
    labs_shell = get_labels_shell(mol)
    labs_core = {}
    labs_val = {}

    basis_core  = {}
    basis_xcore = {}
    basis_val   = {}

    # ZHC NOTE
    # check if mol contains non-standard atomic symbols.
    for ele in labs_shell.keys():
        if ele not in core_dic and _std_symbol(ele) in core_dic:
            logger.warn(mol, "mol contains non-standard symbols: %s \n"
                        "     do not fully match core_dic: %s \n"
                        "     Will convert the standard to the non-standard ones.",
                        labs_shell.keys(), core_dic.keys())
            break
    for ele in labs_shell.keys():
        if ele not in val_dic and _std_symbol(ele) in val_dic:
            logger.warn(mol, "mol contains non-standard symbols: %s \n"
                        "     do not match val_dic: %s \n"
                        "     Will convert the standard to the non-standard ones.",
                        labs_shell.keys(), val_dic.keys())
            break

    core_dic, labs_shell_xcore = _convert_shell_dic_to_list(labs_shell, core_dic)
    val_dic, labs_shell_xval = _convert_shell_dic_to_list(labs_shell_xcore, val_dic)

    # check sanity
    for ele, lbs in labs_shell.items():
        cores = core_dic[ele]
        vals = val_dic[ele]

        cores_set = set(cores)
        vals_set = set(vals)
        lbs_set = set(lbs)

        if len(cores) != len(cores_set):
            logger.warn(mol, "core_dic['%s'] has replicate labels... %s", ele, cores)
            core_dic[ele] = list(cores_set)
        if len(vals) != len(vals_set):
            logger.warn(mol, "val_dic['%s'] has replicate labels... %s", ele, vals)
            val_dic[ele] = list(vals_set)
        if not (cores_set <= lbs_set):
            raise ValueError("%s cores contains unknown labels ... %s"
                             % (ele, cores_set - lbs_set))
        if not (vals_set <= lbs_set):
            raise ValueError("%s vals contains unknown labels ... %s"
                             % (ele, vals_set - lbs_set))

    for ele, bas_ele in basis_dic.items():
        cores = core_dic[ele]
        vals  = val_dic[ele]
        bas_core  = []
        bas_xcore = []
        bas_val   = []
        lbs_core = []
        lbs_val = []
        lbs_cur = copy.deepcopy(labs_shell[ele])

        cores = [val.lower() for val in cores]
        nshell_core = len(cores)
        itot = 0
        for b in bas_ele:
            #l = SPDF[b[0]]
            exp_coeff = np.array(b[1:])
            exp = exp_coeff[:, [0]]
            coeff = exp_coeff[:, 1:]
            ncoeff = coeff.shape[-1]

            idx_c = []
            for i in range(ncoeff):
                lb = lbs_cur[itot].lower()
                if lb in cores:
                    lbs_core.append(lb)
                    idx_c.append(i)
                itot += 1
            if len(idx_c) > 0:
                bas_core.append([b[0]] + np.hstack((exp, coeff[:, idx_c])).tolist())

        vals  = [val.lower() for val in vals]
        nshell_val = len(vals)
        itot = 0
        for b in bas_ele:
            #l = SPDF[b[0]]
            exp_coeff = np.array(b[1:])
            exp = exp_coeff[:, [0]]
            coeff = exp_coeff[:, 1:]
            ncoeff = coeff.shape[-1]

            idx_v = []
            for i in range(ncoeff):
                lb = lbs_cur[itot].lower()
                if lb in vals:
                    lbs_val.append(lb)
                    idx_v.append(i)
                itot += 1
            if len(idx_v) > 0:
                bas_val.append([b[0]] + np.hstack((exp, coeff[:, idx_v])).tolist())

        labs_core[ele] = lbs_core
        labs_val[ele] = lbs_val
        if bas_core:
            basis_core[ele] = bas_core
        if bas_xcore:
            basis_xcore[ele] = bas_xcore
        if bas_val:
            basis_val[ele] = bas_val

        if len(lbs_core) != nshell_core:
            logger.warn(mol, "%s # labels of core %s != # core %s",
                        ele, lbs_core, core_dic[ele])
        if len(lbs_val) != nshell_val:
            logger.warn(mol, "%s # labels of val %s != # val %s",
                        ele, lbs_val, val_dic[ele])

    logger.debug1(mol, "-" * 120)
    logger.debug1(mol, "split basis")
    logger.debug1(mol, "%8s   %-30s  %-30s  %-40s" %("element", "core",
                "valence", "virtual"))
    for ele, lbs in labs_core.items():
        lbs_val = labs_val[ele]
        lbs_xval = labs_shell_xval[ele]
        logger.debug1(mol, "%8s   %-30s  %-30s  %-40s"%(ele, " ".join(lbs),
                    " ".join(lbs_val), " ".join(lbs_xval)))
    logger.debug1(mol, "-" * 120)

    return basis_core, basis_val


def expand_shared_exp(mol):
    """
    Construct a mol with basis without shared exponents.

    Args:
        mol: Mole or Cell object.

    Returns:
        mol_new: A new mol without shared exponents.
    """
    basis = mol._basis
    basis_new = {}
    for ele, bas in basis.items():
        bas_new = []
        for b in bas:
            l = b[0]
            exp_coeff = b[1:]
            nexp = len(exp_coeff)
            ncoeff = len(exp_coeff[0]) - 1
            if ncoeff > 1:
                # 2 or more coeff share 1 exp
                for ic in range(ncoeff):
                    tmp = [l]
                    for ie in range(nexp):
                        tmp.append([exp_coeff[ie][0], exp_coeff[ie][ic+1]])
                    bas_new.append(tmp)
            else:
                bas_new.append(b)

        basis_new[ele] = bas_new

    mol_new = mol.copy()
    mol_new.basis = basis_new
    with lib.temporary_env(sys, stderr=open(os.devnull, "w")):
        mol_new.build(dump_input=False)
    return mol_new


def merge_basis(basis_core, basis_val):
    keys = basis_core.keys() | basis_val.keys()
    basis_new = {}
    for key in keys:
        bas = []
        if key in basis_core:
            bas.extend(basis_core[key])
        if key in basis_val:
            bas.extend(basis_val[key])
        basis_new[key] = bas
    return basis_new


def convert_basis_to_nwchem(symb, basis):
    """
    Convert the internal basis format to NWChem format string.
    More digits than the default pyscf one.
    """
    res = []
    symb = _std_symbol(symb)

    # pass 1: comment line
    ls = [b[0] for b in basis]
    nprims = [len(b[1:]) for b in basis]
    nctrs = [len(b[1])-1 for b in basis]
    prim_to_ctr = {}
    for i, l in enumerate(ls):
        if l in prim_to_ctr:
            prim_to_ctr[l][0] += nprims[i]
            prim_to_ctr[l][1] += nctrs[i]
        else:
            prim_to_ctr[l] = [nprims[i], nctrs[i]]
    nprims = []
    nctrs = []
    for l in set(ls):
        nprims.append(str(prim_to_ctr[l][0])+SPDF[l].lower())
        nctrs.append(str(prim_to_ctr[l][1])+SPDF[l].lower())
    res.append('#BASIS SET: (%s) -> [%s]' % (','.join(nprims), ','.join(nctrs)))

    # pass 2: basis data
    for bas in basis:
        res.append('%-2s    %s' % (symb, SPDF[bas[0]]))
        for dat in bas[1:]:
            res.append(' '.join('%27.17f'%x for x in dat))
    return '\n'.join(res)


def write_nwchem(fname, basis):
    ele_dic_std = set()
    string = ""
    for element, bas in basis.items():
        if _std_symbol(element) not in ele_dic_std:
            string += convert_basis_to_nwchem(element, bas)
            string += "\n"
            ele_dic_std.add(_std_symbol(element))
    with open(fname, 'w') as f:
        f.write(string)


if __name__ == "__main__":
    from pyscf import gto
    mol = gto.Mole()
    mol.atom = '''
    O        0.000000    0.000000    0.117790
    H        0.000000    0.755453   -0.471161
    H        0.000000   -0.755453   -0.471161'''
    mol.basis = '321g'
    mol.build()

    basis = get_atomic_hf_basis(mol, tol=1e-7, save_nwchem=None)
    print(basis)
