# Code Preview of "Intrinsic Wannier Functions for Hamiltonian downfolding"

The source code of the Intrinsic Wannier Function (IWF) method project, accompanying the manuscript "Intrinsic Wannier Functions for Hamiltonian downfolding": [arXiv:2608.15557](https://arxiv.org/abs/2608.15557).

This is a preview release of the code used to produce the results in the paper. It is built on top of [PySCF](https://github.com/pyscf/pyscf) (periodic mean-field module) and a modified [pyWannier90](https://github.com/hungpham2017/pyWannier90) interface, and implements the IWF construction for downfolding periodic mean-field calculations onto a compact, chemically intuitive orbital basis.

## Package layout

* `iwf/iwf.py` — core `IWF` class: builds reference (IAO/meta-Lowdin/NAO) orbitals, computes orbital character against the reference basis, and constructs the intrinsic Wannier functions (with optional smearing) that downfold the mean-field bands within a chosen energy window.
* `iwf/workflow.py` — `IWFWorkflow` wrapper that drives the end-to-end pipeline: running the periodic mean-field calculation, band structure setup, and the downfolding step.
* `iwf/plot.py` — plotting utilities (via Plotly) for visualizing band structures and IWF-related diagnostics.
* `iwf/utils.py` — helper functions shared across the package and benchmarks (e.g. reading VASP `POSCAR` files, Wannier90 projection keywords).
* `iwf/backend/` — supporting backends:
  * `pywannier90.py` — modified Wannier90/PySCF interface (based on pyWannier90).
  * `scdm.py` — Selected Columns of the Density Matrix (SCDM) localization, used for benchmarking.
  * `orth.py` — orbital orthogonalization utilities (Löwdin, meta-Löwdin, NAO).
  * `iaohelper.py` — helpers for Intrinsic Atomic Orbital (IAO) construction.
  * `atom_hf_pp.py`, `atom_scf_basis.py` — atomic HF/basis-set utilities used to build reference orbitals.

## Usage

To use the modified pywannier90, please include the path of your compiled `libwannier.so` into the environment variable `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=/path/to/your/wan_lib:$LD_LIBRARY_PATH
```

* Built with assistance from [Claude](https://anthropic.com) by Anthropic.
