import os
import sys

import numpy as np
from six_hump_camel import six_hump_camel

from libensemble import Ensemble
from libensemble.alloc_funcs.persistent_aposmm_alloc import persistent_aposmm_alloc as alloc_f
from libensemble.executors import MPIExecutor
from libensemble.gen_funcs.persistent_aposmm import aposmm as gen_f
from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, LibeSpecs, SimSpecs

if __name__ == "__main__":
    exctr = MPIExecutor()

    num_workers = 4

    libE_specs = LibeSpecs(
        nworkers=num_workers,
        gen_on_manager=True,
    )

    sim_specs = SimSpecs(
        sim_f=six_hump_camel,
        inputs=["x"],
        outputs=[("f", float)],
    )

    n = 2
    gen_specs = GenSpecs(
        gen_f=gen_f,
        inputs=["sim_id", "x", "x_on_cube", "f", "sim_ended", "local_pt", "local_min"],
        persis_in=["sim_id", "x", "x_on_cube", "f", "sim_ended", "local_pt", "local_min"],
        outputs=[("x", float, n), ("x_on_cube", float, n), ("sim_id", int),
                 ("local_min", bool), ("local_pt", bool)],
        user={
            "initial_sample_size": num_workers,
            "localopt_method": "scipy_Nelder-Mead",
            "opt_return_codes": [0],
            "xatol": 1e-2,  # Looser tolerance for faster convergence
            "fatol": 1e-2,  # Looser tolerance for faster convergence
            "nu": 1e-8,
            "mu": 1e-8,
            "dist_to_bound_multiple": 0.01,
            "max_active_runs": 6,
            "lb": np.array([-1.9872, 0.0161]),
            "ub": np.array([2.2963, 3.9902])
        }
    )

    alloc_specs = AllocSpecs(
        alloc_f=alloc_f,
        
    )

    exit_criteria = ExitCriteria(sim_max=300)
    # Load previous results as H0 (find latest .npy file)
    import glob
    _npy_files = sorted(glob.glob("run_example_history_*.npy"), key=lambda f: os.path.getmtime(f))
    if _npy_files:
        _H0_all = np.load(_npy_files[-1])
        _lb = gen_specs.user["lb"]
        _ub = gen_specs.user["ub"]
        
        # Filter to COMPLETED points within current bounds that have been given back to gen
        _completed = _H0_all["sim_ended"] & _H0_all["gen_informed"]
        _in_bounds = np.all((_H0_all["x"] >= _lb) & (_H0_all["x"] <= _ub), axis=1)
        _valid = _completed & _in_bounds
        H0 = _H0_all[_valid].copy()
        
        if len(H0) > 0:
            # Recompute x_on_cube for new bounds
            H0["x_on_cube"] = (H0["x"] - _lb) / (_ub - _lb)
            # Renumber sim_ids to be sequential (required by libEnsemble)
            H0["sim_id"] = np.arange(len(H0))
            print("Using", len(H0), "of", len(_H0_all), "completed points from", _npy_files[-1])
        else:
            H0 = None
            print("No completed points within current bounds, starting fresh")
    else:
        H0 = None
        print("No previous results found, starting fresh")


    ensemble = Ensemble(
        libE_specs=libE_specs,
        gen_specs=gen_specs,
        sim_specs=sim_specs,
        alloc_specs=alloc_specs,
        exit_criteria=exit_criteria,
        executor=exctr,
        H0=H0
    )

    ensemble.add_random_streams()
    H, persis_info, flag = ensemble.run()

    if ensemble.is_manager:
        print("First 3:", H[["sim_id", "x", "f"]][:3])
        print("Last 3:", H[["sim_id", "x", "f"]][-3:])

        num_local_min = np.sum(H["local_min"])
        print(f"\nNumber of local minima found: {num_local_min}")
        print("Local minima x values:", H[H["local_min"]]["x"])

        ensemble.save_output(__file__)

        # Goal: Find at least 3 local minima
        min_required = 3
        if num_local_min >= min_required:
            print(f"\nAPOSMM_SUCCESS: Found {num_local_min} local minima (required: {min_required})")
        else:
            print(f"\nAPOSMM_NOT_SUCCEEDED: Found only {num_local_min} local minima (required: {min_required})")
