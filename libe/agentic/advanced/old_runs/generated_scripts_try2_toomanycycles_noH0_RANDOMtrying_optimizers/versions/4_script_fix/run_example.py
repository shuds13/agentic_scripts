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
        inputs=[],
        persis_in=["sim_id", "x", "x_on_cube", "f"],
        outputs=[("x", float, n), ("x_on_cube", float, n), ("sim_id", int),
                 ("local_min", bool), ("local_pt", bool)],
        user={
            "initial_sample_size": 30,
            "localopt_method": "LN_BOBYQA",
            "opt_return_codes": [0],
            "xatol": 1e-2,
            "fatol": 1e-2,
            "nu": 1e-6,
            "mu": 1e-8,
            "dist_to_bound_multiple": 0.01,
            "max_active_runs": 10,
            "lb": np.array([-0.089, 0.711]),
            "ub": np.array([-0.088, 0.712])
        }
    )

    alloc_specs = AllocSpecs(
        alloc_f=alloc_f,
    )

    exit_criteria = ExitCriteria(sim_max=300)

    ensemble = Ensemble(
        libE_specs=libE_specs,
        gen_specs=gen_specs,
        sim_specs=sim_specs,
        alloc_specs=alloc_specs,
        exit_criteria=exit_criteria,
        executor=exctr
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

        # Goal: Find at least 5 local minima
        min_required = 5
        if num_local_min >= min_required:
            print(f"\nAPOSMM_SUCCESS: Found {num_local_min} local minima (required: {min_required})")
        else:
            print(f"\nAPOSMM_NOT_SUCCEEDED: Found only {num_local_min} local minima (required: {min_required})")
