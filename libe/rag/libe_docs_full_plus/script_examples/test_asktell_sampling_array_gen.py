"""
Runs libEnsemble with Latin hypercube sampling on a simple 1D problem

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_sampling_asktell_gen.py
   python test_sampling_asktell_gen.py --nworkers 3 --comms local
   python test_sampling_asktell_gen.py --nworkers 3 --comms tcp

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 2 4

import numpy as np
from gest_api import Generator
from gest_api.vocs import VOCS
from gest_api.vocs import ContinuousVariable

# Import libEnsemble items for this test
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.gen_classes.sampling import UniformSample
from libensemble.libE import libE
from libensemble.specs import GenSpecs
from libensemble.tools import add_unique_random_streams, parse_args


class UniformSampleArray(Generator):
    """
    This sampler adheres to the gest-api VOCS interface and data structures.

    Uses one array variable of any dimension. Array is a numpy array.
    """

    def __init__(self, VOCS: VOCS):
        self.VOCS = VOCS
        self.rng = np.random.default_rng(1)
        super().__init__(VOCS)

    def _validate_vocs(self, VOCS):
        assert len(self.VOCS.variables) == 1, "VOCS must contain exactly one variable."

    def suggest(self, n_trials):
        output = []
        key = list(self.VOCS.variables.keys())[0]
        var = self.VOCS.variables[key]
        for _ in range(n_trials):
            trial = {key: np.array([
                self.rng.uniform(bounds[0], bounds[1])
                for bounds in var.domain
            ])}
            output.append(trial)
        return output

    def ingest(self, calc_in):
        pass  # random sample so nothing to tell


def sim_f(In):
    Out = np.zeros(1, dtype=[("f", float)])
    Out["f"] = np.linalg.norm(In)
    return Out


if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()
    libE_specs["gen_on_manager"] = True

    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float)],
    }

    gen_specs = GenSpecs(
        persis_in=["x", "f", "sim_id"],
        outputs=[("x", float, (2,))],
        initial_batch_size=20,
        batch_size=10,
        user={
            "initial_batch_size": 20,  # for wrapper
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
        },
    )

    # variables = {"x0": [-3, 3], "x1": [-2, 2]}


    variables = {"x": ContinuousVariable(dtype=(2,),domain=[[-3, 3], [-2, 2]])}


    objectives = {"energy": "EXPLORE"}

    vocs = VOCS(variables=variables, objectives=objectives)

    alloc_specs = {"alloc_f": alloc_f}
    exit_criteria = {"gen_max": 201}
    persis_info = add_unique_random_streams({}, nworkers + 1, seed=1234)

    for test in range(1):  # 3
        if test == 0:
            generator = UniformSampleArray(vocs)

        # elif test == 1:
        #     persis_info["num_gens_started"] = 0
        #     generator = UniformSample(vocs)

        # elif test == 2:
        #     persis_info["num_gens_started"] = 0
        #     generator = UniformSample(vocs, variables_mapping={"x": ["x0", "x1"], "f": ["energy"]})

        gen_specs.generator = generator
        H, persis_info, flag = libE(
            sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs=libE_specs
        )

        if is_manager:
            print(H[["sim_id", "x", "f"]][:10])
            assert len(H) >= 201, f"H has length {len(H)}"
