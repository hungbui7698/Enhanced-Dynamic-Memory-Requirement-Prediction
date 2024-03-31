#!/usr/bin/env python3
from enum import Enum, auto

from ConfigSpace import (
    ConfigurationSpace,
    EqualsCondition,
    InCondition,
    Float,
    Uniform,
    AndConjunction,
)
from ConfigSpace.conditions import OrConjunction
from distributed import get_client
from smac import HyperparameterOptimizationFacade, Scenario

from . import (
    run_configuration,
    make_model_from_hyperparameter,
    get_benchmark_params_from_hyperparameter,
)
from .. import Globals, reset_shortened_names, show_shortened_names
from ..files import DataDirectory
from ..models import get_standard_models
from ..plotting import (
    plot_overall_wastage,
    plot_overall_efficacy,
    plot_overall_potential,
)
from ..progress import progress
from ..simulation import (
    benchmark_result_to_df,
    WastageTypes,
    PotentialTypes,
    calc_total_work,
    benchmark_flat,
)
from pathlib import Path


class HyperParameterTypes(Enum):
    k = auto()
    model_type = auto()
    regression_model = auto()
    retry_mode = auto()
    retry_model = auto()
    retry_buffer_percentage = auto()
    do_denoise = auto()
    stride = auto()
    shuffle_repeats = auto()
    depth = auto()


def make_configspace(optimistic: bool = Globals.do_optimistic_smac_params):
    configspace = ConfigurationSpace(
        {
            # global paramterers
            "k": (2, 15),
            "segment_model": [
                "k-equal",
                "k-bottom-up",
                "k-binseg",
                "k-pelt",
                "k-window",
                "k-dynp",
                "k-kernel-cpd",
            ],  # TODO: Change this to single model and run 3x (on separate machines?!)
            "regression_model": [
                "ransac",
                "ridge",
                "ard",
                "bayesian",
                "hubert",
                "theilsen",
                "linear",
            ],
            "retry_mode": ["selective", "partial"],
            "retry_model": ["default", "buffer"],
            "retry_buffer_percentage": (0.10, 1.00),
            # Denoise parameters
            "do_denoise": [True, False],
            "dynamic_stride": [True, False],
            "denoise_maxer_type": ["blockify", "stepify"],
            "stride": (5, 200),
            "shuffle_repeats": 5,
            "depth": -1,
            # TODO (now): add more rupture parameters
            "rupture_model_l1_l2_rbf": [
                "l1",
                "l2",
                "rbf",
            ],
            "rupture_model_linear_rgf_cosine": ["linear", "rbf", "cosine"],
            "rupture_minsize": (1, 5) if optimistic else 2,
            "rupture_jump": (3, 10) if optimistic else 5,
            "rupture_use_pen_epsilon": [True, False],
            "rupture_penalty": Float("rupture_penalty", (0.01, 1000), log=True),
            "rupture_epsilon": Float("rupture_epsilon", (0.01, 1000), log=True),
            "rupture_width": (10, 200),
            # TODO (future): Regression parameter (sklearn documentation)
            "ridge_alpha": Float(
                "ridge_alpha", (1e-6, 10), distribution=Uniform(), log=True
            )
            if optimistic
            else 1e-6,
            "ard_alpha_1": Float(
                "ard_alpha_1", (1e-6, 10), distribution=Uniform(), log=True
            )
            if optimistic
            else 1e-6,
            "ard_alpha_2": Float(
                "ard_alpha_2", (1e-6, 10), distribution=Uniform(), log=True
            )
            if optimistic
            else 1e-6,
            "ard_lambda_1": Float(
                "ard_lambda_1", (1e-6, 10), distribution=Uniform(), log=True
            )
            if optimistic
            else 1e-6,
            "ard_lambda_2": Float(
                "ard_lambda_2", (1e-6, 10), distribution=Uniform(), log=True
            )
            if optimistic
            else 1e-6,
            "bayesian_alpha_1": Float(
                "bayesian_alpha_1", (1e-6, 10), distribution=Uniform(), log=True
            )
            if optimistic
            else 1e-6,
            "bayesian_alpha_2": Float(
                "bayesian_alpha_2", (1e-6, 10), distribution=Uniform(), log=True
            )
            if optimistic
            else 1e-6,
            "bayesian_lambda_1": Float(
                "bayesian_lambda_1", (1e-6, 10), distribution=Uniform(), log=True
            )
            if optimistic
            else 1e-6,
            "bayesian_lambda_2": Float(
                "bayesian_lambda_2", (1e-6, 10), distribution=Uniform(), log=True
            )
            if optimistic
            else 1e-6,
            "hubert_epsilon": Float(
                "hubert_epsilon", (1, 10), distribution=Uniform(), log=True
            )
            if optimistic
            else 1,
            "hubert_alpha": Float(
                "hubert_alpha", (1e-6, 10), distribution=Uniform(), log=True
            )
            if optimistic
            else 1e-6,
            # TODO (now): offset parameter (for segment model)
        }
    )

    # Conditions for parameters ...
    for param in ["ridge_alpha"]:
        configspace.add_condition(
            EqualsCondition(
                configspace[param], configspace["regression_model"], "ridge"
            )
        )
    for param in ["ard_alpha_1", "ard_alpha_2", "ard_lambda_1", "ard_lambda_2"]:
        configspace.add_condition(
            EqualsCondition(configspace[param], configspace["regression_model"], "ard")
        )
    for param in [
        "bayesian_alpha_1",
        "bayesian_alpha_2",
        "bayesian_lambda_1",
        "bayesian_lambda_2",
    ]:
        configspace.add_condition(
            EqualsCondition(
                configspace[param], configspace["regression_model"], "bayesian"
            )
        )
    for param in ["hubert_epsilon", "hubert_alpha"]:
        configspace.add_condition(
            EqualsCondition(
                configspace[param], configspace["regression_model"], "hubert"
            )
        )
    for rpt_param in ["minsize", "jump"]:
        configspace.add_condition(
            InCondition(
                configspace["rupture_" + rpt_param],
                configspace["segment_model"],
                [
                    "k-bottom-up",
                    "k-binseg",
                    "k-pelt",
                    "k-window",
                    "k-dynp",
                    "k-kernel-cpd",
                ],
            )
        )
    configspace.add_condition(
        InCondition(
            configspace["rupture_use_pen_epsilon"],
            configspace["segment_model"],
            [
                "k-bottom-up",
                "k-binseg",
                "k-window",
                "k-kernel-cpd",
            ],
        )
    )
    configspace.add_condition(
        InCondition(
            configspace["rupture_width"], configspace["segment_model"], ["k-window"]
        )
    )
    configspace.add_condition(
        InCondition(
            configspace["rupture_model_l1_l2_rbf"],
            configspace["segment_model"],
            ["k-bottom-up", "k-binseg", "k-pelt", "k-window", "k-dynp"],
        )
    )
    configspace.add_condition(
        InCondition(
            configspace["rupture_model_linear_rgf_cosine"],
            configspace["segment_model"],
            ["k-kernel-cpd"],
        )
    )
    configspace.add_condition(
        OrConjunction(
            EqualsCondition(
                configspace["rupture_penalty"], configspace["segment_model"], "k-pelt"
            ),
            AndConjunction(
                InCondition(
                    configspace["rupture_penalty"],
                    configspace["segment_model"],
                    ["k-bottom-up", "k-binseg", "k-pelt", "k-window", "k-kernel-cpd"],
                ),
                EqualsCondition(
                    configspace["rupture_penalty"],
                    configspace["rupture_use_pen_epsilon"],
                    True,
                ),
            ),
        )
    )
    configspace.add_condition(
        AndConjunction(
            InCondition(
                configspace["rupture_epsilon"],
                configspace["segment_model"],
                ["k-bottom-up", "k-binseg", "k-window"],
            ),
            EqualsCondition(
                configspace["rupture_epsilon"],
                configspace["rupture_use_pen_epsilon"],
                True,
            ),
        )
    )
    configspace.add_condition(
        OrConjunction(
            InCondition(
                configspace["k"], configspace["segment_model"], ["k-equal", "k-dynp"]
            ),
            AndConjunction(
                InCondition(
                    configspace["k"],
                    configspace["segment_model"],
                    [
                        "k-equal",
                        "k-bottom-up",
                        "k-binseg",
                        "k-window",
                        "k-dynp",
                        "k-kernel-cpd",
                    ],
                ),
                EqualsCondition(
                    configspace["k"], configspace["rupture_use_pen_epsilon"], False
                ),
            ),
        )
    )

    configspace.add_condition(
        EqualsCondition(
            configspace["dynamic_stride"],
            configspace["do_denoise"],
            True,
        )
    )
    configspace.add_condition(
        EqualsCondition(
            configspace["denoise_maxer_type"],
            configspace["do_denoise"],
            True,
        )
    )
    configspace.add_condition(
        AndConjunction(
            EqualsCondition(configspace["stride"], configspace["do_denoise"], True),
            EqualsCondition(
                configspace["stride"], configspace["dynamic_stride"], False
            ),
        )
    )
    configspace.add_condition(
        EqualsCondition(
            configspace["retry_buffer_percentage"],  # Only use retry_buffer_percentage
            configspace["retry_model"],  # when retry_model == "buffer"
            "buffer",
        )
    )

    return configspace


def hyperparameter_optimization():
    # Parameters to optimize ...
    configspace = make_configspace()

    # How to optimize ...
    scenario = Scenario(
        configspace,
        deterministic=True,
        n_trials=100007,  # n_trials
        # walltime_limit=129600,         # Max overall runtime in sec (36h)
        # output_directory=Path("/home/jfechner/MPDS-Project/experiment-2"),
        # trial_walltime_limit=timedelta(minutes=10).total_seconds(),  # Max runtime per trial (10min) in sec
        # n_workers=16,  # Parallel computation
    )

    # 'cache' files in globals
    files = DataDirectory(Globals.base_data_path)
    if Globals.do_parallel:
        Globals.data_files = files

    # Optimization starting here ...
    smac = HyperparameterOptimizationFacade(
        scenario, run_configuration, dask_client=get_client()
    )
    retries = 10
    while retries:
        try:
            incumbent = smac.optimize(data_to_scatter={"files": files})
        except:
            retries -= 1
            if retries > 0:
                continue
            else:
                exit(1)
    print("=>", incumbent)
    winner = make_model_from_hyperparameter(incumbent)
    (
        retry_mode,
        retry_model,
        shuffle_repeats,
    ) = get_benchmark_params_from_hyperparameter(incumbent)
    # Globals.do_parallel = False
    percentages = Globals.default_percentages
    total_work = calc_total_work(
        files,
        [winner] + get_standard_models(),
        percentages,
        [retry_mode],
        [retry_model],
        shuffle_repeats,
    )
    with progress.get_progress() as prog:
        progress.add_prog_bar("Simulate", total_work)
        reset_shortened_names()
        df = benchmark_result_to_df(
            benchmark_flat(
                [winner] + get_standard_models(),
                files=files,
                percentages=percentages,
                retry_mode=retry_mode,
                retry_model=retry_model,
                shuffle_repeats=shuffle_repeats,
            )
        )
        progress.reset()
    plot_overall_wastage(df, wastage_type=WastageTypes.Total)
    plot_overall_wastage(df, wastage_type=WastageTypes.Over)
    plot_overall_wastage(df, wastage_type=WastageTypes.Under)
    plot_overall_efficacy(df)
    plot_overall_potential(df, potential_type=PotentialTypes.With_retries)
    plot_overall_potential(df, potential_type=PotentialTypes.Full)
    show_shortened_names()


if __name__ == "__main__":
    hyperparameter_optimization()
