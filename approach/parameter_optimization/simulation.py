import ctypes
from typing import Dict, Any

from ConfigSpace import Configuration
from distributed import get_client, Future, secede, rejoin
from pynisher import WallTimeoutException, MemoryLimitException
from sklearn.linear_model import (
    RANSACRegressor,
    Ridge,
    ARDRegression,
    BayesianRidge,
    HuberRegressor,
    TheilSenRegressor,
    LinearRegression,
)

from .. import Globals, timed_function, rc, FailedTrainingWarning
from ..files import (
    DenoiseParameters,
    DataDirectory,
    RuptureParams,
    StaticDenoiseParameters,
    DynamicDenoiseParameters,
    basic_dynamic_stride,
)
from ..models import (
    KEqualSegmentingModel,
    KBottomUpSegmentingModel,
    KBinSegSegmentingModel,
    RetryMode,
    DefaultRetryModel,
    BufferRetryModel,
    TaskModel,
    RetryModel,
    KPeltSegmentingModel,
    KWindowSegmentingModel,
    KDynpSegmentingModel,
    KKernelCPDSegmentingModel,
)
from ..preprocessing import MaxerTypes
from ..progress import progress
from ..simulation import (
    evaluate_benchmark,
    benchmark_data,
    calc_total_work,
    benchmark_flat,
    BadModelException,
)

MODEL_TYPES = {
    "k-equal": KEqualSegmentingModel,
    "k-bottom-up": KBottomUpSegmentingModel,
    "k-binseg": KBinSegSegmentingModel,
    "k-pelt": KPeltSegmentingModel,
    "k-window": KWindowSegmentingModel,
    "k-dynp": KDynpSegmentingModel,
    "k-kernel-cpd": KKernelCPDSegmentingModel,
}

REGRESSION_MODELS = {
    "ransac": RANSACRegressor,  # takes another model as parameter ...
    "ridge": Ridge,  # alpha
    "ard": ARDRegression,  # alpha_1, alpha_2, lambda_1, lambda_2
    "bayesian": BayesianRidge,  # alpha_1, alpha_2, lambda_1, lambda_2
    "hubert": HuberRegressor,  # epsilon (>=1), alpha
    # "sgd": SGDRegressor,
    "theilsen": TheilSenRegressor,
    "linear": LinearRegression,
}
MAXER_TYPES = {"blockify": MaxerTypes.blockify, "stepify": MaxerTypes.stepify}

RETRY_MODES = {"selective": RetryMode.Selective, "partial": RetryMode.Partial}

RETRY_MODELS = {"default": DefaultRetryModel, "buffer": BufferRetryModel}


def make_model_from_hyperparameter(hyperparameter: Configuration) -> TaskModel:
    k = hyperparameter.get("k")
    reg_model = REGRESSION_MODELS[hyperparameter["regression_model"]]
    model_type = MODEL_TYPES[hyperparameter["segment_model"]]
    do_denoise = hyperparameter["do_denoise"]
    if do_denoise:
        dynamic_stride = hyperparameter["dynamic_stride"]
        if not dynamic_stride:
            stride = hyperparameter["stride"]
            denoise_params = StaticDenoiseParameters(
                stride, maxer_type=MAXER_TYPES[hyperparameter["denoise_maxer_type"]]
            )
        else:
            denoise_params = DynamicDenoiseParameters(
                basic_dynamic_stride,
                maxer_type=MAXER_TYPES[hyperparameter["denoise_maxer_type"]],
            )
    else:
        denoise_params = False
    depth = hyperparameter["depth"]
    if depth == -1:
        depth = None

    match hyperparameter["regression_model"]:
        case "ridge":  # alpha
            reg_model_params = {"alpha": hyperparameter["ridge_alpha"]}
        case "ard":  # alpha_1, alpha_2, lambda_1, lambda_2
            reg_model_params = {
                "alpha_1": hyperparameter["ard_alpha_1"],
                "alpha_2": hyperparameter["ard_alpha_2"],
                "lambda_1": hyperparameter["ard_lambda_1"],
                "lambda_2": hyperparameter["ard_lambda_2"],
            }
        case "bayesian":  # alpha_1, alpha_2, lambda_1, lambda_2
            reg_model_params = {
                "alpha_1": hyperparameter["bayesian_alpha_1"],
                "alpha_2": hyperparameter["bayesian_alpha_2"],
                "lambda_1": hyperparameter["bayesian_lambda_1"],
                "lambda_2": hyperparameter["bayesian_lambda_2"],
            }
        case "hubert":  # epsilon (>=1), alpha
            reg_model_params = {
                "epsilon": hyperparameter["hubert_epsilon"],
                "alpha": hyperparameter["hubert_alpha"],
            }
        case _:
            reg_model_params = {}

    # Init model
    if model_type is KEqualSegmentingModel:
        return model_type(
            k,
            reg_model_type=reg_model,
            use_preprocessed=denoise_params,
            reg_model_params=reg_model_params,
        )
    else:
        rpt_model = hyperparameter.get("rupture_model_l1_l2_rbf")
        if rpt_model is None:
            rpt_model = hyperparameter.get("rupture_model_linear_rgf_cosine")
        return model_type(
            RuptureParams(
                num_segments=k,
                model=rpt_model,
                min_size=hyperparameter["rupture_minsize"],
                jump=hyperparameter["rupture_jump"],
                width=hyperparameter.get("rupture_width"),
                pen=hyperparameter.get("rupture_penalty"),
                epsilon=hyperparameter.get("rupture_epsilon"),
            ),
            reg_model_type=reg_model,
            use_preprocessed=denoise_params,
            reg_model_params=reg_model_params,
        )


def get_benchmark_params_from_hyperparameter(
    hyperparameter: Configuration,
) -> tuple[RetryMode, RetryModel, int]:
    retry_mode = RETRY_MODES[hyperparameter["retry_mode"]]
    retry_model = RETRY_MODELS[hyperparameter["retry_model"]]
    if hyperparameter["retry_model"] == "buffer":
        retry_model = retry_model(hyperparameter["retry_buffer_percentage"])
    else:
        retry_model = retry_model()
    shuffle_repeats = hyperparameter["shuffle_repeats"]

    return retry_mode, retry_model, shuffle_repeats


@timed_function(enabled=Globals.do_parallel)
def run_configuration(
    hyperparameter: Configuration,
    seed: int = None,
    data_to_scatter: Dict[str, Any] = None,
) -> float:
    # def trim_memory() -> int:
    #     libc = ctypes.CDLL("libc.so.6")
    #     return libc.malloc_trim(0)
    #
    # client = get_client()
    # client.run(trim_memory)
    try:
        if data_to_scatter is None:
            if Globals.do_parallel:
                files = Globals.data_files
            else:
                files = DataDirectory(Globals.base_data_path)
        else:
            files = data_to_scatter["files"]  # todo remove caches from files
    except Exception as e:
        rc.print_exception()
        return float("inf")
    # Extract values from current parameter-dict
    try:
        (
            retry_mode,
            retry_model,
            shuffle_repeats,
        ) = get_benchmark_params_from_hyperparameter(hyperparameter)
    except Exception as e:
        rc.print_exception()
        return float("inf")

    # Init model
    try:
        model = make_model_from_hyperparameter(hyperparameter)
    except Exception as e:
        rc.print_exception()
        return float("inf")
    percentages = Globals.default_percentages
    total_work = calc_total_work(
        files,
        [model],
        percentages,
        [retry_mode],
        [retry_model],
        shuffle_repeats,
    )
    try:
        bench = benchmark_flat(
            [model],
            files=files,
            percentages=percentages,
            retry_mode=retry_mode,
            retry_model=retry_model,
            shuffle_repeats=shuffle_repeats,
        )
    except Exception as e:
        if not isinstance(
            e, (BadModelException, WallTimeoutException, MemoryLimitException)
        ):
            rc.print_exception()
        return float("inf")
    try:
        results = evaluate_benchmark(bench).groupby(by=["orig"])["val"].mean().values[0]
    except Exception as e:
        rc.print_exception()
        return float("inf")
    if results >= 10_000:
        return float("inf")
    # with progress.get_progress() as prog:
    #     progress.add_prog_bar("Simulate", total_work)
    #     results = evaluate_benchmark(
    #         benchmark_data(
    #             [model],
    #             files=files,
    #             percentages=percentages,
    #             retry_mode=retry_mode,
    #             retry_model=retry_model,
    #             shuffle_repeats=shuffle_repeats,
    #         )
    #     )
    #     progress.reset()
    return results
