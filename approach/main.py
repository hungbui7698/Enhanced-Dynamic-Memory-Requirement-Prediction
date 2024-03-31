from sklearn.linear_model import (
    RANSACRegressor,
    LinearRegression,
    Ridge,
    ARDRegression,
    BayesianRidge,
    HuberRegressor,
    SGDRegressor,
    TheilSenRegressor,
)

from approach import (
    Globals,
    TimedBlock,
    show_shortened_names,
    rc,
    dask_client,
)
from approach.files import (
    WFCSVDirectory,
    DenoiseParameters,
    DataDirectory,
    RuptureParams,
    DynamicDenoiseParameters,
    basic_dynamic_stride,
    StaticDenoiseParameters,
)
from approach.models import (
    WittModel,
    TovarModel,
    RetryMode,
    KEqualSegmentingModel,
    DefaultRetryModel,
    KBinSegSegmentingModel,
    KBottomUpSegmentingModel,
    BufferRetryModel,
    WittMode,
    KPeltSegmentingModel,
    KKernelCPDSegmentingModel,
    KDynpSegmentModel,
    KDynpSegmentingModel,
    KWindowSegmentingModel,
    get_standard_models,
)
from approach.plotting import (
    plot_task_wastage,
    plot_overall_wastage,
    plot_overall_efficacy,
    plot_overall_potential,
    PotentialTypes,
)
from approach.progress import progress
from approach.simulation import (
    benchmark_workflow,
    WastageTypes,
    benchmark_data,
    benchmark_result_to_df,
    workflow_result_to_df,
    calc_total_work,
    benchmark_flat,
)
from approach.parameter_optimization import (
    hyperparameter_optimization,
    make_configspace,
    make_model_from_hyperparameter,
)


def main():
    with TimedBlock("Execution"):
        dask_client()
        do_optimization: bool = True
        do_whole_data: bool = True
        do_many_models: bool = True
        do_random_models: int = 100
        if do_optimization:
            hyperparameter_optimization()
        else:
            with rc.status("Make models"):
                models = get_standard_models()
                if do_many_models:
                    if do_random_models == 0:
                        for model in (KEqualSegmentingModel,):
                            for k in range(3, 6):
                                for reg_model in (
                                    # RANSACRegressor,
                                    # Ridge,
                                    # ARDRegression,
                                    # BayesianRidge,
                                    # HuberRegressor,
                                    # SGDRegressor,
                                    TheilSenRegressor,
                                    LinearRegression,
                                ):
                                    models.append(
                                        model(
                                            k,
                                            reg_model_type=reg_model,
                                            use_preprocessed=False,
                                        )
                                    )
                                    for stride in (25,):  # 35, 45,  # , 55, 65):
                                        for depth in (None,):  # 1, 2):
                                            models.append(
                                                model(
                                                    k,
                                                    reg_model_type=reg_model,
                                                    use_preprocessed=StaticDenoiseParameters(
                                                        stride=stride, depth=depth
                                                    ),
                                                )
                                            )
                                    for stride in (
                                        basic_dynamic_stride,
                                    ):  # 35, 45,  # , 55, 65):
                                        for depth in (None,):  # 1, 2):
                                            models.append(
                                                model(
                                                    k,
                                                    reg_model_type=reg_model,
                                                    use_preprocessed=DynamicDenoiseParameters(
                                                        stride=stride, depth=depth
                                                    ),
                                                )
                                            )
                        for model in (
                            KBottomUpSegmentingModel,
                            KBinSegSegmentingModel,
                            KKernelCPDSegmentingModel,
                            KDynpSegmentingModel,
                            KWindowSegmentingModel,
                        ):
                            for k in range(3, 6):
                                for reg_model in (
                                    # RANSACRegressor,
                                    # Ridge,
                                    # ARDRegression,
                                    # BayesianRidge,
                                    # HuberRegressor,
                                    # SGDRegressor,
                                    TheilSenRegressor,
                                    LinearRegression,
                                ):
                                    rupture_params = RuptureParams(k)
                                    models.append(
                                        model(
                                            rupture_params,
                                            reg_model_type=reg_model,
                                            use_preprocessed=False,
                                        )
                                    )
                                    for stride in (25,):  # 35, 45,  # , 55, 65):
                                        for depth in (None,):  # 1, 2):
                                            models.append(
                                                model(
                                                    rupture_params,
                                                    reg_model_type=reg_model,
                                                    use_preprocessed=StaticDenoiseParameters(
                                                        stride=stride, depth=depth
                                                    ),
                                                )
                                            )
                                    for stride in (
                                        basic_dynamic_stride,
                                    ):  # 35, 45,  # , 55, 65):
                                        for depth in (None,):  # 1, 2):
                                            models.append(
                                                model(
                                                    rupture_params,
                                                    reg_model_type=reg_model,
                                                    use_preprocessed=DynamicDenoiseParameters(
                                                        stride=stride, depth=depth
                                                    ),
                                                )
                                            )
                        for model in (KPeltSegmentingModel,):
                            for reg_model in (
                                # RANSACRegressor,
                                # Ridge,
                                # ARDRegression,
                                # BayesianRidge,
                                # HuberRegressor,
                                # SGDRegressor,
                                TheilSenRegressor,
                                LinearRegression,
                            ):
                                rupture_params = RuptureParams(pen=50)
                                models.append(
                                    model(
                                        rupture_params,
                                        reg_model_type=reg_model,
                                        use_preprocessed=False,
                                    )
                                )
                                for stride in (25,):  # 35, 45,  # , 55, 65):
                                    for depth in (None,):  # 1, 2):
                                        models.append(
                                            model(
                                                rupture_params,
                                                reg_model_type=reg_model,
                                                use_preprocessed=StaticDenoiseParameters(
                                                    stride=stride, depth=depth
                                                ),
                                            )
                                        )
                                for stride in (
                                    basic_dynamic_stride,
                                ):  # 35, 45,  # , 55, 65):
                                    for depth in (None,):  # 1, 2):
                                        models.append(
                                            model(
                                                rupture_params,
                                                reg_model_type=reg_model,
                                                use_preprocessed=DynamicDenoiseParameters(
                                                    stride=stride, depth=depth
                                                ),
                                            )
                                        )
                    else:
                        configspace = make_configspace()
                        models.extend(
                            [
                                make_model_from_hyperparameter(config)
                                for config in configspace.sample_configuration(
                                    do_random_models
                                )
                            ]
                        )
            #
            with rc.status("Parse data"):
                if do_whole_data:
                    files = DataDirectory(Globals.base_data_path)
                else:
                    files = WFCSVDirectory(
                        Globals.base_data_path.joinpath("chipseq"),
                    )
                #
                percentages = Globals.default_percentages
                retry_modes = [RetryMode.Partial]  # RetryMode.Selective,
                retry_models = [
                    DefaultRetryModel(),
                    # BufferRetryModel(0.75),
                ]
                shuffle_repeats = 5
                total_work = calc_total_work(
                    files,
                    models,
                    percentages,
                    retry_modes,
                    retry_models,
                    shuffle_repeats,
                )
            #
            with progress.get_progress() as prog:
                progress.add_prog_bar("Simulate", total_work)
                if do_whole_data:
                    # df = benchmark_result_to_df(
                    #     benchmark_data(
                    #         models,
                    #         files=files,
                    #         percentages=percentages,
                    #         retry_mode=retry_modes,
                    #         retry_model=retry_models,
                    #         shuffle_repeats=shuffle_repeats,
                    #     ),
                    # )
                    df = benchmark_result_to_df(
                        benchmark_flat(
                            models,
                            files=files,
                            percentages=percentages,
                            retry_mode=retry_modes,
                            retry_model=retry_models,
                            shuffle_repeats=shuffle_repeats,
                        ),
                    )
                else:
                    df = workflow_result_to_df(
                        benchmark_workflow(
                            models,
                            percentages=percentages,
                            files=files,
                            retry_mode=retry_modes,
                            retry_model=retry_models,
                            shuffle_repeats=shuffle_repeats,
                        ),
                    )
                progress.reset()
            #
            with rc.status("Make plots"):
                if False:
                    for l, d in df.groupby("task"):
                        plot_task_wastage(df, l)
                if True:
                    plot_overall_wastage(df, wastage_type=WastageTypes.Total)
                    plot_overall_wastage(df, wastage_type=WastageTypes.Over)
                    plot_overall_wastage(df, wastage_type=WastageTypes.Under)
                    plot_overall_efficacy(df)
                    plot_overall_potential(
                        df, potential_type=PotentialTypes.With_retries
                    )
                    plot_overall_potential(df, potential_type=PotentialTypes.Full)
                    show_shortened_names()
            # rc.log(execution_counts["benchmark_train_test_split"])


if __name__ == "__main__":
    main()
