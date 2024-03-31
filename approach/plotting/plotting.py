import pandas as pds
import seaborn as sns
from matplotlib import pyplot as plt

from .. import (
    timed_function,
)
from ..simulation import (
    WastageTypes,
    col_name_by_wastage_type,
    PotentialTypes,
    col_name_by_potential_type,
    order_models,
)


# todo all these plotting functions are really similar: DRY
@timed_function()
def plot_task_wastage(
    df,
    task_name,
    val_col="val",
    do_percent=False,
    wastage_type: WastageTypes = WastageTypes.Total,
    split_retry_modes: bool = True,
    split_retry_models: bool = True,
) -> None:
    sns.set_theme(style="darkgrid")
    if do_percent:
        val_col = "as_p"
        d_ = df_to_percentages(df.copy(), perc_col=val_col)
    else:
        d_ = df
    filter_cat = col_name_by_wastage_type[wastage_type]
    g = sns.catplot(
        d_.query("task==@task_name and cat==@filter_cat"),
        kind="bar",
        x="short_name",
        y=val_col,
        hue="perc",
        row="retry_mode" if split_retry_modes else None,
        col="retry_model" if split_retry_models else None,
        aspect=max(len(d_.query("cat==@filter_cat")["short_name"].unique()) / 8, 1),
        order=order_models(d_, col_name_by_wastage_type[WastageTypes.Total])[
            "short_name"
        ],
    )
    g.figure.suptitle(f"{task_name} {wastage_type.name.lower()} wastage")
    for ax in g.axes.ravel():
        ax.tick_params(axis="x", which="both", rotation=45)
    g.tight_layout()
    g.set_titles(template="{row_name}, {col_name}")
    plt.show()


@timed_function()
def plot_task_efficacy(
    df,
    task_name,
    val_col="val",
    do_percent=False,
    split_retry_modes: bool = True,
    split_retry_models: bool = True,
) -> None:
    sns.set_theme(style="darkgrid")
    if do_percent:
        val_col = "as_p"
        d_ = df_to_percentages(df.copy(), perc_col=val_col)
    else:
        d_ = df
    g = sns.catplot(
        d_.query("task==@task_name and cat=='Efficacy'"),
        kind="bar",
        x="short_name",
        y=val_col,
        hue="perc",
        row="retry_mode" if split_retry_modes else None,
        col="retry_model" if split_retry_models else None,
        aspect=max(len(d_.query("cat=='Efficacy'")["short_name"].unique()) / 8, 1),
        order=order_models(d_, col_name_by_wastage_type[WastageTypes.Total])[
            "short_name"
        ],
    )
    g.figure.suptitle(f"{task_name} efficacy")
    for ax in g.axes.ravel():
        ax.tick_params(axis="x", which="both", rotation=45)
    g.tight_layout()
    g.set_titles(template="{row_name}, {col_name}")
    plt.show()


@timed_function()
def plot_task_potential(
    df,
    task_name,
    val_col="val",
    do_percent=False,
    potential_type: PotentialTypes = PotentialTypes.Full,
    split_retry_modes: bool = True,
    split_retry_models: bool = True,
) -> None:
    sns.set_theme(style="darkgrid")
    if do_percent:
        val_col = "as_p"
        d_ = df_to_percentages(df.copy(), perc_col=val_col)
    else:
        d_ = df
    filter_cat = col_name_by_potential_type[potential_type]
    g = sns.catplot(
        d_.query("task==@task_name and cat==@filter_cat"),
        kind="bar",
        x="short_name",
        y=val_col,
        hue="perc",
        row="retry_mode" if split_retry_modes else None,
        col="retry_model" if split_retry_models else None,
        aspect=max(len(d_.query("cat==@filter_cat")["short_name"].unique()) / 8, 1),
        order=order_models(d_, col_name_by_wastage_type[WastageTypes.Total])[
            "short_name"
        ],
    )
    g.figure.suptitle(f"{task_name} {potential_type.name.lower()} potential")
    for ax in g.axes.ravel():
        ax.tick_params(axis="x", which="both", rotation=45)
    g.tight_layout()
    g.set_titles(template="{row_name}, {col_name}")
    plt.show()


@timed_function()
def plot_overall_wastage(
    df,
    val_col="val",
    do_percent=False,
    wastage_type: WastageTypes = WastageTypes.Total,
    split_retry_modes: bool = True,
    split_retry_models: bool = True,
) -> None:
    sns.set_theme(style="darkgrid")
    # todo this should instead first aggregate total wastage per method and training percentage and then convert that to percentages of own waste vs max waste
    if do_percent:
        val_col = "as_p"
        d_ = df_to_percentages(df.copy(), perc_col=val_col)
    else:
        d_ = df
    filter_cat = col_name_by_wastage_type[wastage_type]
    g = sns.catplot(
        d_.query("cat==@filter_cat")
        .groupby(by=["short_name", "perc", "retry_mode", "retry_model"])[val_col]
        .mean()
        .reset_index(),
        kind="bar",
        x="short_name",
        y=val_col,
        hue="perc",
        row="retry_mode" if split_retry_modes else None,
        col="retry_model" if split_retry_models else None,
        aspect=max(len(d_.query("cat==@filter_cat")["short_name"].unique()) / 8, 1),
        order=order_models(d_, col_name_by_wastage_type[WastageTypes.Total])[
            "short_name"
        ],
    )
    g.figure.suptitle(f"Average {wastage_type.name.lower()} wastage")
    for ax in g.axes.ravel():
        ax.tick_params(axis="x", which="both", rotation=45)
    g.tight_layout()
    g.set_titles(template="{row_name}, {col_name}")
    plt.show()


@timed_function()
def plot_overall_efficacy(
    df,
    val_col="val",
    do_percent=False,
    split_retry_modes: bool = True,
    split_retry_models: bool = True,
) -> None:
    sns.set_theme(style="darkgrid")
    # todo this should instead first aggregate total wastage per method and training percentage and then convert that to percentages of own waste vs max waste
    if do_percent:
        val_col = "as_p"
        d_ = df_to_percentages(df.copy(), perc_col=val_col)
    else:
        d_ = df
    g = sns.catplot(
        d_.query("cat=='Efficacy'")
        .groupby(by=["short_name", "perc", "retry_mode", "retry_model"])[val_col]
        .mean()
        .reset_index(),
        kind="bar",
        x="short_name",
        y=val_col,
        hue="perc",
        row="retry_mode" if split_retry_modes else None,
        col="retry_model" if split_retry_models else None,
        aspect=max(len(d_.query("cat=='Efficacy'")["short_name"].unique()) / 8, 1),
        order=order_models(d_, col_name_by_wastage_type[WastageTypes.Total])[
            "short_name"
        ],
    )
    g.figure.suptitle(f"Average efficacy")
    for ax in g.axes.ravel():
        ax.tick_params(axis="x", which="both", rotation=45)
    g.tight_layout()
    g.set_titles(template="{row_name}, {col_name}")
    plt.show()


@timed_function()
def plot_overall_potential(
    df,
    val_col="val",
    do_percent=False,
    potential_type: PotentialTypes = PotentialTypes.With_retries,
    split_retry_modes: bool = True,
    split_retry_models: bool = True,
) -> None:
    sns.set_theme(style="darkgrid")
    # todo this should instead first aggregate total wastage per method and training percentage and then convert that to percentages of own waste vs max waste
    if do_percent:
        val_col = "as_p"
        d_ = df_to_percentages(df.copy(), perc_col=val_col)
    else:
        d_ = df
    filter_cat = col_name_by_potential_type[potential_type]
    g = sns.catplot(
        d_.query("cat==@filter_cat")
        .groupby(by=["short_name", "perc", "retry_mode", "retry_model"])[val_col]
        .mean()
        .reset_index(),
        kind="bar",
        x="short_name",
        y=val_col,
        hue="perc",
        row="retry_mode" if split_retry_modes else None,
        col="retry_model" if split_retry_models else None,
        aspect=max(len(d_.query("cat==@filter_cat")["short_name"].unique()) / 8, 1),
        order=order_models(d_, col_name_by_wastage_type[WastageTypes.Total])[
            "short_name"
        ],
    )
    g.figure.suptitle(f"Average {potential_type.name.lower()} potential")
    for ax in g.axes.ravel():
        ax.tick_params(axis="x", which="both", rotation=45)
    g.tight_layout()
    g.set_titles(template="{row_name}, {col_name}")
    plt.show()


# unused for now
def df_to_percentages(df: pds.DataFrame, perc_col="as_p", val_col="val"):
    # todo completely scuffed
    # df[perc_col] = (
    #     df.groupby(by=["cat", "perc", "task"])["val"]
    #     .apply(lambda x: x / x.max())
    #     .reset_index()
    #     .set_index(["level_3"])["val"]
    # )
    df[perc_col] = df[val_col] / df.groupby(by=["cat", "perc", "task"])[
        "val"
    ].transform("max")
    return df
