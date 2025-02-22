import os
from typing import Callable, Optional, Union

import polars as pl
from acme_dw import DW, DatasetMetadata


def get_dw():
    bucket_name = os.environ.get("DW_BUCKET_NAME")
    if bucket_name is None:
        raise ValueError("DW_BUCKET_NAME environment variable not set")
    return DW(bucket_name)


def add_new_metrics(
    metrics_metadata: Optional[DatasetMetadata],
    new_data: Union[pl.DataFrame, DatasetMetadata],
    metrics_function: Callable,
):
    """Calculate new metrics and combine with existing metrics.

    Args:
        metrics_metadata (Optional[DatasetMetadata]): Metadata for existing metrics dataset.
            Must use df_type="polars" if provided.
        new_data (Union[pl.DataFrame, DatasetMetadata]): New data to calculate metrics from.
            Can be either a Polars DataFrame or DatasetMetadata with df_type="polars".
        metrics_function (Callable): Function that calculates metrics. Should take two
            arguments (existing_metrics and new_data) and return a Polars DataFrame with
            the same schema as existing_metrics.
    Raises:
        ValueError: If:
            - metrics_metadata is provided but not of type DatasetMetadata with df_type="polars"
            - new_data is not a Polars DataFrame or DatasetMetadata with df_type="polars"
            - metrics_function is not callable
            - New metrics have different columns than existing metrics
    Returns:
        None: Results are written directly to the data warehouse specified by metrics_metadata
    """
    # check types
    if (
        isinstance(metrics_metadata, DatasetMetadata)
        and metrics_metadata.df_type != "polars"
    ):
        raise ValueError('metrics_metadata must use df_type="polars"')
    elif metrics_metadata is not None and not isinstance(
        metrics_metadata, DatasetMetadata
    ):
        raise ValueError("metrics_metadata must be None or DatasetMetadata type")

    if isinstance(new_data, DatasetMetadata) and new_data.df_type != "polars":
        raise ValueError('new_data must use df_type="polars"')
    elif not isinstance(new_data, pl.DataFrame) and not isinstance(
        new_data, DatasetMetadata
    ):
        raise ValueError("new_data must be a DatasetMetadata or a pl.DataFrame")

    if not callable(metrics_function):
        raise ValueError("metrics_function must be a callable function")

    # load existing metrics from dw
    dw = get_dw()
    if isinstance(metrics_metadata, DatasetMetadata):
        existing_metrics = dw.read_df(metrics_metadata)
    else:
        existing_metrics = pl.DataFrame()

    # load new_data from dw if new_data is a DatasetMetadata
    if isinstance(new_data, DatasetMetadata):
        new_data = dw.read_df(new_data)

    # Calculate new metrics
    new_metrics = metrics_function(existing_metrics, new_data)
    # TODO: print the metrics using correct format

    # Check if new_metrics have the same columns as existing_metrics
    if existing_metrics.shape[0] > 0:
        common_cols = set(existing_metrics.columns).intersection(
            set(new_metrics.columns)
        )
        if len(common_cols) != len(existing_metrics.columns):
            msg = (
                "New metrics do not have the same columns as existing metrics"
                f"\nexisting: {existing_metrics.columns}\nnew: {new_metrics.columns}\ncommon: {common_cols}"
            )
            raise ValueError(msg)

    # Combine existing and new metrics
    if existing_metrics.shape[0] > 0:
        combined_metrics = pl.concat([existing_metrics, new_metrics])
    else:
        combined_metrics = new_metrics

    # Save updated metrics
    # TODO: this rewrites the entire dataset, which is not efficient
    dw.write_df(combined_metrics, metrics_metadata)
