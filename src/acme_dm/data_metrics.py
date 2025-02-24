import logging
import os
from typing import Callable, Union

import polars as pl
from acme_dw import DW, DatasetMetadata, DatasetPrefix

logger = logging.getLogger(__name__)


def get_dw():
    bucket_name = os.environ.get("DW_BUCKET_NAME")
    if bucket_name is None:
        raise ValueError("DW_BUCKET_NAME environment variable not set")
    return DW(bucket_name)


def _validate_inputs(
    metrics_metadata: DatasetMetadata,
    new_data: Union[pl.DataFrame, DatasetMetadata],
    metrics_function: Callable,
) -> None:
    # Input type validation
    assert (
        isinstance(metrics_metadata, DatasetMetadata)
        and metrics_metadata.df_type == "polars"
    ), 'metrics_metadata must be DatasetMetadata with df_type="polars"'

    assert isinstance(new_data, (pl.DataFrame, DatasetMetadata)), (
        "new_data must be a DatasetMetadata or a pl.DataFrame"
    )
    if isinstance(new_data, DatasetMetadata):
        assert new_data.df_type == "polars", 'new_data must use df_type="polars"'

    assert callable(metrics_function), "metrics_function must be a callable function"


def add_new_metrics(
    metrics_metadata: DatasetMetadata,
    new_data: Union[pl.DataFrame, DatasetMetadata],
    metrics_function: Callable,
):
    """Calculate new metrics and combine with existing metrics.

    Args:
        metrics_metadata (DatasetMetadata): Metadata for existing metrics dataset.
            Must use df_type="polars".
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
        FileNotFoundError: If existing metrics are not found
    Returns:
        None: Results are written directly to the data warehouse specified by metrics_metadata
    """
    _validate_inputs(metrics_metadata, new_data, metrics_function)

    # load existing metrics from dw
    dw = get_dw()
    try:
        existing_metrics = dw.read_df(metrics_metadata)
    except FileNotFoundError:
        logger.error(
            "Existing metrics not found! Use `create_metrics` to create metrics from scratch."
        )
        raise

    # load new_data from dw if new_data is a DatasetMetadata
    if isinstance(new_data, DatasetMetadata):
        df = dw.read_df(new_data)
    else:
        df = new_data

    # Calculate new metrics
    new_metrics = metrics_function(existing_metrics, df)

    if existing_metrics.shape[0] > 0 and existing_metrics.shape[1] > 0:
        # Check if new_metrics have the same columns as existing_metrics
        common_cols = set(existing_metrics.columns).intersection(
            set(new_metrics.columns)
        )
        if len(common_cols) != len(existing_metrics.columns):
            msg = (
                "New metrics do not have the same columns as existing metrics"
                f"\nexisting: {existing_metrics.columns}\nnew: {new_metrics.columns}\ncommon: {common_cols}"
            )
            raise ValueError(msg)

        combined_metrics = pl.concat([existing_metrics, new_metrics])
    else:
        combined_metrics = new_metrics

    # Save updated metrics
    # TODO: this rewrites the entire dataset, which is not efficient
    dw.write_df(combined_metrics, metrics_metadata)


def create_metrics(
    metrics_metadata: DatasetMetadata,
    new_data: Union[pl.DataFrame, DatasetMetadata, DatasetPrefix],
    metrics_function: Callable,
):
    """Calculate metrics from scratch and write to the data warehouse.

    Args:
        metrics_metadata (DatasetMetadata): Metadata defining where and how to store the
            calculated metrics. Must have df_type="polars".
        new_data (Union[pl.DataFrame, DatasetMetadata, DatasetPrefix]): Data to calculate
            metrics from. Can be:
            - A polars DataFrame containing the data
            - DatasetMetadata pointing to data in the warehouse
            - DatasetPrefix pointing to data in the warehouse
        metrics_function (Callable): Function that calculates metrics. Should take two
            arguments:
            - First argument: Empty polars DataFrame (for consistency with update_metrics)
            - Second argument: DataFrame containing new data
            Returns calculated metrics as a polars DataFrame.

    Raises:
        ValueError: If new_data is not one of the accepted types.
        AssertionError: If metrics_metadata is not DatasetMetadata with df_type="polars",
            if new_data metadata doesn't use df_type="polars", or if metrics_function
            is not callable.

    Returns:
        None. Results are written to the data warehouse using metrics_metadata.
    """
    # validate types
    assert (
        isinstance(metrics_metadata, DatasetMetadata)
        and metrics_metadata.df_type == "polars"
    ), 'metrics_metadata must be DatasetMetadata with df_type="polars"'

    if isinstance(new_data, (DatasetMetadata, DatasetPrefix)):
        assert new_data.df_type == "polars", 'new_data must use df_type="polars"'

    assert callable(metrics_function), "metrics_function must be a callable function"

    # load new_data from dw if new_data is a DatasetMetadata/DatasetPrefix
    dw = get_dw()
    if isinstance(new_data, DatasetMetadata):
        df = dw.read_df(new_data)
    elif isinstance(new_data, DatasetPrefix):
        df = dw.read_dataset(new_data)
    elif isinstance(new_data, pl.DataFrame):
        df = new_data
    else:
        raise ValueError(
            "new_data must be a DatasetMetadata, DatasetPrefix, or a pl.DataFrame"
        )

    new_metrics = metrics_function(pl.DataFrame(), df)

    dw.write_df(new_metrics, metrics_metadata)
