# acme-dm

Compute metrics from data

# Problem

There is often a need to monitor data quality of some dataset.
This can be done by calculating some new metrics from existing data and reporting the value of the metric.

# Features

* Compute arbitrary metrics from data
* Icrementally build metrics as dataset expands
* Use `polars` for efficient computation
* Load & Store metrics in `acme-dw`
* Optionally load new data for metrics calculation from `acme-dw`

# Dev environment

The project comes with a python development environment.
To generate it, after checking out the repo run:

    chmod +x create_env.sh

Then to generate the environment (or update it to latest version based on state of `uv.lock`), run:

    ./create_env.sh

This will generate a new python virtual env under `.venv` directory. You can activate it via:

    source .venv/bin/activate

If you are using VSCode, set to use this env via `Python: Select Interpreter` command.

## Example usage

```python
import polars as pl
from acme_dm.data_metrics import add_new_metrics
from acme_dw import DatasetMetadata

# Define a sample metrics function
def sample_metrics_function(existing_metrics, new_data):
    # Example: calculate the mean of a column
    new_metrics = new_data.select(pl.col("value").mean().alias("mean_value"))
    return new_metrics

# Create sample new data
new_data = pl.DataFrame({
    "value": [1, 2, 3, 4, 5]
})

# Optionally, provide existing metrics metadata
metrics_metadata = DatasetMetadata(
    source="metrics_store",
    name="sample_metrics",
    version="v1",
    process_id="sample_process",
    partitions=["date=20100101"],
    file_name="metrics.parquet",
    file_type="parquet",
    df_type="polars"
)

# Call the add_new_metrics function
add_new_metrics(
    metrics_metadata=metrics_metadata,
    new_data=new_data,
    metrics_function=sample_metrics_function
)
```

# Project template

This project has been setup with `acme-project-create`, a python code template library.

# Required setup post use

* Enable GitHub Pages to be published via [GitHub Actions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site#publishing-with-a-custom-github-actions-workflow) by going to `Settings-->Pages-->Source`
* Create `release-pypi` environment for [GitHub Actions](https://docs.github.com/en/actions/managing-workflow-runs-and-deployments/managing-deployments/managing-environments-for-deployment#creating-an-environment) to enable uploads of the library to PyPi. Set protections on what tags can deploy to this environment (Point 10). Set it to tags following pattern `v*`.
* Setup auth to PyPI for the GitHub Action implemented in `.github/workflows/release.yml` via [Trusted Publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/) `uv publish` [doc](https://docs.astral.sh/uv/guides/publish/#publishing-your-package)
* Once you create the python environment for the first time add the `uv.lock` file that will be created in project directory to the source control and update it each time environment is rebuilt
* In order not to replicate documentation in `docs/docs/index.md` file and `README.md` in root of the project setup a symlink from `README.md` file to the `index.md` file.
To do this, from `docs/docs` dir run:

    ln -sf ../../README.md index.md
* Run `pre-commit install` to install the pre-commit hooks.