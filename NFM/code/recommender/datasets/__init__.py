import os

import recommender.models

from recommender.datasets.frappe import FrappeDataset
import recommender.utils as utils

datasets = {
    "frappe": FrappeDataset,
}


def build_dataset(
    dataset_name,
    output_dir=None,
    verbose=True,
    maybe_download=True,
    maybe_preprocess=True,
):
    """Build Dataset instance.

  Args:
    dataset_name: A string indicating the dataset to be build.
    output_dir: A string indicating the output directory used to store auxiliary files. If None,
    then the default path defined in `utils.py` will be used.
    maybe_download: A boolean indicating if the files should be downloaded.
    maybe_preprocess: A boolean indicating if the files should be preprocessed.

  Returns:
    A Dataset instance.
  """
    if not output_dir:
        output_dir = os.path.join(utils.DEFAULT_OUTPUT_DIR, dataset_name)

    if dataset_name in datasets:
        dataset = datasets[dataset_name](
            output_dir=output_dir, dataset_name=dataset_name, verbose=verbose
        )
    else:
        raise ValueError("Unkown dataset {}".format(dataset_name))

    if maybe_download:
        dataset.maybe_download()

    if maybe_preprocess:
        dataset.maybe_preprocess()

    # Some datasets need a building step.
    dataset.build()

    return dataset
