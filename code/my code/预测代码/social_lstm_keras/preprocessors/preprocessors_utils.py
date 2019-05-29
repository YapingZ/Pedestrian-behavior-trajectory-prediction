from general_utils import DatasetKind, _check_dataset_kind
from preprocessors.eth_dataset_preprocessor import EthDatasetPreprosessor
from preprocessors.ucy_dataset_preprocessor import UcyDatasetPreprocessor


def create_dataset_preprocessor(data_dir, dataset_kind):
    dataset_kind = _check_dataset_kind(dataset_kind)
    # ETH dataset
    if dataset_kind in (DatasetKind.eth, DatasetKind.hotel):
        return EthDatasetPreprosessor(data_dir, dataset_kind)

    if dataset_kind in (DatasetKind.zara1, DatasetKind.zara2, DatasetKind.zara3, DatasetKind.ucy, DatasetKind.students003):
        return UcyDatasetPreprocessor(data_dir, dataset_kind)

    raise ValueError("dataset_kind")
