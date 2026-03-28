"""radiology_dataset_db package."""

# from importlib import import_module

# __all__ = [
#     "extract_radiology_dataset_info_with_agent",
#     "extract_scrnaseq_dataset_info_with_agent",
#     "extract_bulk_genomics_dataset_info_with_agent",
#     "extract_spatial_transcriptomics_dataset_info_with_agent",
# ]

# _EXPORTS = {
#     "extract_radiology_dataset_info_with_agent": (
#         "radiology_dataset_db.extract_radiology_dataset_information_llm",
#         "extract_radiology_dataset_info_with_agent",
#     ),
#     "extract_scrnaseq_dataset_info_with_agent": (
#         "radiology_dataset_db.extract_scrnaseq_dataset_information_llm",
#         "extract_scrnaseq_dataset_info_with_agent",
#     ),
#     "extract_bulk_genomics_dataset_info_with_agent": (
#         "radiology_dataset_db.extract_bulk_genomics_dataset_information_llm",
#         "extract_bulk_genomics_dataset_info_with_agent",
#     ),
#     "extract_spatial_transcriptomics_dataset_info_with_agent": (
#         "radiology_dataset_db.extract_spatial_transcriptomics_dataset_information_llm",
#         "extract_spatial_transcriptomics_dataset_info_with_agent",
#     ),
# }


# def __getattr__(name: str):
#     if name in _EXPORTS:
#         module_name, attr_name = _EXPORTS[name]
#         module = import_module(module_name)
#         value = getattr(module, attr_name)
#         globals()[name] = value
#         return value
#     raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from .extract_bulk_genomics_dataset_information_llm import extract_bulk_genomics_dataset_info_with_agent
from .extract_radiology_dataset_information_llm import extract_radiology_dataset_info_with_agent
from .extract_scrnaseq_dataset_information_llm import extract_scrnaseq_dataset_info_with_agent
from .extract_spatial_transcriptomics_dataset_information_llm import extract_spatial_transcriptomics_dataset_info_with_agent

__version__ = "0.1.0"
__author__ = "Joseph Rich"
__email__ = "josephrich98@gmail.com"
