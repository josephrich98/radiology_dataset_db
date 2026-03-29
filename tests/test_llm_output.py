import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import (
    _bulk_genomics_paper_ground_truth,
    _radiology_paper_ground_truth,
    _scrnaseq_paper_ground_truth,
    _spatial_transcriptomics_paper_ground_truth,
)

radiology_module = pytest.importorskip(
    "radiology_dataset_db.extract_radiology_dataset_information_llm",
    reason="radiology extraction module dependencies are required",
)
scrnaseq_module = pytest.importorskip(
    "radiology_dataset_db.extract_scrnaseq_dataset_information_llm",
    reason="scRNA-seq extraction module dependencies are required",
)
bulk_genomics_module = pytest.importorskip(
    "radiology_dataset_db.extract_bulk_genomics_dataset_information_llm",
    reason="bulk genomics extraction module dependencies are required",
)
spatial_transcriptomics_module = pytest.importorskip(
    "radiology_dataset_db.extract_spatial_transcriptomics_dataset_information_llm",
    reason="spatial transcriptomics extraction module dependencies are required",
)

#* add additional modality constants here for other modalities, e.g. genomics, pathology, etc


RADIOLOGY = "radiology"
SCRNASEQ = "scrnaseq"
BULK_GENOMICS = "bulk_genomics"
SPATIAL_TRANSCRIPTOMICS = "spatial_transcriptomics"

#* add additional modality constants here for other modalities, e.g. genomics, pathology, etc

NUM_TRIES_AGENT = 8
INTEGRATION_CALL_TIMEOUT_SECONDS = 30


def _normalize_name(name):
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


def _has_integration_dependencies():
    required = ["dotenv", "pydantic_ai", "Bio", "pydantic"]
    for pkg in required:
        try:
            __import__(pkg)
        except Exception:
            return False
    return True


def get_modality_info(modality):
    if modality == RADIOLOGY:
        return {
            "module": radiology_module,
            "paper_ground_truth": _radiology_paper_ground_truth(),
            "extract_dataset_info_with_agent": radiology_module.extract_radiology_dataset_info_with_agent,
            "dataset_class": radiology_module.RadiologyDataset,
            "serialize_dataset_output": radiology_module.serialize_dataset_output,
            "enum_fields": {
                "modalities": radiology_module.Modality,
                "body_regions": radiology_module.BodyRegion,
                "additional_data": radiology_module.AdditionalData,
            },
        }
    if modality == SCRNASEQ:
        return {
            "module": scrnaseq_module,
            "paper_ground_truth": _scrnaseq_paper_ground_truth(),
            "extract_dataset_info_with_agent": scrnaseq_module.extract_scrnaseq_dataset_info_with_agent,
            "dataset_class": scrnaseq_module.ScRNASeqDataset,
            "serialize_dataset_output": scrnaseq_module.serialize_dataset_output,
            "enum_fields": {
                "sequencing_technology": scrnaseq_module.SequencingTechnology,
                "cell_or_nuclei": scrnaseq_module.CellOrNuclei,
            },
        }
    if modality == BULK_GENOMICS:
        return {
            "module": bulk_genomics_module,
            "paper_ground_truth": _bulk_genomics_paper_ground_truth(),
            "extract_dataset_info_with_agent": bulk_genomics_module.extract_bulk_genomics_dataset_info_with_agent,
            "dataset_class": bulk_genomics_module.BulkGenomicsDataset,
            "serialize_dataset_output": bulk_genomics_module.serialize_dataset_output,
            "enum_fields": {
                "modalities": bulk_genomics_module.BulkGenomicsModality,
            },
        }
    if modality == SPATIAL_TRANSCRIPTOMICS:
        return {
            "module": spatial_transcriptomics_module,
            "paper_ground_truth": _spatial_transcriptomics_paper_ground_truth(),
            "extract_dataset_info_with_agent": spatial_transcriptomics_module.extract_spatial_transcriptomics_dataset_info_with_agent,
            "dataset_class": spatial_transcriptomics_module.SpatialTranscriptomicsDataset,
            "serialize_dataset_output": spatial_transcriptomics_module.serialize_dataset_output,
            "enum_fields": {
                "sequencing_technology": spatial_transcriptomics_module.SequencingTechnology,
                "cell_or_nuclei": spatial_transcriptomics_module.CellOrNuclei,
            },
        }
    #* add additional modality info here for other modalities, e.g. genomics, pathology, etc
    raise ValueError(f"Unsupported modality: {modality}")


def _build_dataset_from_truth(modality, paper):
    info = get_modality_info(modality)
    expected = dict(paper["expected"])

    for field_name, enum_cls in info["enum_fields"].items():
        value = expected.get(field_name)
        if value is None:
            continue
        if isinstance(value, list):
            expected[field_name] = [enum_cls(item) for item in value]
        else:
            expected[field_name] = enum_cls(value)

    return info["dataset_class"](
        **expected,
        paper_title=paper["title"],
        paper_link=paper.get("link"),
        paper_year=paper.get("year"),
        paper_authors=paper.get("authors") or [],
        paper_journal=paper.get("journal"),
        paper_citation_count=paper.get("citation_count"),
    )


def _assert_serialized_dataset(modality, dataset, paper):
    info = get_modality_info(modality)
    expected = paper["expected"]
    output = info["serialize_dataset_output"](dataset)
    parsed = json.loads(output)

    for key, expected_value in expected.items():
        assert parsed[key] == expected_value

    assert parsed["paper_title"] == paper["title"]
    assert parsed["paper_link"] == paper.get("link")
    assert output.startswith("{\n")


def _run_serialize_test_for_modality(modality, paper_key):
    info = get_modality_info(modality)
    paper = info["paper_ground_truth"][paper_key]
    if paper.get("expected") is None:
        pytest.skip(f"No expected output defined for {paper_key}")

    dataset = _build_dataset_from_truth(modality, paper)
    _assert_serialized_dataset(modality, dataset, paper)


def _run_agent_integration_test_for_modality(modality, paper_key):
    if not (os.getenv("VLLM_PORT") or os.getenv("OPENAI_API_KEY")):
        pytest.skip("Integration test requires VLLM_PORT or OPENAI_API_KEY")
    if not _has_integration_dependencies():
        pytest.skip("Integration test requires dotenv, pydantic_ai, Bio, and pydantic")

    info = get_modality_info(modality)
    paper = info["paper_ground_truth"][paper_key]

    publication_metadata = {
        "link": paper.get("link"),
        "year": paper.get("year"),
        "authors": paper.get("authors"),
        "journal": paper.get("journal"),
        "citation_count": paper.get("citation_count"),
        "pmid": paper.get("pmid"),
    }

    dataset = None
    for _ in range(NUM_TRIES_AGENT):
        try:
            dataset = asyncio.run(
                asyncio.wait_for(
                    info["extract_dataset_info_with_agent"](
                        title=paper["title"],
                        abstract=paper["abstract"],
                        publication_metadata=publication_metadata,
                    ),
                    timeout=INTEGRATION_CALL_TIMEOUT_SECONDS,
                )
            )
        except asyncio.TimeoutError:
            pytest.fail(
                f"Extraction timed out after {INTEGRATION_CALL_TIMEOUT_SECONDS}s "
                f"for modality '{modality}', paper '{paper_key}'."
            )
        if dataset is not None:
            break

    assert dataset is not None
    assert isinstance(dataset, info["dataset_class"])
    assert dataset.paper_title == paper["title"]
    assert dataset.paper_link == publication_metadata["link"]
    assert dataset.name

    normalized_name = _normalize_name(dataset.name)
    name_aliases = paper.get("name_aliases", [paper_key])
    if paper_key not in name_aliases:
        name_aliases.append(paper_key)
    assert any(_normalize_name(alias) in normalized_name for alias in name_aliases)


#$ Radiology
RADIOLOGY_PAPER_KEYS = tuple(get_modality_info(RADIOLOGY)["paper_ground_truth"].keys())
@pytest.mark.parametrize("paper_key", RADIOLOGY_PAPER_KEYS)
def test_serialize_radiology_dataset_against_ground_truth(paper_key):
    _run_serialize_test_for_modality(RADIOLOGY, paper_key)

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("paper_key", RADIOLOGY_PAPER_KEYS)
def test_extract_radiology_dataset_info_with_agent_integration(paper_key):
    _run_agent_integration_test_for_modality(RADIOLOGY, paper_key)


# #$ scRNA-seq
# SCRNASEQ_PAPER_KEYS = tuple(get_modality_info(SCRNASEQ)["paper_ground_truth"].keys())
# @pytest.mark.parametrize("paper_key", SCRNASEQ_PAPER_KEYS)
# def test_serialize_scrnaseq_dataset_against_ground_truth(paper_key):
#     _run_serialize_test_for_modality(SCRNASEQ, paper_key)

# @pytest.mark.integration
# @pytest.mark.slow
# @pytest.mark.parametrize("paper_key", SCRNASEQ_PAPER_KEYS)
# def test_extract_scrnaseq_dataset_info_with_agent_integration(paper_key):
#     _run_agent_integration_test_for_modality(SCRNASEQ, paper_key)


# #$ bulk genomics
# BULK_GENOMICS_PAPER_KEYS = tuple(get_modality_info(BULK_GENOMICS)["paper_ground_truth"].keys())
# @pytest.mark.parametrize("paper_key", BULK_GENOMICS_PAPER_KEYS)
# def test_serialize_bulk_genomics_dataset_against_ground_truth(paper_key):
#     _run_serialize_test_for_modality(BULK_GENOMICS, paper_key)

# @pytest.mark.integration
# @pytest.mark.slow
# @pytest.mark.parametrize("paper_key", BULK_GENOMICS_PAPER_KEYS)
# def test_extract_bulk_genomics_dataset_info_with_agent_integration(paper_key):
#     _run_agent_integration_test_for_modality(BULK_GENOMICS, paper_key)

# #$ spatial transcriptomics
# SPATIAL_TRANSCRIPTOMICS_PAPER_KEYS = tuple(get_modality_info(SPATIAL_TRANSCRIPTOMICS)["paper_ground_truth"].keys())
# @pytest.mark.parametrize("paper_key", SPATIAL_TRANSCRIPTOMICS_PAPER_KEYS)
# def test_serialize_spatial_transcriptomics_dataset_against_ground_truth(paper_key):
#     _run_serialize_test_for_modality(SPATIAL_TRANSCRIPTOMICS, paper_key)

# @pytest.mark.integration
# @pytest.mark.slow
# @pytest.mark.parametrize("paper_key", SPATIAL_TRANSCRIPTOMICS_PAPER_KEYS)
# def test_extract_spatial_transcriptomics_dataset_info_with_agent_integration(paper_key):
#     _run_agent_integration_test_for_modality(SPATIAL_TRANSCRIPTOMICS, paper_key)


#* add additional test functions for other modalities here, e.g. genomics, pathology, etc



@pytest.mark.integration
@pytest.mark.slow
def test_build_db_script_integration_runs_with_max10_min0(tmp_path):
    if not os.getenv("ENTREZ_EMAIL"):
        pytest.skip("Integration test requires ENTREZ_EMAIL")
    if not (os.getenv("VLLM_PORT") or os.getenv("OPENAI_API_KEY")):
        pytest.skip("Integration test requires VLLM_PORT or OPENAI_API_KEY")
    if not _has_integration_dependencies():
        pytest.skip("Integration test requires dotenv, pydantic_ai, Bio, and pydantic")

    repo_root = Path(__file__).resolve().parents[1]
    output_path = tmp_path / "db.csv"
    command = [
        sys.executable,
        "scripts/build_db.py",
        "-m", "radiology",
        "-o",
        str(output_path),
        "-max",
        "10",
        "-min",
        "0",
    ]

    result = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=1800,
    )

    assert result.returncode == 0, (
        "build_db.py integration command failed.\n"
        f"Command: {' '.join(command)}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert output_path.exists(), "Output CSV file was not created by build_db.py"
