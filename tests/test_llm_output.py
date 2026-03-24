import asyncio
import json
import os

import pytest

from conftest import _paper_ground_truth, load_build_module

def _build_dataset_from_truth(module, paper):
    expected = paper["expected"]
    modalities = [getattr(module.Modality, m.replace("-", "").upper()) for m in expected["modalities"]]
    body_regions = [getattr(module.BodyRegion, b.upper()) for b in expected["body_regions"]]
    additional_data = [getattr(module.AdditionalData, a.upper()) for a in expected["additional_data"]]

    return module.RadiologyDataset(
        name=expected["name"],
        num_images=expected["num_images"],
        num_patients=expected["num_patients"],
        modalities=modalities,
        body_regions=body_regions,
        additional_data=additional_data,
        paper_title=paper["title"],
        paper_abstract=paper["abstract"],
        paper_link=paper.get("link"),
        paper_year=paper.get("year"),
        paper_authors=paper.get("authors") or [],
        paper_journal=paper.get("journal"),
        paper_citation_count=paper.get("citation_count")
    )


def _assert_serialized_dataset(module, dataset, paper):
    expected = paper["expected"]
    output = module.serialize_dataset_output(dataset)
    parsed = json.loads(output)

    assert parsed["name"] == expected["name"]
    assert parsed["num_images"] == expected["num_images"]
    assert parsed["num_patients"] == expected["num_patients"]
    assert parsed["modalities"] == expected["modalities"]
    assert parsed["body_regions"] == expected["body_regions"]
    assert parsed["additional_data"] == expected["additional_data"]
    assert parsed["paper_title"] == paper["title"]
    assert parsed["paper_link"] == paper["link"]
    assert expected["title_hint"] in parsed["paper_abstract"]
    assert module.name_matches_title(parsed["name"], parsed["paper_title"])
    assert output.startswith("{\n")


def _normalize_name(name):
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _has_integration_dependencies():
    required = ["dotenv", "pydantic_ai", "Bio", "pydantic"]

    for pkg in required:
        try:
            __import__(pkg)
        except Exception:
            return False

    return True

PAPER_KEYS = _paper_ground_truth().keys()  # eg ['radimagenet', 'mimic_cxr', 'uk_biobank', 'tcia', 'merlin']

@pytest.mark.parametrize("paper_key", PAPER_KEYS)
def test_serialize_dataset_against_ground_truth(monkeypatch, paper_key):
    module = load_build_module(monkeypatch)
    papers = _paper_ground_truth()
    paper = papers[paper_key]

    dataset = _build_dataset_from_truth(module, paper)
    _assert_serialized_dataset(module, dataset, paper)


NUM_TRIES = 5

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("paper_key", PAPER_KEYS)
def test_extract_with_agent_integration(monkeypatch, paper_key):
    if not (os.getenv("VLLM_PORT") or os.getenv("OPENAI_API_KEY")):
        pytest.skip("Integration test requires VLLM_PORT or OPENAI_API_KEY")
    if not _has_integration_dependencies():
        pytest.skip("Integration test requires dotenv, pydantic_ai, Bio, and pydantic")

    module = load_build_module(monkeypatch, module_name="build_database_table_live_under_test", live=True)
    papers = _paper_ground_truth()
    paper = papers[paper_key]

    title = paper["title"]
    abstract = paper["abstract"]
    publication_metadata = {
        "link": paper.get("link"),
        "year": paper.get("year"),
        "authors": paper.get("authors"),
        "journal": paper.get("journal"),
        "citation_count": paper.get("citation_count")
    }

    dataset = None
    for _ in range(NUM_TRIES):
        dataset = asyncio.run(module.extract_with_agent(title=title, abstract=abstract, publication_metadata=publication_metadata))
        if dataset is not None:
            break

    assert dataset is not None
    assert isinstance(dataset, module.RadiologyDataset)
    assert dataset.paper_title == title
    assert dataset.paper_link == publication_metadata["link"]
    assert dataset.name
    assert module.name_matches_title(dataset.name, title)

    normalized_name = _normalize_name(dataset.name)
    assert any(alias in normalized_name for alias in paper["name_aliases"])
