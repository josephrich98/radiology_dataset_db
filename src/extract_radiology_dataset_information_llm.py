# src/build_dataset_table.py

import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from src.config import (ADD_TEXT_EXTRACT, EXTRACTION_AGENT_INSTRUCTIONS,
                        EXTRACTION_INSTRUCTIONS, LOG_LEVEL, MODEL)
from src.is_database_paper_classifier_llm import llm_thinks_not_dataset_paper

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

load_dotenv()

# -----------------------------
# SCHEMA
# -----------------------------
class Modality(str, Enum):
    CT = "CT"
    MRI = "MRI"
    XRAY = "X-ray"
    US = "US"
    PET = "PET"

class BodyRegion(str, Enum):
    BRAIN = "brain"
    CHEST = "chest"
    ABDOMEN = "abdomen"
    PELVIS = "pelvis"
    LIMBS = "limbs"

class AdditionalData(str, Enum):
    REPORTS = "reports"
    CAPTIONS = "captions"
    SEGMENTATION = "segmentation"
    GENOMICS = "genomics"
    HISTOLOGY = "histology"
    VQA = "VQA"


# see RSNA schema: https://github.com/RSNA/ATLAS/blob/main/model.json - consider expanding modalities, and adding file format, resolution, 2D vs 3D (would require reading full text files and possibly following links to dataset repositories, which is more complex but could be future work)
class RadiologyDataset(BaseModel):
    name: Optional[str] = Field(default=None)
    num_images: Optional[int] = None
    num_patients: Optional[int] = None
    modalities: List[Modality] = Field(default_factory=list)
    body_regions: List[BodyRegion] = Field(default_factory=list)
    additional_data: List[AdditionalData] = Field(default_factory=list)
    dataset_link: Optional[str] = None
    paper_title: Optional[str] = None
    # paper_abstract: Optional[str] = None
    paper_link: Optional[str] = None
    paper_year: Optional[int] = None
    paper_authors: List[str] = Field(default_factory=list)
    paper_journal: Optional[str] = None
    pmid: Optional[str] = None
    paper_citation_count: Optional[int] = None
    mesh_terms: List[str] = Field(default_factory=list)
    pubmed_matches: Optional[List[List[str]]] = None



# -----------------------------
# DEPENDENCIES
# -----------------------------
@dataclass
class ExtractionDeps:
    title: str
    abstract: str

# -----------------------------
# AGENT
# -----------------------------
dataset_agent = Agent(
    MODEL,
    deps_type=ExtractionDeps,
    output_type=RadiologyDataset,
    instructions=EXTRACTION_INSTRUCTIONS
)

@dataset_agent.instructions
async def add_text(ctx: RunContext[ExtractionDeps]) -> str:
    return f"""
Text:
Title: {ctx.deps.title}
Abstract: {ctx.deps.abstract}

{ADD_TEXT_EXTRACT}
"""

def serialize_dataset_output(dataset: Union[RadiologyDataset, str]) -> str:
    if isinstance(dataset, RadiologyDataset):
        return dataset.model_dump_json(indent=4)
    return str(dataset)


def normalize_dataset_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None

    normalized = name.casefold()
    normalized = re.sub(r"\bdatabase\b", "", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"[^a-z0-9]", "", normalized)
    return normalized or None

STOPWORDS = {
    "a", "an", "the", "of", "and", "or", "for", "to", "in", "on", "with",
    "by", "from", "at", "as", "is", "are", "this", "that", "database", "dataset", "data", "open", "large", "scale", "public",
}

def normalize_tokens(text: str):
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if t not in STOPWORDS}

def name_matches_title(dataset_name: str, title: str) -> bool:
    """
    Returns True if the dataset name has meaningful token overlap with the title,
    OR if the dataset name appears as a substring in the title (handles hyphenated
    names like 'MIMIC-CXR' that tokenize differently).
    """
    # Direct substring match (case-insensitive) — catches "MIMIC-CXR" in title
    if dataset_name.lower() in title.lower():
        return True

    # Token overlap (original logic)
    name_tokens = normalize_tokens(dataset_name)
    title_tokens = normalize_tokens(title)
    if name_tokens & title_tokens:
        return True

    # Acronym check: if dataset name looks like an acronym (all caps / short),
    # check if it appears anywhere in the combined text
    if re.match(r"^[A-Z0-9\-]{2,10}$", dataset_name):
        if dataset_name.upper() in title.upper():
            return True

    return False



# -----------------------------
# LLM EXTRACTION (ASYNC)
# -----------------------------
async def extract_radiology_dataset_info_with_agent(
    title: str,
    abstract: str,
    publication_metadata: Optional[dict] = None,
):
    if not title or not abstract:
        logger.debug("Missing title or abstract, skipping extraction.")
        return None

    deps = ExtractionDeps(title=title, abstract=abstract)

    try:
        unlikely_dataset_paper = await llm_thinks_not_dataset_paper(title, abstract)
        if unlikely_dataset_paper:
            #* 1. Classification LLM thinks this is unlikely to be a dataset paper → reject immediately without extraction (saves cost and prevents hallucination of dataset info when paper is not about a dataset)
            logger.debug("LLM thinks this is unlikely to be a dataset paper, rejecting without extraction.")
            return None

        #* Now run the extraction agent.
        result = await dataset_agent.run(
            EXTRACTION_AGENT_INSTRUCTIONS, deps=deps
        )

        output = result.output
        logger.debug("LLM output: %s", output)
        if isinstance(output, RadiologyDataset):
            output.paper_title = title
            # output.paper_abstract = abstract
            
            publication_metadata = publication_metadata or {}
            output.paper_link = publication_metadata.get("link")
            output.paper_year = publication_metadata.get("year")
            output.paper_authors = publication_metadata.get("authors")
            output.paper_journal = publication_metadata.get("journal")
            output.pmid = publication_metadata.get("pmid")
            output.paper_citation_count = publication_metadata.get("citation_count")
            output.mesh_terms = publication_metadata.get("mesh_terms")
            output.pubmed_matches = publication_metadata.get("pubmed_matches")

            if not output.name:
                #* 2. No dataset name extracted → reject (this is a strong signal that the paper is not actually about a dataset, since the name is usually the easiest thing to extract and often appears in the title)
                logger.debug("No dataset name extracted, rejecting output.")
                for msg in result.all_messages():
                    logger.debug(msg)
                return None  # reject if no dataset name extracted
            # else:
            #     if not name_matches_title(output.name, title):  # reject if dataset name doesn't have meaningful match with title
            #         #* 3. Dataset name extracted but doesn't match title → reject (this is a signal that the extracted name may be spurious and not actually the name of a dataset created by the paper, especially if the LLM also thinks it's not likely to be a dataset paper)
            #         logger.debug(
            #             "Dataset name '%s' has no meaningful match with title and LLM does not think it's a dataset paper, rejecting output.",
            #             output.name
            #         )
            #         return None

        return output

    except Exception as e:
        logger.error("LLM extraction failed: %s", e)
        return None
