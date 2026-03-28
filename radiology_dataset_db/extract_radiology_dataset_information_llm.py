# src/build_dataset_table.py

import logging
import os
import re
from enum import Enum
from typing import List, Optional, Union

from dotenv import load_dotenv
from pydantic import Field
from pydantic_ai import Agent, RunContext

from radiology_dataset_db.config import (ADD_TEXT_EXTRACT, EXTRACTION_AGENT_INSTRUCTIONS,
                        EXTRACTION_INSTRUCTIONS, LOG_LEVEL, MODEL)
from radiology_dataset_db.extraction_utils import (
    DatasetWithPaperMetadata,
    ExtractionDeps,
    run_dataset_extraction_with_common_logic,
)

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
class RadiologyDataset(DatasetWithPaperMetadata):
    num_images: Optional[int] = None
    num_patients: Optional[int] = None
    modalities: List[Modality] = Field(default_factory=list)
    body_regions: List[BodyRegion] = Field(default_factory=list)
    additional_data: List[AdditionalData] = Field(default_factory=list)

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
    return await run_dataset_extraction_with_common_logic(
        title=title,
        abstract=abstract,
        publication_metadata=publication_metadata,
        dataset_agent=dataset_agent,
        agent_instructions=EXTRACTION_AGENT_INSTRUCTIONS,
        output_model=RadiologyDataset,
        logger=logger,
    )
