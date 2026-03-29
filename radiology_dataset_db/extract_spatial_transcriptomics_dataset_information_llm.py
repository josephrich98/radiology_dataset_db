import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from radiology_dataset_db.config import (
    ADD_TEXT_EXTRACT_SPATIAL_TRANSCRIPTOMICS,
    EXTRACTION_AGENT_INSTRUCTIONS_SPATIAL_TRANSCRIPTOMICS,
    EXTRACTION_INSTRUCTIONS_SPATIAL_TRANSCRIPTOMICS,
    LOG_LEVEL,
    MODEL,
)
from radiology_dataset_db.is_database_paper_classifier_llm import llm_thinks_not_dataset_paper

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

load_dotenv()


class SequencingTechnology(str, Enum):
    VISIUM = "Visium"
    VISIUM_HD = "Visium HD"
    XENIUM = "Xenium"
    COSMX = "CosMx"
    MERFISH = "MERFISH"
    SEQFISH = "seqFISH"
    STEREO_SEQ = "Stereo-seq"
    SLIDE_SEQ = "Slide-seq"
    GEOMX_DSP = "GeoMx DSP"
    OTHER = "other"


class CellOrNuclei(str, Enum):
    CELL = "cell"
    NUCLEI = "nuclei"
    BOTH = "both"


class SpatialTranscriptomicsDataset(BaseModel):
    name: Optional[str] = Field(default=None)
    num_patients: Optional[int] = None
    sequencing_technology: List[SequencingTechnology] = Field(default_factory=list)
    disease: List[str] = Field(default_factory=list)
    species: List[str] = Field(default_factory=list)
    tissue: List[str] = Field(default_factory=list)
    cell_or_nuclei: Optional[CellOrNuclei] = None
    dataset_link: Optional[str] = None
    paper_title: Optional[str] = None
    paper_link: Optional[str] = None
    paper_year: Optional[int] = None
    paper_authors: List[str] = Field(default_factory=list)
    paper_journal: Optional[str] = None
    pmid: Optional[str] = None
    paper_citation_count: Optional[int] = None
    mesh_terms: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    pubmed_matches: Optional[List[List[str]]] = None


@dataclass
class ExtractionDeps:
    title: str
    abstract: str


dataset_agent = Agent(
    MODEL,
    deps_type=ExtractionDeps,
    output_type=SpatialTranscriptomicsDataset,
    instructions=EXTRACTION_INSTRUCTIONS_SPATIAL_TRANSCRIPTOMICS,
)


@dataset_agent.instructions
async def add_text(ctx: RunContext[ExtractionDeps]) -> str:
    return f"""
Text:
Title: {ctx.deps.title}
Abstract: {ctx.deps.abstract}

{ADD_TEXT_EXTRACT_SPATIAL_TRANSCRIPTOMICS}
"""


def serialize_dataset_output(dataset: Union[SpatialTranscriptomicsDataset, str]) -> str:
    if isinstance(dataset, SpatialTranscriptomicsDataset):
        return dataset.model_dump_json(indent=4)
    return str(dataset)


async def extract_spatial_transcriptomics_dataset_info_with_agent(
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
            logger.debug(
                "LLM thinks this is unlikely to be a dataset paper, rejecting without extraction."
            )
            return None

        result = await dataset_agent.run(
            EXTRACTION_AGENT_INSTRUCTIONS_SPATIAL_TRANSCRIPTOMICS, deps=deps
        )

        output = result.output
        logger.debug("LLM output: %s", output)
        if isinstance(output, SpatialTranscriptomicsDataset):
            output.paper_title = title

            publication_metadata = publication_metadata or {}
            output.paper_link = publication_metadata.get("link")
            output.paper_year = publication_metadata.get("year")
            output.paper_authors = publication_metadata.get("authors")
            output.paper_journal = publication_metadata.get("journal")
            output.pmid = publication_metadata.get("pmid")
            output.paper_citation_count = publication_metadata.get("citation_count")
            output.mesh_terms = publication_metadata.get("mesh_terms")
            output.keywords = publication_metadata.get("keywords")
            output.pubmed_matches = publication_metadata.get("pubmed_matches")

            if not output.name:
                logger.debug("No dataset name extracted, rejecting output.")
                for msg in result.all_messages():
                    logger.debug(msg)
                return None

        return output

    except Exception as e:
        logger.error("LLM extraction failed: %s", e)
        return None