import logging
from enum import Enum
from typing import List, Optional, Union

from dotenv import load_dotenv
from pydantic import Field
from pydantic_ai import Agent, RunContext

from radiology_dataset_db.config import (
    ADD_TEXT_EXTRACT_BULK_GENOMICS,
    EXTRACTION_AGENT_INSTRUCTIONS_BULK_GENOMICS,
    EXTRACTION_INSTRUCTIONS_BULK_GENOMICS,
    LOG_LEVEL,
    MODEL,
)
from radiology_dataset_db.extraction_utils import (
    DatasetWithPaperMetadata,
    ExtractionDeps,
    run_dataset_extraction_with_common_logic,
)

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

load_dotenv()


class BulkGenomicsModality(str, Enum):
    WGS = "WGS"
    WXS = "WXS"
    BULK_RNA = "bulk RNA"


class BulkGenomicsDataset(DatasetWithPaperMetadata):
    num_patients: Optional[int] = None
    modalities: List[BulkGenomicsModality] = Field(default_factory=list)
    raw_reads_available: Optional[bool] = None
    disease: List[str] = Field(default_factory=list)
    species: List[str] = Field(default_factory=list)
    tissue: List[str] = Field(default_factory=list)


dataset_agent = Agent(
    MODEL,
    deps_type=ExtractionDeps,
    output_type=BulkGenomicsDataset,
    instructions=EXTRACTION_INSTRUCTIONS_BULK_GENOMICS,
)


@dataset_agent.instructions
async def add_text(ctx: RunContext[ExtractionDeps]) -> str:
    return f"""
Text:
Title: {ctx.deps.title}
Abstract: {ctx.deps.abstract}

{ADD_TEXT_EXTRACT_BULK_GENOMICS}
"""


def serialize_dataset_output(dataset: Union[BulkGenomicsDataset, str]) -> str:
    if isinstance(dataset, BulkGenomicsDataset):
        return dataset.model_dump_json(indent=4)
    return str(dataset)


async def extract_bulk_genomics_dataset_info_with_agent(
    title: str,
    abstract: str,
    publication_metadata: Optional[dict] = None,
):
    return await run_dataset_extraction_with_common_logic(
        title=title,
        abstract=abstract,
        publication_metadata=publication_metadata,
        dataset_agent=dataset_agent,
        agent_instructions=EXTRACTION_AGENT_INSTRUCTIONS_BULK_GENOMICS,
        output_model=BulkGenomicsDataset,
        logger=logger,
    )
