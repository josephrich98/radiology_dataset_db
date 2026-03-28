import logging
from enum import Enum
from typing import List, Optional, Union

from dotenv import load_dotenv
from pydantic import Field
from pydantic_ai import Agent, RunContext

from src.config import (
    ADD_TEXT_EXTRACT_SCRNASEQ,
    EXTRACTION_AGENT_INSTRUCTIONS_SCRNASEQ,
    EXTRACTION_INSTRUCTIONS_SCRNASEQ,
    LOG_LEVEL,
    MODEL,
)
from src.extraction_utils import (
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


class SequencingTechnology(str, Enum):
    TENX = "10X"
    SMART_SEQ = "SMART-Seq/SMARTSEQ"
    PARSE = "Parse"
    DROP_SEQ = "Drop-seq"
    IN_DROPS = "inDrops"
    SEQ_WELL = "Seq-Well"
    OTHER = "other"

class CellOrNuclei(str, Enum):
    CELL = "cell"
    NUCLEI = "nuclei"
    BOTH = "both"


class ScRNASeqDataset(DatasetWithPaperMetadata):
    num_patients: Optional[int] = None
    sequencing_technology: List[SequencingTechnology] = Field(default_factory=list)
    disease: List[str] = Field(default_factory=list)
    species: List[str] = Field(default_factory=list)
    tissue: List[str] = Field(default_factory=list)
    cell_or_nuclei: Optional[CellOrNuclei] = None


dataset_agent = Agent(
    MODEL,
    deps_type=ExtractionDeps,
    output_type=ScRNASeqDataset,
    instructions=EXTRACTION_INSTRUCTIONS_SCRNASEQ,
)


@dataset_agent.instructions
async def add_text(ctx: RunContext[ExtractionDeps]) -> str:
    return f"""
Text:
Title: {ctx.deps.title}
Abstract: {ctx.deps.abstract}

{ADD_TEXT_EXTRACT_SCRNASEQ}
"""


def serialize_dataset_output(dataset: Union[ScRNASeqDataset, str]) -> str:
    if isinstance(dataset, ScRNASeqDataset):
        return dataset.model_dump_json(indent=4)
    return str(dataset)


async def extract_scrnaseq_dataset_info_with_agent(
    title: str,
    abstract: str,
    publication_metadata: Optional[dict] = None,
):
    return await run_dataset_extraction_with_common_logic(
        title=title,
        abstract=abstract,
        publication_metadata=publication_metadata,
        dataset_agent=dataset_agent,
        agent_instructions=EXTRACTION_AGENT_INSTRUCTIONS_SCRNASEQ,
        output_model=ScRNASeqDataset,
        logger=logger,
    )
