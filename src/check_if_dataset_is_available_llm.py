import logging
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from src.config import (DATASET_AVAILABILITY_AGENT_INSTRUCTIONS,
                        DATASET_AVAILABILITY_INSTRUCTIONS, LOG_LEVEL, MODEL)

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
class DatasetAvailabilityClassification(BaseModel):
    is_publicly_available: bool


# -----------------------------
# DEPENDENCIES
# -----------------------------
@dataclass
class DatasetAvailabilityDeps:
    title: str
    abstract: str
    full_text: Optional[str] = None


# -----------------------------
# AGENT
# -----------------------------
dataset_availability_agent = Agent(
    MODEL,
    deps_type=DatasetAvailabilityDeps,
    output_type=DatasetAvailabilityClassification,
    instructions=DATASET_AVAILABILITY_INSTRUCTIONS,
)


@dataset_availability_agent.instructions
async def add_text(ctx: RunContext[DatasetAvailabilityDeps]) -> str:
    full_text = ctx.deps.full_text if ctx.deps.full_text else "Not provided."
    return f"""
Title: {ctx.deps.title}
Abstract: {ctx.deps.abstract}
Full text: {full_text}
"""


async def check_if_dataset_is_available(
    title: str,
    abstract: str,
    full_text: Optional[str] = None,
) -> bool:
    """
    Returns:
        True  -> dataset appears publicly available
        False -> no clear public-availability evidence
    """
    if not title and not abstract and not full_text:
        logger.debug("No title, abstract, or full_text provided. Defaulting to unavailable.")
        return False

    try:
        deps = DatasetAvailabilityDeps(
            title=title or "",
            abstract=abstract or "",
            full_text=full_text,
        )

        result = await dataset_availability_agent.run(
            DATASET_AVAILABILITY_AGENT_INSTRUCTIONS,
            deps=deps,
        )

        output = result.output
        if isinstance(output, DatasetAvailabilityClassification):
            return output.is_publicly_available

        logger.debug("Unexpected output type for dataset availability, defaulting to unavailable.")
        return False

    except Exception as e:
        logger.debug(f"LLM dataset availability classification failed: {e}")
        return False
