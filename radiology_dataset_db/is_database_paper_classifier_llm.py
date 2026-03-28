import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from radiology_dataset_db.config import (CLASSIFICATION_AGENT_INSTRUCTIONS,
                        CLASSIFICATION_INSTRUCTIONS, LOG_LEVEL, MODEL)

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
class DatasetClassification(BaseModel):
    is_dataset_creation: bool


# -----------------------------
# DEPENDENCIES
# -----------------------------
@dataclass
class ClassificationDeps:
    title: str
    abstract: str


# -----------------------------
# AGENT
# -----------------------------
classifier_agent = Agent(
    MODEL,
    deps_type=ClassificationDeps,
    output_type=DatasetClassification,
    instructions=CLASSIFICATION_INSTRUCTIONS,
)


@classifier_agent.instructions
async def add_text(ctx: RunContext[ClassificationDeps]) -> str:
    return f"""
Title: {ctx.deps.title}
Abstract: {ctx.deps.abstract}
"""


async def llm_thinks_not_dataset_paper(title: str, abstract: str) -> bool:
    """
    Returns:
        True  -> NOT a dataset paper (reject)
        False -> IS a dataset paper (keep)
    """
    try:
        deps = ClassificationDeps(
            title=title,
            abstract=abstract,
        )

        result = await classifier_agent.run(
            CLASSIFICATION_AGENT_INSTRUCTIONS, 
            deps=deps
        )

        output = result.output

        if isinstance(output, DatasetClassification):
            if output.is_dataset_creation:
                logger.debug("LLM classified as dataset paper.")
                return False  # keep
            else:
                logger.debug("LLM classified as NOT a dataset paper.")
                return True   # reject

        logger.debug("Unexpected output type, defaulting to keep.")
        return False

    except Exception as e:
        logger.debug(f"LLM dataset classification failed: {e}")
        return False  # fail open