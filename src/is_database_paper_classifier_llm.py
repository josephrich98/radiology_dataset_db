import os
import json
import subprocess
import logging
from dataclasses import dataclass
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  #* DEBUG, INFO

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

load_dotenv()

# Reuse same model string from env / config if needed
MODEL = "openai:Qwen/Qwen2.5-7B-Instruct"

vllm_port = os.getenv("VLLM_PORT")
if vllm_port is not None:
    vllm_port = int(vllm_port)

    result = subprocess.run(["curl", f"http://localhost:{vllm_port}/v1/models"], capture_output=True, text=True)
    result = json.loads(result.stdout)
    model_id = result["data"][0]["id"]
    MODEL = f"openai:{model_id}"


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
    dataset_name: str


# -----------------------------
# AGENT
# -----------------------------
classifier_agent = Agent(
    MODEL,
    deps_type=ClassificationDeps,
    output_type=DatasetClassification,
    instructions=(
        "Determine whether the paper INTRODUCES or CREATES a dataset.\n"
        "Return is_dataset_creation = true if:\n"
        "- The paper develops, constructs, introduces, or presents a dataset/database/benchmark\n"
        "- Even if the dataset has no explicit name\n\n"
        "Return false if:\n"
        "- The paper only uses existing datasets\n"
        "- It is a methods/model paper\n"
        "- It analyzes data without creating a dataset\n\n"
        "Be conservative: if unsure, return true."
    ),
)


@classifier_agent.instructions
async def add_text(ctx: RunContext[ClassificationDeps]) -> str:
    return f"""
Title: {ctx.deps.title}

Abstract: {ctx.deps.abstract}

Extracted dataset name: {ctx.deps.dataset_name}
"""


async def llm_thinks_not_dataset_paper(dataset_name: str, title: str, abstract: str) -> bool:
    """
    Returns:
        True  -> NOT a dataset paper (reject)
        False -> IS a dataset paper (keep)
    """
    try:
        deps = ClassificationDeps(
            title=title,
            abstract=abstract,
            dataset_name=dataset_name,
        )

        result = await classifier_agent.run(
            "Classify whether this paper creates a dataset", 
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