# src/build_dataset_table.py

import os
import time
from tqdm import tqdm
import asyncio
from typing import List, Optional
from dataclasses import dataclass

from tqdm import tqdm
from pydantic import BaseModel, Field
from Bio import Entrez
from pydantic_ai import Agent, RunContext


# -----------------------------
# CONFIG
# -----------------------------
OUTPUT_PATH = "data/datasets.json"
MODEL = os.getenv("MODEL", "ollama:qwen2.5")  # openai:gpt-5-3-instant
MAX_PAPERS = 3

QUERY = """
(radiology dataset OR medical imaging dataset) AND (CT OR MRI OR X-ray)
"""


def getenv(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Environment variable {key} is not set.")
    return value


Entrez.email = getenv("ENTREZ_EMAIL")

# -----------------------------
# SCHEMA
# -----------------------------
class RadiologyDataset(BaseModel):
    name: Optional[str] = Field(default=None)
    num_patients: Optional[int] = None
    num_series: Optional[int] = None
    num_images: Optional[int] = None
    modalities: List[str] = Field(default_factory=list)
    body_regions: List[str] = Field(default_factory=list)
    additional_data: List[str] = Field(default_factory=list)
    link: Optional[str] = None


# -----------------------------
# DEPENDENCIES
# -----------------------------
@dataclass
class ExtractionDeps:
    title: str
    abstract: str
    link: Optional[str]


# -----------------------------
# AGENT
# -----------------------------
dataset_agent = Agent(
    MODEL,
    deps_type=ExtractionDeps,
    output_type=RadiologyDataset,
    instructions=(
        "You extract structured information about radiology datasets from papers. "
        "Only extract information explicitly stated or strongly implied. "
        "If a field is unknown, leave it null or empty."
    ),
)


@dataset_agent.instructions
async def add_text(ctx: RunContext[ExtractionDeps]) -> str:
    return f"""
Text:
Title: {ctx.deps.title}
Abstract: {ctx.deps.abstract}

Extract:
- dataset name
- number of patients
- number of imaging series
- number of images
- imaging modalities (CT, MRI, X-ray, etc)
- body regions (brain, chest, abdomen, etc)
- additional data (reports, captions, segmentation, genomics, VQA, etc)
"""


# -----------------------------
# PUBMED SEARCH
# -----------------------------
def search_pubmed(query: str, max_results: int):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    return record["IdList"]


def fetch_pubmed_details(id_list: List[str]):
    handle = Entrez.efetch(db="pubmed", id=",".join(id_list), retmode="xml")
    records = Entrez.read(handle)
    return records["PubmedArticle"]


def extract_text(article):
    try:
        title = article["MedlineCitation"]["Article"]["ArticleTitle"]
    except:
        title = ""

    try:
        abstract_list = article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
        abstract = " ".join(abstract_list)
    except:
        abstract = ""

    try:
        article_id_list = article["PubmedData"]["ArticleIdList"]
        doi = None
        for a in article_id_list:
            if a.attributes.get("IdType") == "doi":
                doi = str(a)
        link = f"https://doi.org/{doi}" if doi else None
    except:
        link = None

    return title, abstract, link


# -----------------------------
# LLM EXTRACTION (ASYNC)
# -----------------------------
async def extract_with_agent(title: str, abstract: str, link: Optional[str]):
    deps = ExtractionDeps(title=title, abstract=abstract, link=link)

    try:
        result = await dataset_agent.run(
            "Extract dataset information.", deps=deps
        )

        output = result.output
        output.link = link
        return output

    except Exception as e:
        print(f"LLM extraction failed: {e}")
        return None


# -----------------------------
# MAIN PIPELINE
# -----------------------------
async def main():
    os.makedirs("data", exist_ok=True)

    ids = search_pubmed(QUERY, MAX_PAPERS)
    articles = fetch_pubmed_details(ids)

    with open(OUTPUT_PATH, "w") as f:
        for article in tqdm(articles):
            title, abstract, link = extract_text(article)

            if not abstract:
                continue

            dataset = await extract_with_agent(title, abstract, link)

            if dataset:
                f.write(dataset.model_dump_json() + "\n")

            await asyncio.sleep(1)  # rate limit


if __name__ == "__main__":
    asyncio.run(main())