# src/build_dataset_table.py

import os
import time
import asyncio
import re
import subprocess
import json
from enum import Enum
import logging
from typing import List, Optional, Union
from dataclasses import dataclass
from collections import Counter
from dotenv import load_dotenv

from tqdm import tqdm
from pydantic import BaseModel, Field
from Bio import Entrez
from pydantic_ai import Agent, RunContext

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

load_dotenv()

# -----------------------------
# CONFIG
# -----------------------------
OUTPUT_PATH = "data/datasets.json"
MODEL = "openai:Qwen/Qwen2.5-7B-Instruct"  # only if VLLM not set
MAX_PAPERS = 3  # 1-10_000 - for debugging

QUERY = """
(radiology dataset OR medical imaging dataset) AND (CT OR MRI OR X-ray)
"""

INSTRUCTIONS = (
    "You extract structured information about radiology datasets from papers.\n\n"

    "Step 1: Determine if this paper introduces or describes a dataset or imaging cohort.\n"
    "- If YES → is_dataset_paper = true\n"
    "- If NO → is_dataset_paper = false\n\n"

    "A dataset paper typically:\n"
    "- introduces or describes a named dataset, cohort, or biobank\n"
    "- describes dataset size (participants/patients/images)\n"
    "- uses phrases like 'we describe', 'we release', 'we present', 'here we describe'\n"
    "- uses phrases like 'data collection', 'data management', 'imaging study', 'imaging enhancement'\n"
    "- describes a population-based cohort or large-scale imaging initiative\n\n"

    "IMPORTANT:\n"
    "- Papers like 'MIMIC-CXR' ARE dataset papers even if phrased as 'here we describe'\n"
    "- Papers like 'UK Biobank' ARE dataset papers even if phrased as 'this article provides an overview'\n"
    "- Papers like 'RadImageNet' ARE dataset papers even if phrased as 'we describe'\n"
    "- If the paper's PRIMARY contribution is a named dataset, cohort, or imaging resource, it IS a dataset paper\n"
    "- Do NOT require the phrase 'we constructed'\n\n"

    "Step 2:\n"
    "- If is_dataset_paper = true → extract dataset info\n"
    "- If false → leave fields empty\n"
    "- The dataset name is often found directly in the paper title — use it if present\n"
)

### END CONFIG ###

DATASET_KEYWORDS = [
    "we constructed",
    "we collected",
    "we curated",
    "we assembled",
    "we present a dataset",
    "this dataset consists of",
    "we introduce",
    "we release",
    "here we describe",
    "we describe",
    "we make available",
    "publicly available dataset",
    "publicly available database",
    "de-identified publicly available",
    "data collection",
    "data management",
    "imaging study",
    "imaging enhancement",
    "cohort study",
    "population-based cohort",
]

USAGE_KEYWORDS = [
    "we used the",
    "we evaluated on",
    "trained on",
    "validated on",
]

vllm_port = os.getenv("VLLM_PORT")
if vllm_port is not None:
    vllm_port = int(vllm_port)

    result = subprocess.run(["curl", f"http://localhost:{vllm_port}/v1/models"], capture_output=True, text=True)
    result = json.loads(result.stdout)
    model_id = result["data"][0]["id"]
    MODEL = f"openai:{model_id}"

def is_dataset_paper_rule_based(text: str):
    text = text.lower()

    if any(k in text for k in DATASET_KEYWORDS):
        return True
    if any(k in text for k in USAGE_KEYWORDS):
        return False

    return None  # let LLM decide

def getenv(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Environment variable {key} is not set.")
    return value


Entrez.email = getenv("ENTREZ_EMAIL")

RADIMAGENET_TITLE = "RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning"
RADIMAGENET_ABSTRACT = "Purpose: To demonstrate the value of pretraining with millions of radiologic images compared with ImageNet photographic images on downstream medical applications when using transfer learning. Materials and methods: This retrospective study included patients who underwent a radiologic study between 2005 and 2020 at an outpatient imaging facility. Key images and associated labels from the studies were retrospectively extracted from the original study interpretation. These images were used for RadImageNet model training with random weight initiation. The RadImageNet models were compared with ImageNet models using the area under the receiver operating characteristic curve (AUC) for eight classification tasks and using Dice scores for two segmentation problems. Results: The RadImageNet database consists of 1.35 million annotated medical images in 131 872 patients who underwent CT, MRI, and US for musculoskeletal, neurologic, oncologic, gastrointestinal, endocrine, abdominal, and pulmonary pathologic conditions. For transfer learning tasks on small datasets-thyroid nodules (US), breast masses (US), anterior cruciate ligament injuries (MRI), and meniscal tears (MRI)-the RadImageNet models demonstrated a significant advantage (P < .001) to ImageNet models (9.4%, 4.0%, 4.8%, and 4.5% AUC improvements, respectively). For larger datasets-pneumonia (chest radiography), COVID-19 (CT), SARS-CoV-2 (CT), and intracranial hemorrhage (CT)-the RadImageNet models also illustrated improved AUC (P < .001) by 1.9%, 6.1%, 1.7%, and 0.9%, respectively. Additionally, lesion localizations of the RadImageNet models were improved by 64.6% and 16.4% on thyroid and breast US datasets, respectively. Conclusion: RadImageNet pretrained models demonstrated better interpretability compared with ImageNet models, especially for smaller radiologic datasets.Keywords: CT, MR Imaging, US, Head/Neck, Thorax, Brain/Brain Stem, Evidence-based Medicine, Computer Applications-General (Informatics) Supplemental material is available for this article. Published under a CC BY 4.0 license.See also the commentary by Cadrin-Chênevert in this issue."
RADIMAGENET_LINK = "https://doi.org/10.1148/ryai.210315"

def get_radimagenet_seed_text():
    return RADIMAGENET_TITLE, RADIMAGENET_ABSTRACT, RADIMAGENET_LINK

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

class RadiologyDataset(BaseModel):
    name: Optional[str] = Field(default=None)
    num_images: Optional[int] = None
    num_patients: Optional[int] = None
    modalities: List[Modality] = Field(default_factory=list)
    body_regions: List[BodyRegion] = Field(default_factory=list)
    additional_data: List[AdditionalData] = Field(default_factory=list)
    paper_title: Optional[str] = None
    paper_abstract: Optional[str] = None
    paper_link: Optional[str] = None


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
    output_type=RadiologyDataset,  #* change to str if running into bugs
    instructions=INSTRUCTIONS,
    # tools=[],
    # toolsets=[]
)


@dataset_agent.instructions
async def add_text(ctx: RunContext[ExtractionDeps]) -> str:
    return f"""
Text:
Title: {ctx.deps.title}
Abstract: {ctx.deps.abstract}

Extract:
- dataset name (often found directly in the title, e.g. "UK Biobank", "MIMIC-CXR", "RadImageNet" — use the name as it appears)
- number of patients or participants
- number of imaging studies or images
- imaging modalities: CT, MRI, X-ray, US, PET
- body regions: brain, chest, abdomen, pelvis, limbs
- additional data (reports, captions, segmentation, genomics, VQA, etc)
"""


# -----------------------------
# PUBMED SEARCH
# -----------------------------
def search_pubmed(query: str, max_results: int):
    logger.info("Searching PubMed...")
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    return record["IdList"]


def fetch_pubmed_details(id_list: List[str]):
    logger.info("Fetching article details from PubMed...")
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
async def extract_with_agent(title: str, abstract: str, link: Optional[str]):
    deps = ExtractionDeps(title=title, abstract=abstract, link=link)

    try:
        result = await dataset_agent.run(
            "Extract dataset information.", deps=deps
        )

        output = result.output
        logger.debug("LLM output: %s", output)
        if isinstance(output, RadiologyDataset):
            output.paper_title = title
            output.paper_abstract = abstract
            output.paper_link = link

            if not output.name:
                logger.debug("No dataset name extracted, rejecting output.")
                for msg in result.all_messages():
                    logger.debug(msg)
                return None  # reject if no dataset name extracted
            else:
                # FIX: use improved name_matches_title instead of raw token overlap
                if not name_matches_title(output.name, title):
                    logger.debug(
                        "Dataset name '%s' has no meaningful match with title, rejecting output.",
                        output.name
                    )
                    return None

        return output

    except Exception as e:
        logger.error("LLM extraction failed: %s", e)
        return None


# -----------------------------
# MAIN PIPELINE
# -----------------------------
async def main():
    if os.path.exists(OUTPUT_PATH):
        logger.warning("Will overwrite existing output file. If you want to keep it, please move or rename it before running the extraction.")
        # logger.warning(f"Output file {OUTPUT_PATH} already exists. Please remove it before running the extraction.")
        # return
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    ids = search_pubmed(QUERY, MAX_PAPERS)
    if not ids:
        logger.warning("No articles found.")
        return
    
    if len(ids) < MAX_PAPERS:
        logger.info(f"Only found {len(ids)} ids, less than the requested {MAX_PAPERS}.")

    articles = fetch_pubmed_details(ids)

    if not articles:
        logger.warning("No articles found.")
        return
    
    if len(articles) < MAX_PAPERS:
        logger.info(f"Only found {len(articles)} articles, less than the requested {MAX_PAPERS}.")

    extracted_datasets: List[RadiologyDataset] = []
    for article in tqdm(articles):
        title, abstract, link = extract_text(article)
        # title, abstract, link = get_radimagenet_seed_text()

        if not abstract:
            continue

        is_dataset_paper = is_dataset_paper_rule_based(title + " " + abstract)  #$ try to filter out non-dataset papers with simple rules before calling LLM
        if is_dataset_paper is False:
            continue  # skip papers that are likely not introducing a dataset

        dataset = await extract_with_agent(title, abstract, link)
        if isinstance(dataset, RadiologyDataset):
            extracted_datasets.append(dataset)

        await asyncio.sleep(1)  # rate limit

    with open(OUTPUT_PATH, "w") as f:
        for dataset in extracted_datasets:
            f.write(serialize_dataset_output(dataset) + "\n")

    logger.info(f"Extraction complete. Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())