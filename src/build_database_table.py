# src/build_dataset_table.py

import os
import time
import asyncio
import re
from enum import Enum
from typing import List, Optional, Union
from dataclasses import dataclass
from collections import Counter
from dotenv import load_dotenv

from tqdm import tqdm
from pydantic import BaseModel, Field
from Bio import Entrez
from pydantic_ai import Agent, RunContext

load_dotenv()

# -----------------------------
# CONFIG
# -----------------------------
OUTPUT_PATH = "data/datasets.json"
MODEL = "openai:Qwen/Qwen2.5-7B-Instruct"  # openai:gpt-5-3-instant
MAX_PAPERS = 9999  # 1-10_000 - for debugging
MIN_APPEARANCES = 3  # minimum number of times a dataset must be mentioned across papers to be included in the final output. Set to >1 to filter out one-off mentions and focus on more widely used datasets. Set to 1 or less to include all extracted datasets regardless of frequency.

QUERY = """
(radiology dataset OR medical imaging dataset) AND (CT OR MRI OR X-ray)
"""

INSTRUCTIONS=(
    "You extract structured information about radiology datasets from papers.\n\n"

    # "First determine:\n"  #$ try to filter out non-dataset papers by asking the LLM to classify whether the paper is likely introducing a new dataset or just using an existing dataset. This can help reduce false positives where the LLM hallucinates a dataset name that isn't actually the focus of the paper.
    # "- Is this paper introducing a NEW dataset? (is_dataset_paper = true)\n"
    # "- Or is it USING an existing dataset? (is_dataset_paper = false)\n\n"

    # "A dataset paper typically:\n"
    # "- introduces a named dataset\n"
    # "- describes how the data was collected\n"
    # "- reports dataset size (patients/images)\n"
    # "- uses phrases like 'we constructed', 'we collected', 'we present'\n\n"

    # "A non-dataset paper typically:\n"
    # "- uses datasets like MIMIC, CheXpert, TCGA\n"
    # "- focuses on model performance or methods\n\n"

    # "Only extract dataset fields if is_dataset_paper = true.\n"
    # "Otherwise return empty/null fields.\n"

    "Dataset name rules:\n"
    "- If a named dataset is referenced (e.g., 'RadImageNet database'), extract that name.\n"
    "- Dataset names often appear in the title or as capitalized named resources.\n"
    "- If a dataset shares the same name as a model (e.g., RadImageNet), still extract it.\n\n"
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
]

USAGE_KEYWORDS = [
    "we used the",
    "we evaluated on",
    "trained on",
    "validated on",
]

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
    num_appearances: Optional[int] = None
    num_series: Optional[int] = None
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
- dataset name
- number of patients
- number of imaging series
- imaging modalities: CT, MRI, X-ray, US, PET
- body regions: brain, chest, abdomen, pelvis, limbs
- additional data (reports, captions, segmentation, genomics, VQA, etc)
"""


# -----------------------------
# PUBMED SEARCH
# -----------------------------
def search_pubmed(query: str, max_results: int):
    print("Searching PubMed...")
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    return record["IdList"]


def fetch_pubmed_details(id_list: List[str]):
    print("Fetching article details from PubMed...")
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


def annotate_appearance_counts(datasets: List[RadiologyDataset]) -> List[RadiologyDataset]:
    normalized_names = [normalize_dataset_name(dataset.name) for dataset in datasets]
    counts = Counter(name for name in normalized_names if name is not None)

    for dataset, normalized_name in zip(datasets, normalized_names):
        dataset.num_appearances = counts.get(normalized_name) if normalized_name is not None else None

    return datasets


def filter_by_min_appearances(
    datasets: List[RadiologyDataset], min_appearances: int
) -> List[RadiologyDataset]:
    if min_appearances <= 1:
        return datasets

    return [
        dataset
        for dataset in datasets
        if (dataset.num_appearances or 0) >= min_appearances
    ]


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
        if isinstance(output, RadiologyDataset):
            output.paper_title = title
            output.paper_abstract = abstract
            output.paper_link = link

            if output.name and output.name.lower() not in output.paper_title.lower():  #$ try to filter out non-dataset papers by checking if the extracted dataset name appears in the paper title. Not perfect but can help catch some false positives where the LLM hallucinates a dataset name that isn't actually the focus of the paper.
                return None

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
    if not ids:
        print("No articles found.")
        return
    
    if len(ids) < MAX_PAPERS:
        print(f"Only found {len(ids)} ids, less than the requested {MAX_PAPERS}.")

    articles = fetch_pubmed_details(ids)

    if not articles:
        print("No articles found.")
        return
    
    if len(articles) < MAX_PAPERS:
        print(f"Only found {len(articles)} articles, less than the requested {MAX_PAPERS}.")

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

    extracted_datasets = annotate_appearance_counts(extracted_datasets)
    extracted_datasets = filter_by_min_appearances(extracted_datasets, MIN_APPEARANCES)

    with open(OUTPUT_PATH, "w") as f:
        for dataset in extracted_datasets:
            f.write(serialize_dataset_output(dataset) + "\n")

    print(f"Extraction complete. Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
