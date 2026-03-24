# src/build_dataset_table.py

import os
import time
import asyncio
from enum import Enum
from typing import List, Optional, Union
from dataclasses import dataclass
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
MODEL = os.getenv("MODEL", "openai:Qwen/Qwen2.5-7B-Instruct")  # openai:gpt-5-3-instant
MAX_PAPERS = 1

QUERY = """
(radiology dataset OR medical imaging dataset) AND (CT OR MRI OR X-ray)
"""

INSTRUCTIONS = """You extract structured information about radiology datasets from papers. 
Only extract information explicitly stated or strongly implied. 
If a field is unknown, leave it null or empty. 
Dataset name rules:
- If a named dataset is referenced (e.g., 'RadImageNet database'), extract that name.
- Dataset names often appear in the title or as capitalized named resources.
- If a dataset shares the same name as a model (e.g., RadImageNet), still extract it.
"""


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


def serialize_dataset_output(dataset: Union[RadiologyDataset, str]) -> str:
    if isinstance(dataset, RadiologyDataset):
        return dataset.model_dump_json(indent=4)
    return str(dataset)


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

    if not articles:
        print("No articles found.")
        return
    
    if len(articles) < MAX_PAPERS:
        print(f"Only found {len(articles)} articles, less than the requested {MAX_PAPERS}.")

    with open(OUTPUT_PATH, "w") as f:
        for article in tqdm(articles):
            title, abstract, link = extract_text(article)
            # title, abstract, link = get_radimagenet_seed_text()

            if not abstract:
                continue

            dataset = await extract_with_agent(title, abstract, link)

            if dataset:
                f.write(serialize_dataset_output(dataset) + "\n")

            await asyncio.sleep(1)  # rate limit

    print(f"Extraction complete. Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
