import json
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

LOG_LEVEL = logging.DEBUG  #* DEBUG, INFO, WARNING, ERROR, CRITICAL

@dataclass
class Config:
    output_path: str = "data/radiology_db.csv"
    output_path_failed: str = "data/radiology_db_failed.csv"
    max_papers: Optional[int] = 9999  # None for all papers; set to small number for debugging
    min_citations: int = 25  # filter out papers with fewer than this many citations (set to 0 to disable)
    num_tries_agent: int = 5
    overwrite: bool = False

def get_model() -> str:
    model = os.getenv("MODEL", "openai:Qwen/Qwen2.5-7B-Instruct")
    vllm_port = os.getenv("VLLM_PORT")
    if vllm_port is not None:
        vllm_port = int(vllm_port)

        result = subprocess.run(["curl", f"http://localhost:{vllm_port}/v1/models"], capture_output=True, text=True)
        if result.returncode == 0:
            result = json.loads(result.stdout)
            model_id = result["data"][0]["id"]
            model = f"openai:{model_id}"
        else:
            print("VLLM request failed - check if it is running and the port is correct.")
            print("stderr:", result.stderr)
    return model


CONFIG = Config()
MODEL = get_model()

# MeSH terms: https://www.ncbi.nlm.nih.gov/mesh/?term=%22radiology%22%5BMeSH%20Terms%5D%20OR%20%22radiographic%22%5BMeSH%20Terms%5D%20OR%20%22radiography%22%5BMeSH%20Terms%5D%20OR%20radiology%5BText%20Word%5D&cmd=DetailsSearch
PUBMED_QUERY = """
("Database Management Systems"[MeSH] OR dataset[ti] OR database[ti] OR "data collection"[ti] OR "information repository"[ti] OR benchmark[ti] OR "challenge data"[ti] OR "data commons"[ti] OR "data repository"[ti] OR "data sharing"[ti])
AND ("Radiology"[MeSH] OR "Radiography"[MeSH] OR "Radiology Information Systems"[MeSH] OR radiology[tiab] OR radiograph[tiab] OR "Diagnostic Imaging"[tiab] OR "Medical Image"[tiab] OR "Medical Imaging"[tiab] OR "Biomedical Image"[tiab] OR "Biomedical Imaging"[tiab] OR XR[tiab] OR CT[tiab] OR MRI[tiab] OR PET[tiab] OR SPECT[tiab] OR "X-ray"[tiab] OR "Computed Tomography"[tiab] OR "Magnetic Resonance"[tiab] OR Ultrasound[tiab] OR "Positron Emission Tomography"[tiab] OR "Single Photon Emission Computed Tomography"[tiab])
"""  # removed "Databases, Factual"[MeSH] because it dropped search space from 12319 to 3877 while keeping all of my test cases
PUBMED_QUERY = " ".join(PUBMED_QUERY.split())  # strip new lines

EXTRACTION_INSTRUCTIONS = (
    "You MUST extract a dataset name.\n"
    "Never return null for name.\n"
    "If uncertain, choose the most likely dataset or cohort name.\n"
    "Prefer names in the title.\n"
    "Examples:\n"
    "- UK Biobank\n"
    "- MIMIC-CXR\n"
    "- RadImageNet\n"
)

ADD_TEXT_EXTRACT = f"""
Extract:
- dataset name (best candidate; often found directly in the title, e.g. "UK Biobank", "MIMIC-CXR", "RadImageNet" — use the name as it appears)
- number of patients or participants
- number of imaging studies or images
- imaging modalities: CT, MRI, X-ray, US, PET
- body regions: brain, chest, abdomen, pelvis, limbs
- additional data (reports, captions, segmentation, genomics, VQA, etc)
"""

EXTRACTION_AGENT_INSTRUCTIONS = "Extract dataset information"

#* for real time, set to None
# IDS_TO_KEEP = None
IDS_TO_KEEP = {
    "36204533",  # RadImageNet
    "31831740",  # MIMIC-CXR
    "32457287",  # UK Biobank
    "23884657",  # TCIA
    "41781626",  # Merlin
}


CLASSIFICATION_INSTRUCTIONS = (
    "Determine whether the paper INTRODUCES or CREATES a dataset.\n"
    "Return is_dataset_creation = true if:\n"
    "- The paper develops, constructs, introduces, or presents a dataset/database/benchmark\n"
    "- Even if the dataset has no explicit name\n\n"
    "Return false if:\n"
    "- The paper only uses existing datasets\n"
    "- It is a methods/model paper\n"
    "- It analyzes data without creating a dataset\n\n"
    "Be conservative: if unsure, return true."
)

CLASSIFICATION_AGENT_INSTRUCTIONS = "Classify whether this paper creates a dataset"