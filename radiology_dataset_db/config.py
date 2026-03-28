import json
import logging
import os
import subprocess

from dotenv import load_dotenv

load_dotenv()

LOG_LEVEL = logging.DEBUG  # DEBUG, INFO, WARNING, ERROR, CRITICAL

#$ for real time, set to None
IDS_TO_KEEP = None
# IDS_TO_KEEP = {
#     "36204533",  # RadImageNet
#     "31831740",  # MIMIC-CXR
#     "32457287",  # UK Biobank
#     "23884657",  # TCIA
#     "41781626",  # Merlin
# }

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

MODEL = get_model()

#* PubMed
# MeSH terms: https://www.ncbi.nlm.nih.gov/mesh/?term=%22radiology%22%5BMeSH%20Terms%5D%20OR%20%22radiographic%22%5BMeSH%20Terms%5D%20OR%20%22radiography%22%5BMeSH%20Terms%5D%20OR%20radiology%5BText%20Word%5D&cmd=DetailsSearch
PUBMED_QUERY = """
("Database Management Systems"[MeSH] OR dataset[ti] OR database[ti] OR "data collection"[ti] OR "information repository"[ti] OR benchmark[ti] OR "challenge data"[ti] OR "data commons"[ti] OR "data repository"[ti] OR "data sharing"[ti])
AND ("Radiology"[MeSH] OR "Radiography"[MeSH] OR "Radiology Information Systems"[MeSH] OR radiology[tiab] OR radiograph[tiab] OR radiographs[tiab] OR "Diagnostic Imaging"[tiab] OR "Medical Image"[tiab] OR "Medical Imaging"[tiab] OR "Biomedical Image"[tiab] OR "Biomedical Imaging"[tiab] OR XR[tiab] OR CT[tiab] OR MRI[tiab] OR PET[tiab] OR SPECT[tiab] OR "X-ray"[tiab] OR "Computed Tomography"[tiab] OR "Magnetic Resonance"[tiab] OR Ultrasound[tiab] OR "Positron Emission Tomography"[tiab] OR "Single Photon Emission Computed Tomography"[tiab])
NOT (("Nuclear Magnetic Resonance"[tiab] OR NMR[tiab]) OR ("X-ray crystallography"[tiab] OR crystallograph*[tiab] OR diffraction[tiab]) OR (mice[tiab] or mouse[tiab]))
"""  # removed "Databases, Factual"[MeSH] because it dropped search space from 12319 to 3877 while keeping all of my test cases
PUBMED_QUERY = " ".join(PUBMED_QUERY.split())  # strip new lines

#* is_database_paper_classifier_llm.py
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

#* check_if_dataset_is_available_llm.py
DATASET_AVAILABILITY_INSTRUCTIONS = (
    "Determine whether the paper indicates that its dataset is publicly available.\n"
    "Return is_publicly_available = true if there is direct evidence in the provided text such as:\n"
    "- explicit language like 'open', 'publicly available', 'public', 'available', 'release', or a data availability statement\n"
    "- a non-DOI URL (e.g., GitHub, Zenodo, institutional repository, challenge site, or dataset website)\n"
    "- wording that readers can access or download the dataset\n\n"
    "Return false if:\n"
    "- the text only says data are available on request\n"
    "- access is restricted, private, proprietary, or behind an application process\n"
    "- only a DOI to the paper is provided without dataset access evidence\n"
    "- there is no clear evidence of public dataset availability\n\n"
    "Use title, abstract, and full text (if provided). Be generous, especially if full text is not provided: if unsure, return true."
)

DATASET_AVAILABILITY_AGENT_INSTRUCTIONS = (
    "Classify whether the dataset described in this paper is publicly available."
)

#* extract_radiology_dataset_information_llm.py
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
- the http link to the dataset, if available at the end of the abstract
"""

EXTRACTION_AGENT_INSTRUCTIONS = "Extract dataset information"

#* extract_scrnaseq_dataset_information_llm.py
EXTRACTION_INSTRUCTIONS_SCRNASEQ = (
    "You MUST extract a dataset name.\n"
    "Never return null for name.\n"
    "If uncertain, choose the most likely dataset or cohort name.\n"
    "Prefer names in the title.\n"
    "This extractor is for scRNA-seq/snRNA-seq dataset papers.\n"
)

ADD_TEXT_EXTRACT_SCRNASEQ = f"""
Extract:
- dataset name (best candidate; often found directly in the title)
- number of patients or participants
- sequencing technology/platform (e.g., 10X, SMART-Seq/SMARTSEQ, Parse, Drop-seq, inDrops, Seq-Well)
- disease(s)
- species
- tissue(s)
- whether the assay is cell, nuclei, or both (for scRNA-seq/snRNA-seq)
- the http link to the dataset, if available at the end of the abstract
"""

EXTRACTION_AGENT_INSTRUCTIONS_SCRNASEQ = "Extract scRNA-seq/snRNA-seq dataset information"

#* extract_bulk_genomics_dataset_information_llm.py
EXTRACTION_INSTRUCTIONS_BULK_GENOMICS = (
    "You MUST extract a dataset name.\n"
    "Never return null for name.\n"
    "If uncertain, choose the most likely dataset or cohort name.\n"
    "Prefer names in the title.\n"
    "This extractor is for bulk genomics dataset papers.\n"
)

ADD_TEXT_EXTRACT_BULK_GENOMICS = f"""
Extract:
- dataset name (best candidate; often found directly in the title)
- number of patients or participants
- sequencing modality/modality labels from: WGS, WXS, bulk RNA
- whether raw reads are available (true/false if evidence is present)
- disease(s)
- species
- tissue(s)
- the http link to the dataset, if available at the end of the abstract
"""

EXTRACTION_AGENT_INSTRUCTIONS_BULK_GENOMICS = "Extract bulk genomics dataset information"

#* extract_spatial_transcriptomics_dataset_information_llm.py
EXTRACTION_INSTRUCTIONS_SPATIAL_TRANSCRIPTOMICS = (
    "You MUST extract a dataset name.\n"
    "Never return null for name.\n"
    "If uncertain, choose the most likely dataset or cohort name.\n"
    "Prefer names in the title.\n"
    "This extractor is for spatial transcriptomics dataset papers.\n"
)

ADD_TEXT_EXTRACT_SPATIAL_TRANSCRIPTOMICS = f"""
Extract:
- dataset name (best candidate; often found directly in the title)
- number of patients or participants
- spatial transcriptomics technology/platform (e.g., Visium, Visium HD, Xenium, CosMx, MERFISH, seqFISH, Stereo-seq, Slide-seq, GeoMx DSP)
- disease(s)
- species
- tissue(s)
- whether the assay is cell, nuclei, or both
- the http link to the dataset, if available at the end of the abstract
"""

EXTRACTION_AGENT_INSTRUCTIONS_SPATIAL_TRANSCRIPTOMICS = "Extract spatial transcriptomics dataset information"

#* add additional instructions and config variables for other modalities here, e.g. genomics, pathology, etc
