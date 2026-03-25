# src/build_dataset_table.py

import argparse
import os
import time
import asyncio
import requests
import re
import subprocess
import json
import xml.etree.ElementTree as ET
from enum import Enum
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from collections import Counter
from dotenv import load_dotenv

import pandas as pd
from more_itertools import chunked
from tqdm import tqdm
from pydantic import BaseModel, Field
from Bio import Entrez
from pydantic_ai import Agent, RunContext

from src.config import CONFIG, MODEL, IDS_TO_KEEP, PUBMED_QUERY, EXTRACTION_INSTRUCTIONS, EXTRACTION_AGENT_INSTRUCTIONS, ADD_TEXT_EXTRACT
from src.is_database_paper_classifier_llm import llm_thinks_not_dataset_paper

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

def parse_args():
    parser = argparse.ArgumentParser(description="Build radiology dataset table")

    parser.add_argument("--output-path", type=str)
    parser.add_argument("--output-path-failed", type=str)
    parser.add_argument("--max-papers", type=int)
    parser.add_argument("--min-citations", type=int)
    parser.add_argument("--num-tries-agent", type=int)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()

def apply_overrides(args):
    for key, value in vars(args).items():
        if value is not None and hasattr(CONFIG, key) and getattr(CONFIG, key) != value:
            logger.info(f"Overriding config: {key} = {value}")
            setattr(CONFIG, key, value)

def getenv(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Environment variable {key} is not set.")
    return value

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
    # paper_abstract: Optional[str] = None
    paper_link: Optional[str] = None
    paper_year: Optional[int] = None
    paper_authors: List[str] = Field(default_factory=list)
    paper_journal: Optional[str] = None
    pmid: Optional[str] = None
    paper_citation_count: Optional[int] = None


# -----------------------------
# DEPENDENCIES
# -----------------------------
@dataclass
class ExtractionDeps:
    title: str
    abstract: str


# -----------------------------
# AGENT
# -----------------------------
dataset_agent = Agent(
    MODEL,
    deps_type=ExtractionDeps,
    output_type=RadiologyDataset,  #* change to str if running into bugs
    instructions=EXTRACTION_INSTRUCTIONS
)



@dataset_agent.instructions
async def add_text(ctx: RunContext[ExtractionDeps]) -> str:
    return f"""
Text:
Title: {ctx.deps.title}
Abstract: {ctx.deps.abstract}

{ADD_TEXT_EXTRACT}
"""

# -----------------------------
# PUBMED SEARCH
# -----------------------------
def search_pubmed(pubmed_query: str, max_results: Optional[int] = None, batch_size: int = 10_000, sleep: float = 0.33) -> List[str]:
    """
    Fetch PubMed IDs using Entrez with support for large queries.

    Args:
        pubmed_query: PubMed search string
        max_results: Max number of IDs to return (None = all)
        batch_size: Number of IDs per request (<= 10,000)
        sleep: Delay between requests (NCBI rate limit)

    Returns:
        List of PubMed IDs (strings)
    """    
    logger.info("Searching PubMed...")
    logger.debug(f"PubMed query: {pubmed_query}")
    pubmed_query = " ".join(pubmed_query.split())  # strip new lines

    if not Entrez.email:
        if not os.getenv("ENTREZ_EMAIL"):
            raise ValueError("Entrez email is not set. Please set the ENTREZ_EMAIL environment variable.")
        else:
            Entrez.email = os.getenv("ENTREZ_EMAIL")
    if not Entrez.api_key and os.getenv("ENTREZ_API_KEY"):
        Entrez.api_key = os.getenv("ENTREZ_API_KEY")
    
    # Initial query with history
    handle = Entrez.esearch(
        db="pubmed",
        term=pubmed_query,
        usehistory="y",
        retmax=0
    )
    record = Entrez.read(handle)

    count = int(record["Count"])
    webenv = record["WebEnv"]
    query_key = record["QueryKey"]

    logger.info(f"PubMed search found {count} articles.")

    # Decide how many to fetch
    if max_results is None:
        total_to_fetch = count
    else:
        total_to_fetch = min(max_results, count)
    
    if total_to_fetch > 10_000:
        raise ValueError(f"Requested {total_to_fetch} results, which exceeds the maximum of 10,000 allowed by NCBI. Please set max_results to 10,000 or fewer.")

    ids = []

    for start in range(0, total_to_fetch, batch_size):
        current_batch_size = min(batch_size, total_to_fetch - start)

        handle = Entrez.esearch(
            db="pubmed",
            term=pubmed_query,
            retstart=start,
            retmax=current_batch_size,
            webenv=webenv,
            query_key=query_key,
        )
        record = Entrez.read(handle)
        ids.extend(record["IdList"])

        time.sleep(sleep)  # respect rate limits

    return ids


def fetch_pubmed_details(id_list, batch_size=200):
    results = []
    for batch in tqdm(list(chunked(id_list, batch_size)), desc="Fetching PubMed details"):
        handle = Entrez.efetch(
            db="pubmed",
            id=",".join(batch),
            retmode="xml"
        )
        records = Entrez.read(handle)
        results.extend(records["PubmedArticle"])
    return results

def fetch_pubmed_citation_counts(pmids, batch_size=100):
    url = "https://icite.od.nih.gov/api/pubs"
    results = {}

    for batch in tqdm(list(chunked(pmids, batch_size)), desc="Fetching citation counts"):
        try:
            response = requests.get(url, params={"pmids": ",".join(batch)})
            
            if response.status_code != 200:
                raise Exception(f"Status {response.status_code}")

            data = response.json()

            # fill results for successful ones
            for p in data.get("data", []):
                results[str(p["pmid"])] = p.get("citation_count", 0)

            # handle missing PMIDs in response
            returned_pmids = {str(p["pmid"]) for p in data.get("data", [])}
            for pmid in batch:
                if pmid not in returned_pmids:
                    results[str(pmid)] = 0  # or None

        except Exception:
            # fallback: assign 0/None for entire failed batch
            for pmid in batch:
                results[str(pmid)] = 0  # or None

    return results

def _extract_title(article) -> str:
    try:
        return article["MedlineCitation"]["Article"]["ArticleTitle"]
    except:
        return ""

def _extract_abstract(article) -> str:
    try:
        abstract_list = article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
        return " ".join(abstract_list)
    except:
        return ""

def _extract_link(article) -> Optional[str]:
    try:
        article_id_list = article["PubmedData"]["ArticleIdList"]
        doi = None
        for a in article_id_list:
            if a.attributes.get("IdType") == "doi":
                doi = str(a)
        return f"https://doi.org/{doi}" if doi else None
    except:
        return None

def _extract_year(article) -> Optional[int]:
    try:
        pub_date = article["MedlineCitation"]["Article"]["Journal"]["JournalIssue"]["PubDate"]
    except Exception:
        pub_date = {}

    year_candidates = []
    if isinstance(pub_date, dict):
        year_candidates.append(pub_date.get("Year"))
        year_candidates.append(pub_date.get("MedlineDate"))

    try:
        article_dates = article["MedlineCitation"]["Article"].get("ArticleDate", [])
        if article_dates:
            year_candidates.append(article_dates[0].get("Year"))
    except Exception:
        pass

    for candidate in year_candidates:
        if candidate is None:
            continue
        text = str(candidate)
        match = re.search(r"\b(19|20)\d{2}\b", text)
        if match:
            return int(match.group(0))

    return None


def _extract_authors(article) -> List[str]:
    try:
        author_list = article["MedlineCitation"]["Article"].get("AuthorList", [])
    except Exception:
        return []

    authors = []
    for author in author_list:
        collective = author.get("CollectiveName")
        if collective:
            authors.append(str(collective))
            continue

        last_name = author.get("LastName")
        initials = author.get("Initials")
        fore_name = author.get("ForeName")

        if last_name and initials:
            authors.append(f"{last_name} {initials}")
        elif fore_name and last_name:
            authors.append(f"{fore_name} {last_name}")
        elif last_name:
            authors.append(str(last_name))

    return authors


def _extract_journal(article) -> Optional[str]:
    try:
        journal = article["MedlineCitation"]["Article"]["Journal"].get("Title")
        return str(journal) if journal else None
    except Exception:
        return None

def _extract_pmid(article) -> Optional[str]:
    try:
        pmid = article["MedlineCitation"]["PMID"]
        if pmid:
            return str(pmid)
    except Exception:
        pass

    try:
        article_id_list = article["PubmedData"]["ArticleIdList"]
        for article_id in article_id_list:
            if article_id.attributes.get("IdType") == "pubmed":
                return str(article_id)
    except Exception:
        pass

    return None


def extract_pubmed_metadata(article, citation_counts_by_pmid: Optional[Dict[str, Optional[int]]] = None):
    citation_counts_by_pmid = citation_counts_by_pmid or {}

    pmid = _extract_pmid(article)
    return {
        "title": _extract_title(article),
        "abstract": _extract_abstract(article),
        "link": _extract_link(article),
        "year": _extract_year(article),
        "authors": _extract_authors(article),
        "journal": _extract_journal(article),
        "pmid": pmid,
        "citation_count": citation_counts_by_pmid.get(pmid) if pmid else None,
    }


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
async def extract_with_agent(
    title: str,
    abstract: str,
    publication_metadata: Optional[dict] = None,
):
    if not title or not abstract:
        logger.debug("Missing title or abstract, skipping extraction.")
        return None

    deps = ExtractionDeps(title=title, abstract=abstract)

    try:
        unlikely_dataset_paper = await llm_thinks_not_dataset_paper(title, abstract)
        if unlikely_dataset_paper:
            #* 1. Classification LLM thinks this is unlikely to be a dataset paper → reject immediately without extraction (saves cost and prevents hallucination of dataset info when paper is not about a dataset)
            logger.debug("LLM thinks this is unlikely to be a dataset paper, rejecting without extraction.")
            return None

        #* Now run the extraction agent.
        result = await dataset_agent.run(
            EXTRACTION_AGENT_INSTRUCTIONS, deps=deps
        )

        output = result.output
        logger.debug("LLM output: %s", output)
        if isinstance(output, RadiologyDataset):
            output.paper_title = title
            # output.paper_abstract = abstract
            
            publication_metadata = publication_metadata or {}
            output.paper_link = publication_metadata.get("link")
            output.paper_year = publication_metadata.get("year")
            output.paper_authors = publication_metadata.get("authors")
            output.paper_journal = publication_metadata.get("journal")
            output.pmid = publication_metadata.get("pmid")
            output.paper_citation_count = publication_metadata.get("citation_count")

            if not output.name:
                #* 2. No dataset name extracted → reject (this is a strong signal that the paper is not actually about a dataset, since the name is usually the easiest thing to extract and often appears in the title)
                logger.debug("No dataset name extracted, rejecting output.")
                for msg in result.all_messages():
                    logger.debug(msg)
                return None  # reject if no dataset name extracted
            else:
                if not name_matches_title(output.name, title):  # reject if dataset name doesn't have meaningful match with title
                    #* 3. Dataset name extracted but doesn't match title → reject (this is a signal that the extracted name may be spurious and not actually the name of a dataset created by the paper, especially if the LLM also thinks it's not likely to be a dataset paper)
                    logger.debug(
                        "Dataset name '%s' has no meaningful match with title and LLM does not think it's a dataset paper, rejecting output.",
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
    args = parse_args()
    apply_overrides(args)

    if os.path.exists(CONFIG.output_path):
        if CONFIG.overwrite:
            confirm = input(f"Output file {CONFIG.output_path} already exists. Do you want to overwrite it? (y/n): ")
            if confirm.lower() != "y":
                logger.info("Exiting without overwriting. Please set overwrite=False to append to the existing file, or move/rename it before running the extraction.")
                return
            os.remove(CONFIG.output_path)
        else:
            logger.info(f"Output file {CONFIG.output_path} already exists. Set overwrite=True to overwrite it, or move/rename it before running the extraction. This will append to it but not replace existing lines, so you may get duplicates if you run multiple times without changing the output path.")
    else:
        logger.info(f"No existing output file found at {CONFIG.output_path}. A new file will be created.")

    if os.path.dirname(CONFIG.output_path):
        os.makedirs(os.path.dirname(CONFIG.output_path), exist_ok=True)

    logger.info(f"Using model: {MODEL}")

    Entrez.email = getenv("ENTREZ_EMAIL")
    if os.getenv("ENTREZ_API_KEY"):
        Entrez.api_key = getenv("ENTREZ_API_KEY")
        logger.info("Using Entrez API key")

    df = None
    titles_existing, pmids_existing = set(), set()
    if os.path.exists(CONFIG.output_path):
        df = pd.read_csv(CONFIG.output_path)
        titles_existing = set(df["paper_title"].dropna().tolist())
        pmids_existing = set(df["pmid"].dropna().astype(str))
        logger.info(f"Found {len(pmids_existing)} existing PMIDs in output file. These will be skipped in the new extraction.")

    ids = search_pubmed(PUBMED_QUERY, CONFIG.max_papers)
    if not ids:
        logger.warning("No articles found.")
        return
    
    logger.info(f"Found {len(ids)} articles from PubMed search.")
    
    if IDS_TO_KEEP:
        ids = [id for id in ids if id in IDS_TO_KEEP]
        logger.info(f"Filtered to {len(ids)} articles based on IDS_TO_KEEP list.")
    
    if pmids_existing:
        ids = [id for id in ids if id not in pmids_existing]
        logger.info(f"After filtering out existing PMIDs, {len(ids)} new articles remain for extraction.")
    
    logger.info(f"{len(ids)} articles will be processed for dataset extraction.")

    articles = fetch_pubmed_details(ids)
    citation_counts_by_pmid = fetch_pubmed_citation_counts(ids)

    if CONFIG.min_citations > 0:
        ids = [id for id in ids if citation_counts_by_pmid.get(id, 0) >= CONFIG.min_citations]
        logger.info(f"After filtering out articles with fewer than {CONFIG.min_citations} citations, {len(ids)} articles remain.")

    if not articles:
        logger.warning("No articles found.")
        return

    extracted_datasets: List[RadiologyDataset] = []
    failed_metadata = []
    for article in tqdm(articles):
        publication_metadata = extract_pubmed_metadata(article, citation_counts_by_pmid)
        title = publication_metadata.get("title")
        abstract = publication_metadata.get("abstract")

        if title and title in titles_existing:
            logger.debug(f"Article title already exists in output, skipping: {title}")
            continue

        dataset = None
        for _ in range(CONFIG.num_tries_agent):
            dataset = await extract_with_agent(title, abstract, publication_metadata)
            if dataset is not None:
                break
        
        if isinstance(dataset, RadiologyDataset):
            extracted_datasets.append(dataset)
        else:
            logger.debug(f"Extraction failed for article: {title}")
            failed_metadata.append(publication_metadata)

        await asyncio.sleep(1)  # rate limit

    # Convert extracted datasets → dict rows
    rows = [d.model_dump() for d in extracted_datasets]
    new_df = pd.DataFrame(rows)

    logger.info(f"Extracted {len(new_df)} datasets from {len(articles)} articles.")
    logger.info(f"Failed to extract datasets from {len(failed_metadata)} articles.")
    
    # convert list fields to comma-separated strings for CSV output
    for col in new_df.columns:
        if not new_df[col].empty and isinstance(new_df[col].iloc[0], list):
            new_df[col] = new_df[col].apply(lambda x: ",".join(x) if isinstance(x, list) else x)

    if df is None or df.empty:
        df = new_df
    else:
        df = pd.concat([df, new_df], ignore_index=True)

    # Save to CSV
    df.to_csv(CONFIG.output_path, index=False)

    if CONFIG.output_path_failed:
        df_failed = pd.DataFrame(failed_metadata)
        df_failed.to_csv(CONFIG.output_path_failed, index=False)
        logger.info(f"Failed extraction metadata saved to {CONFIG.output_path_failed}")

    logger.info(f"Extraction complete. Results saved to {CONFIG.output_path}")


if __name__ == "__main__":
    asyncio.run(main())