import argparse
import asyncio
import logging
import os
from typing import List

import pandas as pd
from Bio import Entrez
from dotenv import load_dotenv
from tqdm import tqdm

from src.config import CONFIG, IDS_TO_KEEP, LOG_LEVEL, MODEL, PUBMED_QUERY
from src.extract_radiology_dataset_information_llm import (RadiologyDataset,
                                                           extract_with_agent)
from src.pubmed_utils import (
    add_column_to_isolate_mesh_terms_from_pubmed_matches,
    extract_pubmed_metadata, fetch_pubmed_citation_counts,
    fetch_pubmed_details, search_pubmed)

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
            os.remove(CONFIG.output_path_failed) if CONFIG.output_path_failed and os.path.exists(CONFIG.output_path_failed) else None
        else:
            logger.info(f"Output file {CONFIG.output_path} already exists. Set overwrite=True to overwrite it, or move/rename it before running the extraction. This will append to it but not replace existing lines, so you may get duplicates if you run multiple times without changing the output path.")
            return
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
    
    citation_counts_by_pmid = fetch_pubmed_citation_counts(ids)
    if CONFIG.min_citations > 0:
        ids = [id for id in ids if citation_counts_by_pmid.get(id, 0) >= CONFIG.min_citations]
        logger.info(f"After filtering out articles with fewer than {CONFIG.min_citations} citations, {len(ids)} articles remain.")
    
    logger.info(f"{len(ids)} articles will be processed for dataset extraction.")
    articles = fetch_pubmed_details(ids)

    if not articles:
        logger.warning("No articles found.")
        return

    extracted_datasets: List[RadiologyDataset] = []
    failed_metadata = []
    for article in tqdm(articles):
        try:
            publication_metadata = extract_pubmed_metadata(article, citation_counts_by_pmid, pubmed_query=PUBMED_QUERY)
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
        
        except KeyboardInterrupt:
            logger.error("\nInterrupted — exiting loop early.")
            break

    # Convert extracted datasets → dict rows
    rows = [d.model_dump() for d in extracted_datasets]
    new_df = pd.DataFrame(rows)

    logger.info(f"Extracted {len(new_df)} datasets from {len(articles)} articles.")
    logger.info(f"Failed to extract datasets from {len(failed_metadata)} articles.")

    # add column to isolate mesh terms in pubmed query
    if "pubmed_matches" in new_df.columns:
        new_df = add_column_to_isolate_mesh_terms_from_pubmed_matches(new_df, source_column="pubmed_matches")
    
    # convert list fields to comma-separated strings for CSV output
    cols_to_skip_joining = {"pubmed_matches"}
    for col in new_df.columns:
        if not new_df[col].empty and isinstance(new_df[col].iloc[0], list) and col not in cols_to_skip_joining:
            new_df[col] = new_df[col].apply(lambda x: ",".join(x) if isinstance(x, list) else x)
    
    if df is None or df.empty:
        df = new_df
    else:
        df = pd.concat([df, new_df], ignore_index=True)

    # Save to CSV
    df.to_csv(CONFIG.output_path, index=False)

    if CONFIG.output_path_failed:
        if len(failed_metadata) == 0:
            logger.info("No failed extractions to save.")
        else:
            df_failed = pd.DataFrame(failed_metadata)
            df_failed.to_csv(CONFIG.output_path_failed, index=False)
            logger.info(f"Failed extraction metadata saved to {CONFIG.output_path_failed}")

    logger.info(f"Extraction complete. Results saved to {CONFIG.output_path}")


if __name__ == "__main__":
    asyncio.run(main())