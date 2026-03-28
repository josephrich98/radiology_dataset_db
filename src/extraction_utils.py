import logging
from dataclasses import dataclass
from typing import Optional, Type

from pydantic import BaseModel, Field

from src.is_database_paper_classifier_llm import llm_thinks_not_dataset_paper


class DatasetWithPaperMetadata(BaseModel):
    name: Optional[str] = Field(default=None)
    dataset_link: Optional[str] = None
    paper_title: Optional[str] = None
    paper_link: Optional[str] = None
    paper_year: Optional[int] = None
    paper_authors: list[str] = Field(default_factory=list)
    paper_journal: Optional[str] = None
    pmid: Optional[str] = None
    paper_citation_count: Optional[int] = None
    mesh_terms: list[str] = Field(default_factory=list)
    pubmed_matches: Optional[list[list[str]]] = None


@dataclass
class ExtractionDeps:
    title: str
    abstract: str


def populate_paper_metadata(
    output: DatasetWithPaperMetadata,
    title: str,
    publication_metadata: Optional[dict],
) -> None:
    output.paper_title = title
    publication_metadata = publication_metadata or {}
    output.paper_link = publication_metadata.get("link")
    output.paper_year = publication_metadata.get("year")
    output.paper_authors = publication_metadata.get("authors")
    output.paper_journal = publication_metadata.get("journal")
    output.pmid = publication_metadata.get("pmid")
    output.paper_citation_count = publication_metadata.get("citation_count")
    output.mesh_terms = publication_metadata.get("mesh_terms")
    output.pubmed_matches = publication_metadata.get("pubmed_matches")


async def run_dataset_extraction_with_common_logic(
    *,
    title: str,
    abstract: str,
    publication_metadata: Optional[dict],
    dataset_agent,
    agent_instructions: str,
    output_model: Type[DatasetWithPaperMetadata],
    logger: logging.Logger,
):
    if not title or not abstract:
        logger.debug("Missing title or abstract, skipping extraction.")
        return None

    deps = ExtractionDeps(title=title, abstract=abstract)

    try:
        unlikely_dataset_paper = await llm_thinks_not_dataset_paper(title, abstract)
        if unlikely_dataset_paper:
            logger.debug(
                "LLM thinks this is unlikely to be a dataset paper, rejecting without extraction."
            )
            return None

        result = await dataset_agent.run(agent_instructions, deps=deps)
        output = result.output
        logger.debug("LLM output: %s", output)

        if isinstance(output, output_model):
            populate_paper_metadata(output, title, publication_metadata)

            if not output.name:
                logger.debug("No dataset name extracted, rejecting output.")
                for msg in result.all_messages():
                    logger.debug(msg)
                return None

        return output

    except Exception as e:
        logger.error("LLM extraction failed: %s", e)
        return None
