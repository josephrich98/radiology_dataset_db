import os
from datetime import datetime
import pytest

from conftest import _paper_ground_truth, _paper_irrelevant, load_build_module

Entrez = pytest.importorskip("Bio.Entrez", reason="biopython is required for Entrez tests")

# keys from _paper_ground_truth that should resolve to DOI-backed papers
EXPECTED_PAPERS = (
    "radimagenet",
    "mimic_cxr",
    "uk_biobank",
    "tcia",
    "merlin",
)

# keys from _paper_irrelevant expected to be absent from PubMed DOI search
EXPECTED_IRRELEVANT_PAPERS = None


def _extract_dois(pubmed_articles):
    dois = set()
    for article in pubmed_articles:
        article_ids = article.get("PubmedData", {}).get("ArticleIdList", [])
        for article_id in article_ids:
            if article_id.attributes.get("IdType") == "doi":
                dois.add(str(article_id).lower())
    return dois


def _doi_from_link(link):
    if not link:
        return None

    lower_link = link.lower().strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi.org/"):
        if lower_link.startswith(prefix):
            return lower_link[len(prefix) :]
    return None

def to_pubmed_date(date_str):
    if not date_str:
        return None

    try:  # Month-YYYY
        dt = datetime.strptime(date_str, "%B-%Y")
        return dt.strftime("%Y/%m")
    except ValueError:
        try:  # DD-Month-YYYY
            dt = datetime.strptime(date_str, "%d-%B-%Y")
            return dt.strftime("%Y/%m/%d")
        except ValueError:
            return None

def _collect_expected_dates(papers, paper_keys):
    expected_dates = [to_pubmed_date(papers[paper_key].get("date")) for paper_key in paper_keys]
    if not all(expected_dates):
        return None  # if any date is missing, return None to skip date checks
    return expected_dates

def _collect_expected_dois(papers, paper_keys):
    expected_dois = []
    for paper_key in paper_keys:
        assert paper_key in papers, f"Paper key '{paper_key}' not found in paper source"
        doi = _doi_from_link(papers[paper_key].get("link"))
        assert doi, f"Paper '{paper_key}' has no DOI-formatted link"
        expected_dois.append(doi)
    return expected_dois

def _perform_pubmed_search(module, query, max_results=9999, expected_dois=None, expect_presence=True):
    try:
        pubmed_ids = module.search_pubmed(pubmed_query=query, max_results=max_results)
        if expect_presence:
            assert pubmed_ids, f"Entrez search returned no PubMed IDs for query: {query}"

        pubmed_articles = module.fetch_pubmed_details(pubmed_ids) if pubmed_ids else []
    except Exception as exc:  # pragma: no cover - network/service dependency
        pytest.skip(f"Entrez service unavailable: {exc}")

    found_dois = _extract_dois(pubmed_articles)

    if expect_presence:
        missing_dois = [doi for doi in expected_dois if doi.lower() not in found_dois]
        assert not missing_dois, (
            f"Missing expected DOI(s) in Entrez results: {missing_dois}. "
            f"Found DOI(s): {sorted(found_dois)}"
        )
    else:
        unexpected_dois = [doi for doi in expected_dois if doi.lower() in found_dois]
        assert not unexpected_dois, (
            f"Unexpected DOI(s) found in Entrez results: {unexpected_dois}. "
            f"Found DOI(s): {sorted(found_dois)}"
        )

def _assert_entrez_for_expected_dois(monkeypatch, expected_dois, expected_dates=None, expect_presence=True):
    entrez_email = os.getenv("ENTREZ_EMAIL")
    if not entrez_email:
        pytest.skip("ENTREZ_EMAIL is required for live Entrez tests")
    Entrez.email = entrez_email

    if os.getenv("ENTREZ_API_KEY"):
        print("Using Entrez API key from environment variable for test.")
        Entrez.api_key = os.getenv("ENTREZ_API_KEY")

    config_module = load_build_module(
        monkeypatch,
        module_name="config",
        live=True,
        skip_on_missing_dependency=True,
    )
    pubmed_utils_module = load_build_module(
        monkeypatch,
        module_name="pubmed_utils",
        live=True,
        skip_on_missing_dependency=True,
    )

    query = config_module.PUBMED_QUERY

    if expected_dates is None:  # search for all one time
        print("Doing single Entrez search without date filtering for expected DOIs.")
        _perform_pubmed_search(pubmed_utils_module, query, expected_dois=expected_dois, expect_presence=expect_presence)
        
    else:
        print("Doing Entrez searches with date filtering for expected DOIs.")
        for expected_doi, expected_date in zip(expected_dois, expected_dates):
            date_query = f"{expected_date}[PDAT]" if expected_date else ""
            specific_query = f"{query} AND {date_query}" if date_query else query

            _perform_pubmed_search(pubmed_utils_module, specific_query, expected_dois=[expected_doi], expect_presence=expect_presence)

@pytest.mark.integration
@pytest.mark.slow
def test_entrez_search_contains_expected_paper_dois(monkeypatch):
    papers = _paper_ground_truth()

    expected_papers = EXPECTED_PAPERS
    if expected_papers is None:
        expected_papers = papers.keys()  # test all papers if EXPECTED_PAPERS is not set

    expected_dois = _collect_expected_dois(papers, expected_papers)
    expected_dates = _collect_expected_dates(papers, expected_papers)
    _assert_entrez_for_expected_dois(monkeypatch, expected_dois, expected_dates=expected_dates, expect_presence=True)

@pytest.mark.integration
@pytest.mark.slow
def test_entrez_search_absent_for_irrelevant_paper_dois(monkeypatch):
    papers = _paper_irrelevant()

    expected_papers = EXPECTED_IRRELEVANT_PAPERS
    if expected_papers is None:
        expected_papers = papers.keys()  # test all papers if EXPECTED_IRRELEVANT_PAPERS is not set

    expected_dois = _collect_expected_dois(papers, expected_papers)
    expected_dates = _collect_expected_dates(papers, expected_papers)
    _assert_entrez_for_expected_dois(monkeypatch, expected_dois, expected_dates=expected_dates, expect_presence=False)
