import ast
import logging
import os
import re
import time
from typing import Dict, List, Optional, Set, Union

import requests
from Bio import Entrez
import pandas as pd
from more_itertools import chunked
from tqdm import tqdm

from radiology_dataset_db.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

Entrez.email = os.getenv("ENTREZ_EMAIL")

def get_count(query):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=0)
    record = Entrez.read(handle)
    return int(record["Count"])


def split_query(query):
    # assumes structure: (A OR B OR C) AND (X OR Y OR Z)
    parts = query.split("AND")
    left = parts[0].strip()[1:-1]   # remove parentheses
    right = parts[1].strip()[1:-1]

    left_terms = [t.strip() for t in left.split(" OR ")]
    right_terms = [t.strip() for t in right.split(" OR ")]

    return left_terms, right_terms


def build_query(left_terms, right_terms):
    left = " OR ".join(left_terms)
    right = " OR ".join(right_terms)
    return f"({left}) AND ({right})"


def drop_one_analysis(query):
    left_terms, right_terms = split_query(query)

    results = []

    base_count = get_count(query)
    results.append(("BASE", base_count))

    # drop from left side
    for i in range(len(left_terms)):
        new_left = left_terms[:i] + left_terms[i+1:]
        new_query = build_query(new_left, right_terms)
        count = get_count(new_query)
        results.append((f"DROP LEFT: {left_terms[i]}", count))

    # drop from right side
    for i in range(len(right_terms)):
        new_right = right_terms[:i] + right_terms[i+1:]
        new_query = build_query(left_terms, new_right)
        count = get_count(new_query)
        results.append((f"DROP RIGHT: {right_terms[i]}", count))

    return results


def extract_mesh_terms_union(pubmed_query: str) -> Set[str]:
    """
    Return the union of all query terms ending in [MeSH].

    Quoted phrases are kept intact as single terms with spaces preserved.
    Example: '"Radiology Information Systems"[MeSH]' -> 'Radiology Information Systems'
    """
    mesh_terms: Set[str] = set()
    mesh_pattern = re.compile(r'("([^"]+)"|[^\s()]+)\s*\[MeSH\]', re.IGNORECASE)

    for full_match, quoted_term in mesh_pattern.findall(pubmed_query):
        term = quoted_term if quoted_term else full_match
        term = term.strip()
        if term:
            mesh_terms.add(term)

    return mesh_terms


def add_mesh_intersection_column(df, source_column: str, mesh_terms: Set[str], output_column: str):
    """
    From a comma-joined term column, write a new column containing the set
    intersection with the provided MeSH terms.
    """
    mesh_lookup = {term.casefold(): term for term in mesh_terms}

    def _intersect_terms(cell):
        if not isinstance(cell, str):
            return set()

        cell_terms = {part.strip() for part in cell.split(",") if part.strip()}
        matched = {
            mesh_lookup[term.casefold()]
            for term in cell_terms
            if term.casefold() in mesh_lookup
        }
        return matched

    df[output_column] = df[source_column].apply(_intersect_terms)
    return df


def add_column_to_isolate_mesh_terms_from_pubmed_matches(df, source_column="pubmed_matches", output_column="mesh_terms_in_pubmed_matches"):
    def _flatten_terms(value):
        if isinstance(value, (list, tuple, set)):
            for item in value:
                yield from _flatten_terms(item)
        elif isinstance(value, str):
            yield value

    def _extract_mesh_terms(cell):
        parsed = cell
        if isinstance(cell, str):
            try:
                parsed = ast.literal_eval(cell)
            except Exception:
                return []

        terms = [term.strip() for term in _flatten_terms(parsed)]
        ti_terms = [term for term in terms if term.endswith("[MeSH]")]
        return ti_terms

    df[output_column] = df[source_column].apply(_extract_mesh_terms)
    return df


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
        # TODO: implement more efficient fetching if we need to go beyond 10k, e.g. by splitting query into smaller date ranges or using other metadata filters
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
    id_list = [str(id) for id in id_list]  # ensure all IDs are strings
    for batch in tqdm(list(chunked(id_list, batch_size)), desc="Fetching PubMed details"):
        handle = Entrez.efetch(
            db="pubmed",
            id=",".join(batch),
            retmode="xml"
        )
        records = Entrez.read(handle)
        results.extend(records["PubmedArticle"])
    return results

def fetch_pubmed_citation_counts(pmids, batch_size=100, get_year=False):
    url = "https://icite.od.nih.gov/api/pubs"
    results = {}
    year_results = {}

    for batch in tqdm(list(chunked(pmids, batch_size)), desc="Fetching citation counts"):
        try:
            response = requests.get(url, params={"pmids": ",".join(batch)}, timeout=10)
            
            if response.status_code != 200:
                raise Exception(f"Status {response.status_code}")

            data = response.json()

            # fill results for successful ones
            for p in data.get("data", []):
                results[str(p["pmid"])] = p.get("citation_count", 0)
                if get_year:
                    year_results[str(p["pmid"])] = p.get("year", None)

            # handle missing PMIDs in response
            returned_pmids = {str(p["pmid"]) for p in data.get("data", [])}
            for pmid in batch:
                if pmid not in returned_pmids:
                    results[str(pmid)] = 0  # or None
                    if get_year:
                        year_results[str(pmid)] = None

        except Exception:
            # fallback: assign 0/None for entire failed batch
            for pmid in batch:
                results[str(pmid)] = 0  # or None
                if get_year:
                    year_results[str(pmid)] = None

    if get_year:
        return results, year_results
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

def _extract_mesh_terms(article) -> List[str]:
    try:
        mesh_list = article["MedlineCitation"].get("MeshHeadingList", [])
        return [mesh["DescriptorName"] for mesh in mesh_list]
    except Exception:
        return []

def _extract_keywords(article) -> List[str]:
    try:
        keyword_list = article["MedlineCitation"].get("KeywordList", [])
        keywords = []
        for kw_list in keyword_list:
            keywords.extend([str(k) for k in kw_list])
        return keywords
    except Exception:
        return []

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

def match_pubmed_query(query, title="", abstract="", mesh_terms=tuple()):
    """
    Match a PubMed boolean query against a paper's title/abstract/MeSH terms.

    Expected query structure:
      (<term1> OR <term2> OR ...) AND (<termA> OR <termB> OR ...) [AND ...]
      with an optional trailing: NOT (<termX> OR <termY> OR ...)

    Each term must end with one of: [MeSH], [ti], [tiab].

    Returns:
        List[List[str]] where each inner list corresponds to one AND section and
        contains the original query terms (from that OR block) that matched.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    if not title and not abstract and not mesh_terms:
        raise ValueError("At least one of title, abstract, or mesh_terms must be provided for matching.")

    query = " ".join(query.split())  # strip new lines
    mesh_terms = mesh_terms or []
    
    def _split_top_level_boolean(expr: str, operator: str) -> List[str]:
        parts = []
        depth = 0
        in_quote = False
        start = 0
        i = 0
        op = f" {operator} "
        n = len(expr)

        while i < n:
            ch = expr[i]
            if ch == '"':
                in_quote = not in_quote
                i += 1
                continue
            if not in_quote:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth < 0:
                        raise ValueError("Unbalanced parentheses in query.")
                elif depth == 0 and expr.startswith(op, i):
                    part = expr[start:i].strip()
                    if not part:
                        raise ValueError(f"Empty section around top-level {operator}.")
                    parts.append(part)
                    i += len(op)
                    start = i
                    continue
            i += 1

        if depth != 0:
            raise ValueError("Unbalanced parentheses in query.")
        if in_quote:
            raise ValueError("Unbalanced double quotes in query.")

        tail = expr[start:].strip()
        if not tail:
            raise ValueError(f"Empty trailing section after top-level {operator}.")
        parts.append(tail)
        return parts

    def _strip_trailing_top_level_not_clause(expr: str) -> str:
        """
        Remove a trailing top-level NOT clause so matching is based only on the
        positive (AND/OR) part of the query.
        """
        depth = 0
        in_quote = False
        i = 0
        n = len(expr)
        op = " NOT "
        last_not_idx = None

        while i < n:
            ch = expr[i]
            if ch == '"':
                in_quote = not in_quote
                i += 1
                continue
            if not in_quote:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth < 0:
                        raise ValueError("Unbalanced parentheses in query.")
                elif depth == 0 and expr.startswith(op, i):
                    last_not_idx = i
            i += 1

        if depth != 0:
            raise ValueError("Unbalanced parentheses in query.")
        if in_quote:
            raise ValueError("Unbalanced double quotes in query.")

        if last_not_idx is None:
            return expr.strip()

        positive_part = expr[:last_not_idx].strip()
        not_clause = expr[last_not_idx + len(op):].strip()
        if not positive_part:
            raise ValueError("Query cannot start with a top-level NOT clause.")
        if not not_clause:
            raise ValueError("Trailing top-level NOT clause is empty.")
        if not (not_clause.startswith("(") and not_clause.endswith(")")):
            raise ValueError("Trailing top-level NOT clause must be parenthesized.")
        return positive_part

    def _strip_outer_parens(section: str) -> str:
        section = section.strip()
        if len(section) < 2 or section[0] != "(" or section[-1] != ")":
            raise ValueError(
                "Query must be grouped as parenthesized OR blocks joined by AND."
            )
        return section[1:-1].strip()

    def _normalize_term_text(term_text: str) -> str:
        term_text = term_text.strip()
        if len(term_text) >= 2 and term_text[0] == '"' and term_text[-1] == '"':
            term_text = term_text[1:-1]
        return term_text.strip()

    def tokenize(text: str):
        return re.findall(r"\b\w+\b", text.casefold())

    query_without_final_not = _strip_trailing_top_level_not_clause(query.strip())
    and_sections = _split_top_level_boolean(query_without_final_not, "AND")
    if len(and_sections) < 1:
        raise ValueError("Query must contain at least one parenthesized OR section.")

    out = []
    field_re = re.compile(r"^(?P<term>.+?)\[(?P<field>MeSH|ti|tiab)\]$")
    title_l = title.casefold()
    abstract_l = abstract.casefold()
    mesh_set = {m.casefold() for m in mesh_terms}

    title_tokens = set(tokenize(title))
    abstract_tokens = set(tokenize(abstract))

    for section in and_sections:
        block = _strip_outer_parens(section)
        or_terms = _split_top_level_boolean(block, "OR")
        if not or_terms:
            raise ValueError("Each AND section must contain at least one OR term.")

        matched_terms = []
        for raw_term in or_terms:
            term_token = raw_term.strip()
            if len(term_token.split()) > 1 and ("'" not in term_token) and ('"' not in term_token):
                raise ValueError(f"Term '{term_token}' has multiple words and does not contain quotes. Please add quotes around the entire term.")
            m = field_re.match(term_token)
            if not m:
                raise ValueError(
                    f"Invalid term '{term_token}'. Each term must end with [MeSH], [ti], or [tiab]."
                )

            term_text = m.group("term")
            term_text_stripped = term_text.strip()
            is_quoted_phrase = (
                len(term_text_stripped) >= 2
                and term_text_stripped[0] == '"'
                and term_text_stripped[-1] == '"'
            )

            query_text = _normalize_term_text(term_text)
            if not query_text:
                raise ValueError(f"Empty term before field suffix in '{term_token}'.")

            field = m.group("field")
            needle = query_text.casefold()
            is_match = False

            if field == "MeSH":
                is_match = needle in mesh_set
            elif field == "ti":
                if is_quoted_phrase:
                    # Keep quoted phrases intact and match with spaces preserved.
                    is_match = needle in title_l
                else:
                    is_match = needle in title_tokens
            elif field == "tiab":
                if is_quoted_phrase:
                    # Keep quoted phrases intact and match with spaces preserved.
                    is_match = needle in title_l or needle in abstract_l
                else:
                    is_match = needle in title_tokens or needle in abstract_tokens

            if is_match:
                matched_terms.append(term_token)

        out.append(matched_terms)

    return out

def extract_pubmed_metadata(article, citation_counts_by_pmid: Optional[Dict[str, Optional[int]]] = None, pubmed_query: Optional[str] = None) -> Dict[str, Union[str, List[str], int, None]]:
    citation_counts_by_pmid = citation_counts_by_pmid or {}

    pmid = _extract_pmid(article)
    title = _extract_title(article)
    abstract = _extract_abstract(article)
    mesh_terms = _extract_mesh_terms(article)
    pubmed_matches = None
    if pubmed_query is not None:
        try:
            pubmed_matches = match_pubmed_query(pubmed_query, title=title, abstract=abstract, mesh_terms=mesh_terms)
            # for i, pubmed_matches_list in enumerate(pubmed_matches):
            #     if not pubmed_matches_list:
            #         logger.error(f"Query section {i} had no matches for article with PMID {pmid}. This may indicate an issue with the query structure or the matching logic.")
            #         pubmed_matches_list = None
        except Exception as e:
            logger.error(f"Error matching PubMed query for article with PMID {pmid}: {e}")
    
    pubmed_metadata = {
        "title": title,
        "abstract": abstract,
        "link": _extract_link(article),
        "year": _extract_year(article),
        "authors": _extract_authors(article),
        "journal": _extract_journal(article),
        "mesh_terms": mesh_terms,
        "keywords": _extract_keywords(article),
        "pmid": pmid,
        "citation_count": citation_counts_by_pmid.get(pmid) if pmid else None,
        "pubmed_matches": pubmed_matches,
    }

    return pubmed_metadata


def try_to_get_full_text(article, delay: float = 0.34) -> str | None:
    """
    Given a PubMed article dict (from Entrez.read),
    attempt to retrieve full text.

    Returns:
        - PMC full text XML (str), OR
        - PDF URL (str), OR
        - None
    """

    try:
        # -----------------------------
        # Extract identifiers
        # -----------------------------
        article_ids = article["PubmedData"]["ArticleIdList"]

        doi = None
        pmcid = None

        for id_obj in article_ids:
            id_type = id_obj.attributes.get("IdType")
            if id_type == "doi":
                doi = str(id_obj)
            elif id_type == "pmc":
                pmcid = str(id_obj)

        # -----------------------------
        # Try PMC (best case)
        # -----------------------------
        if pmcid:
            try:
                handle = Entrez.efetch(
                    db="pmc",
                    id=pmcid,
                    rettype="full",
                    retmode="xml"
                )
                full_text_xml = handle.read()
                time.sleep(delay)
                return full_text_xml
            except Exception:
                pass

        # -----------------------------
        # Try Unpaywall (PDF fallback)
        # -----------------------------
        if doi:
            try:
                url = f"https://api.unpaywall.org/v2/{doi}"
                params = {"email": Entrez.email}
                r = requests.get(url, params=params, timeout=10)

                if r.status_code == 200:
                    data = r.json()
                    pdf_url = data.get("best_oa_location", {}).get("url_for_pdf")
                    if pdf_url:
                        return pdf_url
            except Exception:
                pass

        return None

    except Exception:
        return None
    
def check_url(url, timeout=5):
    if pd.isna(url):
        return None
    try:
        response = requests.head(url, allow_redirects=True, timeout=timeout)
        if response.status_code < 400:
            return True
        # fallback to GET (some servers block HEAD)
        response = requests.get(url, allow_redirects=True, timeout=timeout)
        return response.status_code < 400
    except requests.RequestException:
        return False