import os
from Bio import Entrez

Entrez.email = os.getenv("ENTREZ_EMAIL")

def get_mesh_terms(pmid):
    handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
    records = Entrez.read(handle)

    article = records["PubmedArticle"][0]
    mesh_list = article["MedlineCitation"].get("MeshHeadingList", [])

    mesh_terms = [
        mesh["DescriptorName"] for mesh in mesh_list
    ]

    for mesh in mesh_terms:
        print(mesh)

    return mesh_terms

from Bio import Entrez
import itertools

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
