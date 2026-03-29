from __future__ import annotations

from difflib import SequenceMatcher
import inspect
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import pandas as pd


def _validate_required_columns(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    required_columns: Iterable[str],
) -> None:
    """Ensure both dataframes contain required columns."""
    for col in required_columns:
        if col not in df_left.columns:
            raise ValueError(f"Left dataframe is missing required column: {col}")
        if col not in df_right.columns:
            raise ValueError(f"Right dataframe is missing required column: {col}")


def _validate_identical_columns(df_left: pd.DataFrame, df_right: pd.DataFrame) -> None:
    """Ensure both dataframes have exactly the same set of columns."""
    left_cols = set(df_left.columns)
    right_cols = set(df_right.columns)
    if left_cols != right_cols:
        only_left = sorted(left_cols - right_cols)
        only_right = sorted(right_cols - left_cols)
        raise ValueError(
            "Dataframes must have identical columns. "
            f"Only in left: {only_left}. Only in right: {only_right}."
        )


def _unique_title_set(df: pd.DataFrame, paper_title_col: str = "paper_title") -> Set[str]:
    """Return normalized non-empty unique paper titles."""
    titles = (
        df[paper_title_col]
        .dropna()
        .astype(str)
        .str.strip()
    )
    titles = titles[titles != ""]
    return set(titles.tolist())


def _to_bool(value: Any) -> Optional[bool]:
    """Best-effort conversion of heterogeneous values to boolean."""
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    value_str = str(value).strip().lower()
    if value_str in {"true", "t", "yes", "y", "1"}:
        return True
    if value_str in {"false", "f", "no", "n", "0"}:
        return False
    return None


def _build_paired_frame(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    paper_title_col: str = "paper_title",
) -> pd.DataFrame:
    """Inner-join two dataframes on paper title for pairwise comparisons."""
    _validate_required_columns(df_left, df_right, [paper_title_col])
    return df_left.merge(df_right, on=paper_title_col, how="inner", suffixes=("_left", "_right"))


def _to_set_from_comma_separated(value: Any) -> Set[str]:
    """Parse comma-separated strings into normalized token sets."""
    if pd.isna(value):
        return set()
    items = [item.strip() for item in str(value).split(",")]
    return {item for item in items if item}


def compare_dbs(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    merge_col: str = "paper_title",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Merge two dataframes by paper title and report uniqueness stats.

    Returns:
        merged_df: outer merge of both dataframes on paper_title
        report: counts of unique titles in each dataframe and combined
    """
    # _validate_identical_columns(df_left, df_right)
    _validate_required_columns(df_left, df_right, [merge_col])

    left_titles = _unique_title_set(df_left, merge_col)
    right_titles = _unique_title_set(df_right, merge_col)

    report = {
        "titles_left": len(left_titles),
        "titles_right": len(right_titles),
        "titles_union": len(left_titles | right_titles),
        "titles_only_left": len(left_titles - right_titles),
        "titles_only_right": len(right_titles - left_titles),
        "titles_intersection": len(left_titles & right_titles),
    }

    # merged_df = df_left.merge(
    #     df_right,
    #     on=merge_col,
    #     how="outer",
    #     suffixes=("_left", "_right"),
    #     indicator=True,
    # )
    return report

def verified_unverified_report(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    merge_col: str = "paper_title",
    verified_col: str = "verified",
) -> Optional[Dict[str, int]]:
    """
    Report unique verified/unverified paper-title counts in each dataframe and combined.

    Returns None if the verified column is missing from either dataframe.
    """
    if verified_col not in df_left.columns or verified_col not in df_right.columns:
        return None
    _validate_required_columns(df_left, df_right, [merge_col, verified_col])

    def build_sets(df: pd.DataFrame) -> Tuple[Set[str], Set[str]]:
        valid = df[[merge_col, verified_col]].copy()
        valid["_verified_bool"] = valid[verified_col].apply(_to_bool).astype("boolean")
        valid = valid.dropna(subset=[merge_col, "_verified_bool"])
        valid[merge_col] = valid[merge_col].astype(str).str.strip()
        valid = valid[valid[merge_col] != ""]

        verified_titles = set(valid.loc[valid["_verified_bool"], merge_col].tolist())
        unverified_titles = set(valid.loc[~valid["_verified_bool"], merge_col].tolist())
        return verified_titles, unverified_titles

    left_verified, left_unverified = build_sets(df_left)
    right_verified, right_unverified = build_sets(df_right)

    return {
        "total_left": len(left_verified) + len(left_unverified),
        "total_right": len(right_verified) + len(right_unverified),
        "verified_left": len(left_verified),
        "verified_right": len(right_verified),
        "verified_union": len(left_verified | right_verified),
        "verified_only_left": len(left_verified - right_verified),
        "verified_only_right": len(right_verified - left_verified),
        "verified_intersection": len(left_verified & right_verified),
        "unverified_left": len(left_unverified),
        "unverified_right": len(right_unverified),
        "unverified_union": len(left_unverified | right_unverified),
        "unverified_only_left": len(left_unverified - right_unverified),
        "unverified_only_right": len(right_unverified - left_unverified),
        "unverified_intersection": len(left_unverified & right_unverified),
        "fraction_verified_left": len(left_verified) / (len(left_verified) + len(left_unverified)) if (len(left_verified) + len(left_unverified)) > 0 else float("nan"),  # aka PPV
        "fraction_verified_right": len(right_verified) / (len(right_verified) + len(right_unverified)) if (len(right_verified) + len(right_unverified)) > 0 else float("nan"),  # aka PPV
    }


def mean_sequence_matcher_in_column(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    column_name: str,
    merge_col: str = "paper_title",
) -> Dict[str, float]:
    """Compute mean difflib.SequenceMatcher ratio on pairwise title-matched rows."""
    print(f"{inspect.currentframe().f_code.co_name} for column {column_name}")
    _validate_required_columns(df_left, df_right, [merge_col, column_name])
    paired = _build_paired_frame(df_left, df_right, merge_col)

    col_left = f"{column_name}_left"
    col_right = f"{column_name}_right"
    if col_left not in paired.columns or col_right not in paired.columns:
        results = {"mean_ratio": float("nan"), "n_compared": 0}
        print(results)
        return None

    paired = paired[[merge_col, col_left, col_right]].copy()
    
    ratios = []
    for left_val, right_val in zip(paired[col_left], paired[col_right]):
        if pd.isna(left_val) or pd.isna(right_val):
            continue
        ratio = SequenceMatcher(None, str(left_val), str(right_val)).ratio()
        ratios.append(ratio)

    if not ratios:
        results = {"mean_ratio": float("nan"), "n_compared": 0}
        print(results)
        return None
    
    results = {"mean_ratio": float(sum(ratios) / len(ratios)), "n_compared": len(ratios)}
    print(results)
    return paired

def identical_numbers_in_column(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    column_name: str,
    merge_col: str = "paper_title",
) -> Dict[str, Any]:
    """Check whether numbers in a given column are identical across title-matched rows."""
    print(f"{inspect.currentframe().f_code.co_name} for column {column_name}")
    _validate_required_columns(df_left, df_right, [merge_col, column_name])
    paired = _build_paired_frame(df_left, df_right, merge_col)

    col_left = f"{column_name}_left"
    col_right = f"{column_name}_right"
    if col_left not in paired.columns or col_right not in paired.columns:
        results = {
            "all_identical": False,
            "n_compared": 0,
            "n_identical": 0,
            "identical_fraction": float("nan"),
        }
        print(results)
        return None
    paired = paired[[merge_col, col_left, col_right]].copy()

    left_num = pd.to_numeric(paired[col_left], errors="coerce")
    right_num = pd.to_numeric(paired[col_right], errors="coerce")
    valid_mask = left_num.notna() & right_num.notna()

    compared = int(valid_mask.sum())
    if compared == 0:
        results = {
            "all_identical": False,
            "n_compared": 0,
            "n_identical": 0,
            "identical_fraction": float("nan"),
        }
        print(results)
        return None

    identical_mask = (left_num[valid_mask] == right_num[valid_mask])
    n_identical = int(identical_mask.sum())
    fraction = n_identical / compared
    results = {
        "all_identical": n_identical == compared,
        "n_compared": compared,
        "n_identical": n_identical,
        "identical_fraction": float(fraction),
    }
    print(results)
    return paired

def mean_jaccard_in_column(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    column_name: str,
    merge_col: str = "paper_title",
) -> Dict[str, float]:
    """Compute mean Jaccard index for comma-separated set values on title-matched rows."""
    print(f"{inspect.currentframe().f_code.co_name} for column {column_name}")
    _validate_required_columns(df_left, df_right, [merge_col, column_name])
    paired = _build_paired_frame(df_left, df_right, merge_col)

    col_left = f"{column_name}_left"
    col_right = f"{column_name}_right"
    if col_left not in paired.columns or col_right not in paired.columns:
        results = {"mean_jaccard": float("nan"), "n_compared": 0}
        print(results)
        return None
    
    paired = paired[[merge_col, col_left, col_right]].copy()
    
    jaccards = []
    for left_val, right_val in zip(paired[col_left], paired[col_right]):
        if pd.isna(left_val) and pd.isna(right_val):
            continue

        left_set = _to_set_from_comma_separated(left_val)
        right_set = _to_set_from_comma_separated(right_val)
        union = left_set | right_set
        if not union:
            # Treat two empty sets as a perfect match.
            jaccards.append(1.0)
        else:
            jaccards.append(len(left_set & right_set) / len(union))

    if not jaccards:
        results = {"mean_jaccard": float("nan"), "n_compared": 0}
        print(results)
        return None
    
    results = {"mean_jaccard": float(sum(jaccards) / len(jaccards)), "n_compared": len(jaccards)}
    print(results)
    return paired
