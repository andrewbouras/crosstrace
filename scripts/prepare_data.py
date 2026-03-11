#!/usr/bin/env python3
"""
Prepare training data for hypothesis generation fine-tuning.

Converts our JSONL reasoning traces + HypoGen dataset into unified
chat-template format with domain-stratified train/val/test splits.

Usage:
    python3 prepare_data.py --our-data all_traces.jsonl --output-dir ./prepared
    python3 prepare_data.py --our-data all_traces.jsonl --hypogen --output-dir ./prepared
    python3 prepare_data.py --our-data all_traces.jsonl --hypogen --max-ratio 2.0 --output-dir ./prepared
"""

import argparse
import json
import hashlib
import random
from pathlib import Path
from collections import defaultdict

SYSTEM_PROMPT = (
    "You are a scientific reasoning assistant. Given a description of the "
    "current state of knowledge in a research field and the conventional "
    "assumption being challenged, generate a novel hypothesis with "
    "step-by-step reasoning."
)

DATATRACE_SYSTEM_PROMPT = (
    "You are a scientific reasoning assistant. Given a dataset description "
    "and the current state of knowledge, analyze the data context and "
    "generate a novel hypothesis grounded in what the data reveals, "
    "with step-by-step reasoning from data observation to insight."
)


def format_datatrace_record(record: dict) -> dict:
    """Convert a DataTrace JSONL record to chat-template format.

    DataTrace records have a data_anchor in the input that describes the
    dataset and the key observation. The reasoning trace starts from data.
    """
    inp = record["input"]
    out = record["output"]
    trace = record.get("trace", [])
    meta = record.get("metadata", {})
    data_anchor = inp.get("data_anchor", {})

    # Build user message — include dataset context
    user_parts = []

    if data_anchor:
        ds_desc = []
        if data_anchor.get("dataset_id"):
            ds_desc.append(f"Dataset: {data_anchor['dataset_id']}")
        if data_anchor.get("dataset_type"):
            ds_desc.append(f"Type: {data_anchor['dataset_type']}")
        if data_anchor.get("what_they_measured"):
            ds_desc.append(f"Measurement: {data_anchor['what_they_measured']}")
        if data_anchor.get("key_observation"):
            ds_desc.append(f"Key observation: {data_anchor['key_observation']}")
        if ds_desc:
            user_parts.append("Data context:\n" + "\n".join(ds_desc))

    if inp.get("field_context"):
        user_parts.append(f"Field context: {inp['field_context']}")
    if inp.get("conventional_assumption"):
        user_parts.append(f"Conventional assumption: {inp['conventional_assumption']}")

    prior_work = inp.get("key_prior_work", [])
    if prior_work:
        refs = []
        for pw in prior_work:
            if isinstance(pw, dict):
                refs.append(pw.get("ref", str(pw)))
            else:
                refs.append(str(pw))
        if refs:
            user_parts.append(f"Key prior work: {'; '.join(refs)}")

    user_msg = "\n\n".join(user_parts)

    # Build assistant response
    assistant_parts = []
    if out.get("spark"):
        assistant_parts.append(f"Core insight: {out['spark']}")

    if trace:
        reasoning_lines = [f"{i}. {step}" for i, step in enumerate(trace, 1)]
        assistant_parts.append("Reasoning:\n" + "\n".join(reasoning_lines))

    if out.get("hypothesis"):
        assistant_parts.append(f"Hypothesis: {out['hypothesis']}")
    if out.get("approach"):
        assistant_parts.append(f"Approach: {out['approach']}")

    assistant_msg = "\n\n".join(assistant_parts)

    return {
        "messages": [
            {"role": "system", "content": DATATRACE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        "metadata": {
            "source": "datatrace",
            "domain": meta.get("domain", "unknown"),
            "data_subtype": meta.get("data_subtype", "unknown"),
            "paper_id": meta.get("paper_id", ""),
            "extraction_confidence": meta.get("extraction_confidence", 0),
            "explicitness": meta.get("explicitness", 0),
            "patterns": meta.get("patterns", []),
        },
    }


def format_our_record(record: dict) -> dict:
    """Convert our JSONL record to chat-template format."""
    inp = record["input"]
    out = record["output"]
    trace = record.get("trace", [])
    meta = record.get("metadata", {})

    # Build user message
    user_parts = []
    if inp.get("field_context"):
        user_parts.append(f"Field context: {inp['field_context']}")
    if inp.get("conventional_assumption"):
        user_parts.append(f"Conventional assumption: {inp['conventional_assumption']}")

    prior_work = inp.get("key_prior_work", [])
    if prior_work:
        refs = []
        for pw in prior_work:
            if isinstance(pw, dict):
                refs.append(pw.get("ref", str(pw)))
            else:
                refs.append(str(pw))
        if refs:
            user_parts.append(f"Key prior work: {'; '.join(refs)}")

    user_msg = "\n\n".join(user_parts)

    # Build assistant response
    assistant_parts = []
    if out.get("spark"):
        assistant_parts.append(f"Core insight: {out['spark']}")

    if trace:
        reasoning_lines = []
        for i, step in enumerate(trace, 1):
            reasoning_lines.append(f"{i}. {step}")
        assistant_parts.append("Reasoning:\n" + "\n".join(reasoning_lines))

    if out.get("hypothesis"):
        assistant_parts.append(f"Hypothesis: {out['hypothesis']}")
    if out.get("approach"):
        assistant_parts.append(f"Approach: {out['approach']}")

    assistant_msg = "\n\n".join(assistant_parts)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        "metadata": {
            "source": "ours",
            "domain": meta.get("domain", "unknown"),
            "paper_id": meta.get("paper_id", ""),
            "extraction_confidence": meta.get("extraction_confidence", 0),
            "explicitness": meta.get("explicitness", 0),
            "patterns": meta.get("patterns", []),
        },
    }


def format_hypogen_record(record: dict) -> dict:
    """Convert HypoGen dataset record to our chat-template format."""
    # HypoGen fields: bit, flip, spark, chain_of_reasoning, abstract, title
    user_parts = []

    # Use abstract as field context if available
    if record.get("abstract"):
        # Truncate long abstracts to keep consistent with our data
        abstract = record["abstract"][:2000]
        user_parts.append(f"Field context: {abstract}")

    if record.get("bit"):
        user_parts.append(f"Conventional assumption: {record['bit']}")

    user_msg = "\n\n".join(user_parts)

    # Build assistant response
    assistant_parts = []
    if record.get("spark"):
        assistant_parts.append(f"Core insight: {record['spark']}")

    if record.get("chain_of_reasoning"):
        # HypoGen chain_of_reasoning is a narrative paragraph.
        # Split into sentences and number them to match our format.
        chain = record["chain_of_reasoning"]
        # Keep as narrative since HypoGen uses first-person paragraph style
        assistant_parts.append(f"Reasoning:\n{chain}")

    if record.get("flip"):
        assistant_parts.append(f"Hypothesis: {record['flip']}")

    assistant_msg = "\n\n".join(assistant_parts)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        "metadata": {
            "source": "hypogen",
            "domain": "cs_ml",
            "paper_id": record.get("paper_id", ""),
            "extraction_confidence": 1.0,  # Human-curated
            "explicitness": 1.0,
            "patterns": [],
        },
    }


def load_our_data(path: str) -> list:
    """Load our JSONL export."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_hypogen_data() -> tuple:
    """Load HypoGen dataset from HuggingFace. Returns (train, test)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("UniverseTBD/hypogen-dr1")
        train_records = [dict(r) for r in ds["train"]]
        test_records = [dict(r) for r in ds["test"]]
        return train_records, test_records
    except Exception as e:
        print(f"Failed to load HypoGen dataset: {e}")
        print("Falling back to local file if available...")

        base_dir = Path(__file__).parent
        train_path = base_dir / "hypogen_train.jsonl"
        test_path = base_dir / "hypogen_test.jsonl"

        train_records = []
        test_records = []

        if train_path.exists():
            with open(train_path) as f:
                for line in f:
                    if line.strip():
                        train_records.append(json.loads(line))
            print(f"  Loaded {len(train_records)} HypoGen train from local file")

        if test_path.exists():
            with open(test_path) as f:
                for line in f:
                    if line.strip():
                        test_records.append(json.loads(line))
            print(f"  Loaded {len(test_records)} HypoGen test from local file")

        if not train_records:
            print("  ERROR: No HypoGen data found locally or from HuggingFace")

        return train_records, test_records


def domain_stratified_split(records: list, train_pct=0.85, val_pct=0.075):
    """Split records by domain, maintaining proportions in each split.

    Also ensures lineage-linked records (same paper_id chain) stay together.
    """
    # Group by domain
    by_domain = defaultdict(list)
    for r in records:
        domain = r["metadata"]["domain"]
        by_domain[domain].append(r)

    train, val, test = [], [], []

    for domain, domain_records in by_domain.items():
        # Shuffle deterministically
        random.shuffle(domain_records)
        n = len(domain_records)
        n_train = int(n * train_pct)
        n_val = int(n * val_pct)

        train.extend(domain_records[:n_train])
        val.extend(domain_records[n_train : n_train + n_val])
        test.extend(domain_records[n_train + n_val :])

    # Shuffle each split
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def compute_stats(records: list) -> dict:
    """Compute statistics for a set of records."""
    domains = defaultdict(int)
    sources = defaultdict(int)
    total_user_chars = 0
    total_assistant_chars = 0

    for r in records:
        domains[r["metadata"]["domain"]] += 1
        sources[r["metadata"]["source"]] += 1
        for msg in r["messages"]:
            if msg["role"] == "user":
                total_user_chars += len(msg["content"])
            elif msg["role"] == "assistant":
                total_assistant_chars += len(msg["content"])

    n = len(records)
    return {
        "count": n,
        "domains": dict(domains),
        "sources": dict(sources),
        "avg_user_chars": total_user_chars // max(n, 1),
        "avg_assistant_chars": total_assistant_chars // max(n, 1),
        "est_avg_tokens": (total_user_chars + total_assistant_chars) // max(n, 1) // 4,
    }


def write_jsonl(records: list, path: Path, include_metadata: bool = False):
    """Write records to JSONL. Optionally strip metadata for training."""
    with open(path, "w") as f:
        for r in records:
            if include_metadata:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            else:
                # Training format: just messages
                f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for hypothesis generation")
    parser.add_argument("--our-data", required=True, help="Path to our JSONL export")
    parser.add_argument("--hypogen", action="store_true", help="Include HypoGen augmentation data")
    parser.add_argument("--output-dir", required=True, help="Output directory for prepared data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    parser.add_argument("--train-pct", type=float, default=0.85, help="Training set percentage")
    parser.add_argument("--val-pct", type=float, default=0.075, help="Validation set percentage")
    parser.add_argument("--max-ratio", type=float, default=0, help="Max ratio of HypoGen:ours. 0=no limit, 2.0=cap HypoGen at 2x our train size")
    parser.add_argument("--oversample-ours", type=float, default=1.0, help="Oversample our data by this factor (e.g. 2.0 = duplicate our records)")
    parser.add_argument("--datatrace", type=str, default=None, help="Path to DataTrace JSONL export (data-anchored traces)")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load and convert our data ---
    print(f"Loading our data from {args.our_data}...")
    raw_ours = load_our_data(args.our_data)
    print(f"  Loaded {len(raw_ours)} records")

    ours_formatted = [format_our_record(r) for r in raw_ours]
    print(f"  Formatted {len(ours_formatted)} records")

    # --- Split our data ---
    print(f"\nSplitting our data ({args.train_pct}/{args.val_pct}/{1-args.train_pct-args.val_pct})...")
    our_train, our_val, our_test = domain_stratified_split(
        ours_formatted, train_pct=args.train_pct, val_pct=args.val_pct
    )
    print(f"  Train: {len(our_train)} | Val: {len(our_val)} | Test: {len(our_test)}")

    # --- Check for data leakage ---
    train_ids = {r["metadata"]["paper_id"] for r in our_train}
    val_ids = {r["metadata"]["paper_id"] for r in our_val}
    test_ids = {r["metadata"]["paper_id"] for r in our_test}
    overlap_tv = train_ids & val_ids
    overlap_tt = train_ids & test_ids
    overlap_vt = val_ids & test_ids
    if overlap_tv or overlap_tt or overlap_vt:
        print(f"  WARNING: Data leakage detected!")
        print(f"    Train-Val overlap: {len(overlap_tv)} papers")
        print(f"    Train-Test overlap: {len(overlap_tt)} papers")
        print(f"    Val-Test overlap: {len(overlap_vt)} papers")
    else:
        print(f"  No data leakage detected (paper IDs are disjoint)")

    # --- HypoGen augmentation ---
    hypogen_train_formatted = []
    hypogen_test_formatted = []

    if args.hypogen:
        print(f"\nLoading HypoGen dataset from HuggingFace...")
        hg_train, hg_test = load_hypogen_data()
        print(f"  HypoGen train: {len(hg_train)} | test: {len(hg_test)}")

        if hg_train:
            hypogen_train_formatted = [format_hypogen_record(r) for r in hg_train]
            print(f"  Formatted {len(hypogen_train_formatted)} HypoGen train records")
        if hg_test:
            hypogen_test_formatted = [format_hypogen_record(r) for r in hg_test]
            print(f"  Formatted {len(hypogen_test_formatted)} HypoGen test records")

    # --- DataTrace augmentation ---
    datatrace_formatted = []
    if args.datatrace:
        print(f"\nLoading DataTrace data from {args.datatrace}...")
        raw_dt = load_our_data(args.datatrace)
        print(f"  Loaded {len(raw_dt)} DataTrace records")
        datatrace_formatted = [format_datatrace_record(r) for r in raw_dt]
        print(f"  Formatted {len(datatrace_formatted)} DataTrace records")

    # --- Build final datasets ---
    # Handle ratio capping: limit HypoGen to max_ratio * our_train_size
    hg_for_combined = list(hypogen_train_formatted)
    if args.max_ratio > 0 and hg_for_combined:
        max_hg = int(len(our_train) * args.max_ratio)
        if len(hg_for_combined) > max_hg:
            random.shuffle(hg_for_combined)
            hg_for_combined = hg_for_combined[:max_hg]
            print(f"\n  Capped HypoGen to {max_hg} records (ratio {args.max_ratio}:1)")

    # Handle oversampling of our data
    our_train_for_combined = list(our_train)
    if args.oversample_ours > 1.0:
        n_extra = int(len(our_train) * (args.oversample_ours - 1.0))
        extra = random.choices(our_train, k=n_extra)
        our_train_for_combined.extend(extra)
        print(f"  Oversampled our data: {len(our_train)} -> {len(our_train_for_combined)} records")

    combined_train = our_train_for_combined + hg_for_combined
    random.shuffle(combined_train)

    print(f"\n{'='*60}")
    print(f"FINAL DATASET SUMMARY")
    print(f"{'='*60}")

    # Our-only datasets (for ablation: Run 1)
    print(f"\n--- Our Data Only (Run 1) ---")
    write_jsonl(our_train, output_dir / "train_ours_only.jsonl")
    stats = compute_stats(our_train)
    print(f"  Train: {stats['count']} records | {stats['domains']} | ~{stats['est_avg_tokens']} tok/record")

    write_jsonl(our_val, output_dir / "val.jsonl")
    stats = compute_stats(our_val)
    print(f"  Val:   {stats['count']} records | {stats['domains']}")

    write_jsonl(our_test, output_dir / "test_ours.jsonl", include_metadata=True)
    stats = compute_stats(our_test)
    print(f"  Test:  {stats['count']} records | {stats['domains']}")

    if args.hypogen and hypogen_train_formatted:
        # Combined dataset (for Run 2)
        print(f"\n--- Combined (Run 2) ---")
        write_jsonl(combined_train, output_dir / "train_combined.jsonl")
        stats = compute_stats(combined_train)
        print(f"  Train: {stats['count']} records | sources: {stats['sources']} | ~{stats['est_avg_tokens']} tok/record")

        # Also create a balanced variant: cap HypoGen at 2:1 ratio
        balanced_hg = list(hypogen_train_formatted)
        max_balanced = len(our_train) * 2
        if len(balanced_hg) > max_balanced:
            random.shuffle(balanced_hg)
            balanced_hg = balanced_hg[:max_balanced]
        # Oversample our data 2x in balanced set
        balanced_ours = list(our_train) + random.choices(our_train, k=len(our_train))
        balanced_train = balanced_ours + balanced_hg
        random.shuffle(balanced_train)
        write_jsonl(balanced_train, output_dir / "train_balanced.jsonl")
        stats = compute_stats(balanced_train)
        print(f"\n--- Balanced 2:1 (Run 2b) ---")
        print(f"  Train: {stats['count']} records | sources: {stats['sources']} | ~{stats['est_avg_tokens']} tok/record")

        if hypogen_test_formatted:
            write_jsonl(hypogen_test_formatted, output_dir / "test_hypogen.jsonl", include_metadata=True)
            print(f"  HypoGen test: {len(hypogen_test_formatted)} records")

    # --- DataTrace outputs ---
    if datatrace_formatted:
        print(f"\n--- DataTrace (Run 3) ---")

        # DataTrace-only training set
        write_jsonl(datatrace_formatted, output_dir / "train_datatrace_only.jsonl")
        stats = compute_stats(datatrace_formatted)
        print(f"  DataTrace only: {stats['count']} records | {stats['domains']} | ~{stats['est_avg_tokens']} tok/record")

        # CrossTrace + DataTrace combined
        cross_plus_data = list(our_train) + datatrace_formatted
        random.shuffle(cross_plus_data)
        write_jsonl(cross_plus_data, output_dir / "train_cross_plus_data.jsonl")
        stats = compute_stats(cross_plus_data)
        print(f"  Cross+Data:     {stats['count']} records | sources: {stats['sources']}")

        # Full combined: CrossTrace + DataTrace + HypoGen
        if hypogen_train_formatted:
            full_combined = list(combined_train) + datatrace_formatted
            random.shuffle(full_combined)
            write_jsonl(full_combined, output_dir / "train_full_combined.jsonl")
            stats = compute_stats(full_combined)
            print(f"  Full combined:  {stats['count']} records | sources: {stats['sources']}")

    # Write manifest
    manifest = {
        "seed": args.seed,
        "our_total": len(ours_formatted),
        "our_train": len(our_train),
        "our_val": len(our_val),
        "our_test": len(our_test),
        "hypogen_train": len(hypogen_train_formatted),
        "hypogen_test": len(hypogen_test_formatted),
        "combined_train": len(combined_train),
        "balanced_train": len(balanced_train) if args.hypogen and hypogen_train_formatted else 0,
        "datatrace": len(datatrace_formatted),
        "train_pct": args.train_pct,
        "val_pct": args.val_pct,
        "max_ratio": args.max_ratio,
        "oversample_ours": args.oversample_ours,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {output_dir / 'manifest.json'}")

    print(f"\n{'='*60}")
    print(f"Files written to {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
