#!/usr/bin/env python3
"""
Evaluation harness for hypothesis generation model.

Computes:
1. HypoGen IAScore proxy (sentence-transformer cosine similarity + optional LLM judge)
2. Internal quality metrics (spark similarity, hypothesis ROUGE-L, domain accuracy)

Usage:
    # Evaluate a model on HypoGen test set
    python3 evaluate.py --model-output generated.jsonl --test-set test_hypogen.jsonl --mode hypogen

    # Evaluate on our internal test set
    python3 evaluate.py --model-output generated.jsonl --test-set test_ours.jsonl --mode internal

    # Generate outputs from a model (run on Colab with GPU)
    python3 evaluate.py --generate --model-path ./merged-hypothesis-7b --test-set test_hypogen.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

# Optional imports -- gracefully handle missing
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False


def load_jsonl(path: str) -> list:
    """Load JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_fields_from_assistant(text: str) -> dict:
    """Parse structured fields from assistant response text."""
    fields = {"spark": "", "reasoning": "", "hypothesis": "", "approach": ""}

    lines = text.split("\n")
    current_field = None
    current_content = []

    for line in lines:
        line_lower = line.strip().lower()
        if line_lower.startswith("core insight:"):
            if current_field:
                fields[current_field] = "\n".join(current_content).strip()
            current_field = "spark"
            current_content = [line.split(":", 1)[1].strip()]
        elif line_lower.startswith("reasoning:"):
            if current_field:
                fields[current_field] = "\n".join(current_content).strip()
            current_field = "reasoning"
            current_content = [line.split(":", 1)[1].strip() if ":" in line else ""]
        elif line_lower.startswith("hypothesis:"):
            if current_field:
                fields[current_field] = "\n".join(current_content).strip()
            current_field = "hypothesis"
            current_content = [line.split(":", 1)[1].strip()]
        elif line_lower.startswith("approach:"):
            if current_field:
                fields[current_field] = "\n".join(current_content).strip()
            current_field = "approach"
            current_content = [line.split(":", 1)[1].strip()]
        else:
            current_content.append(line)

    if current_field:
        fields[current_field] = "\n".join(current_content).strip()

    return fields


def compute_cosine_similarity(generated: list, reference: list, model_name="all-MiniLM-L6-v2"):
    """Compute pairwise cosine similarity using sentence-transformers."""
    if not HAS_SBERT:
        print("WARNING: sentence-transformers not installed. Skipping cosine similarity.")
        return []

    model = SentenceTransformer(model_name)
    gen_embeddings = model.encode(generated, convert_to_tensor=True)
    ref_embeddings = model.encode(reference, convert_to_tensor=True)

    similarities = []
    for i in range(len(generated)):
        sim = st_util.cos_sim(gen_embeddings[i], ref_embeddings[i]).item()
        similarities.append(sim)

    return similarities


def compute_rouge_l(generated: list, reference: list):
    """Compute ROUGE-L F1 scores."""
    if not HAS_ROUGE:
        print("WARNING: rouge-score not installed. Skipping ROUGE-L.")
        return []

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for gen, ref in zip(generated, reference):
        result = scorer.score(ref, gen)
        scores.append(result["rougeL"].fmeasure)
    return scores


def evaluate_hypogen(generated_records: list, test_records: list):
    """Evaluate against HypoGen test set.

    Primary metric: IAScore proxy via cosine similarity between
    generated flip/hypothesis and ground-truth flip/hypothesis.
    """
    print(f"\n{'='*60}")
    print(f"HypoGen IAScore Evaluation")
    print(f"{'='*60}")
    print(f"Generated records: {len(generated_records)}")
    print(f"Test records: {len(test_records)}")

    # Extract generated and reference hypotheses
    gen_hypotheses = []
    ref_hypotheses = []
    gen_sparks = []
    ref_sparks = []

    for gen, ref in zip(generated_records, test_records):
        # Parse generated output
        gen_msg = ""
        for msg in gen.get("messages", []):
            if msg["role"] == "assistant":
                gen_msg = msg["content"]
        gen_fields = extract_fields_from_assistant(gen_msg)

        # Parse reference output
        ref_msg = ""
        for msg in ref.get("messages", []):
            if msg["role"] == "assistant":
                ref_msg = msg["content"]
        ref_fields = extract_fields_from_assistant(ref_msg)

        # Combine hypothesis + approach as "flip" (HypoGen's output)
        gen_flip = f"{gen_fields['hypothesis']} {gen_fields['approach']}".strip()
        ref_flip = f"{ref_fields['hypothesis']} {ref_fields['approach']}".strip()

        if gen_flip and ref_flip:
            gen_hypotheses.append(gen_flip)
            ref_hypotheses.append(ref_flip)

        if gen_fields["spark"] and ref_fields["spark"]:
            gen_sparks.append(gen_fields["spark"])
            ref_sparks.append(ref_fields["spark"])

    # Cosine similarity (IAScore proxy)
    print(f"\n--- Cosine Similarity (IAScore Proxy) ---")
    if gen_hypotheses:
        hyp_sims = compute_cosine_similarity(gen_hypotheses, ref_hypotheses)
        if hyp_sims:
            mean_sim = sum(hyp_sims) / len(hyp_sims)
            print(f"  Hypothesis alignment: {mean_sim:.4f} (n={len(hyp_sims)})")
            print(f"  Min: {min(hyp_sims):.4f} | Max: {max(hyp_sims):.4f}")

    if gen_sparks:
        spark_sims = compute_cosine_similarity(gen_sparks, ref_sparks)
        if spark_sims:
            mean_sim = sum(spark_sims) / len(spark_sims)
            print(f"  Spark alignment:      {mean_sim:.4f} (n={len(spark_sims)})")

    # ROUGE-L
    print(f"\n--- ROUGE-L ---")
    if gen_hypotheses:
        rouge_scores = compute_rouge_l(gen_hypotheses, ref_hypotheses)
        if rouge_scores:
            mean_rouge = sum(rouge_scores) / len(rouge_scores)
            print(f"  Hypothesis ROUGE-L: {mean_rouge:.4f}")

    return {
        "n": len(gen_hypotheses),
        "hypothesis_cosine_sim": sum(hyp_sims) / len(hyp_sims) if hyp_sims else None,
        "spark_cosine_sim": sum(spark_sims) / len(spark_sims) if spark_sims else None,
        "hypothesis_rouge_l": sum(rouge_scores) / len(rouge_scores) if rouge_scores else None,
    }


def evaluate_internal(generated_records: list, test_records: list):
    """Evaluate on our internal test set.

    Metrics: spark similarity, hypothesis ROUGE-L, domain accuracy.
    """
    print(f"\n{'='*60}")
    print(f"Internal Quality Evaluation")
    print(f"{'='*60}")
    print(f"Generated records: {len(generated_records)}")
    print(f"Test records: {len(test_records)}")

    gen_sparks, ref_sparks = [], []
    gen_hyps, ref_hyps = [], []
    domain_correct = 0
    domain_total = 0

    for gen, ref in zip(generated_records, test_records):
        gen_msg = ""
        for msg in gen.get("messages", []):
            if msg["role"] == "assistant":
                gen_msg = msg["content"]
        gen_fields = extract_fields_from_assistant(gen_msg)

        ref_msg = ""
        for msg in ref.get("messages", []):
            if msg["role"] == "assistant":
                ref_msg = msg["content"]
        ref_fields = extract_fields_from_assistant(ref_msg)

        if gen_fields["spark"] and ref_fields["spark"]:
            gen_sparks.append(gen_fields["spark"])
            ref_sparks.append(ref_fields["spark"])

        if gen_fields["hypothesis"] and ref_fields["hypothesis"]:
            gen_hyps.append(gen_fields["hypothesis"])
            ref_hyps.append(ref_fields["hypothesis"])

        # Domain accuracy -- check if generated content mentions domain-appropriate terms
        ref_meta = ref.get("metadata", {})
        if ref_meta.get("domain"):
            domain_total += 1
            # Simple heuristic: if biomedical ref and generated mentions medical terms
            # This is a placeholder -- LLM judge would be more accurate
            domain_correct += 1  # Placeholder: always correct until LLM judge implemented

    print(f"\n--- Spark Similarity ---")
    spark_sims = []
    if gen_sparks:
        spark_sims = compute_cosine_similarity(gen_sparks, ref_sparks)
        if spark_sims:
            mean_sim = sum(spark_sims) / len(spark_sims)
            print(f"  Cosine similarity: {mean_sim:.4f} (n={len(spark_sims)})")

    print(f"\n--- Hypothesis Quality ---")
    rouge_scores = []
    if gen_hyps:
        rouge_scores = compute_rouge_l(gen_hyps, ref_hyps)
        if rouge_scores:
            print(f"  ROUGE-L: {sum(rouge_scores)/len(rouge_scores):.4f}")

        hyp_sims = compute_cosine_similarity(gen_hyps, ref_hyps)
        if hyp_sims:
            print(f"  Cosine similarity: {sum(hyp_sims)/len(hyp_sims):.4f}")

    print(f"\n--- Domain Accuracy ---")
    if domain_total:
        print(f"  {domain_correct}/{domain_total} ({100*domain_correct/domain_total:.1f}%)")
        print(f"  (Note: placeholder -- needs LLM judge for real assessment)")

    # Per-domain breakdown
    domain_scores = defaultdict(list)
    for i, ref in enumerate(test_records):
        domain = ref.get("metadata", {}).get("domain", "unknown")
        if i < len(spark_sims):
            domain_scores[domain].append(spark_sims[i])

    if domain_scores:
        print(f"\n--- Per-Domain Spark Similarity ---")
        for domain, scores in sorted(domain_scores.items()):
            print(f"  {domain}: {sum(scores)/len(scores):.4f} (n={len(scores)})")

    return {
        "n": len(gen_sparks),
        "spark_cosine_sim": sum(spark_sims) / len(spark_sims) if spark_sims else None,
        "hypothesis_rouge_l": sum(rouge_scores) / len(rouge_scores) if rouge_scores else None,
    }


def generate_outputs(model_path: str, test_set_path: str, output_path: str):
    """Generate outputs from a model for evaluation.

    Designed to run on Colab Pro with GPU.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        print("ERROR: transformers and torch required. Run on Colab Pro with GPU.")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    test_records = load_jsonl(test_set_path)
    print(f"Generating outputs for {len(test_records)} test records...")

    generated = []
    for i, record in enumerate(test_records):
        # Extract the user message (input)
        messages = []
        for msg in record["messages"]:
            if msg["role"] in ("system", "user"):
                messages.append(msg)

        # Tokenize
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        # Decode only the new tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Build output record matching test format
        gen_record = {
            "messages": messages + [{"role": "assistant", "content": response}],
            "metadata": record.get("metadata", {}),
        }
        generated.append(gen_record)

        if (i + 1) % 10 == 0:
            print(f"  Generated {i+1}/{len(test_records)}")

    # Save
    with open(output_path, "w") as f:
        for r in generated:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nOutputs saved to {output_path}")
    return generated


def main():
    parser = argparse.ArgumentParser(description="Evaluate hypothesis generation model")
    subparsers = parser.add_subparsers(dest="command")

    # Evaluate command
    eval_parser = subparsers.add_parser("eval", help="Evaluate generated outputs")
    eval_parser.add_argument("--model-output", required=True, help="JSONL of generated outputs")
    eval_parser.add_argument("--test-set", required=True, help="JSONL of test records (with ground truth)")
    eval_parser.add_argument("--mode", choices=["hypogen", "internal", "both"], default="both")
    eval_parser.add_argument("--output", help="Save results to JSON file")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate outputs from a model")
    gen_parser.add_argument("--model-path", required=True, help="Path to model")
    gen_parser.add_argument("--test-set", required=True, help="JSONL of test records")
    gen_parser.add_argument("--output", required=True, help="Output JSONL path")

    args = parser.parse_args()

    if args.command == "generate":
        generate_outputs(args.model_path, args.test_set, args.output)

    elif args.command == "eval":
        generated = load_jsonl(args.model_output)
        test_records = load_jsonl(args.test_set)

        if len(generated) != len(test_records):
            print(f"WARNING: {len(generated)} generated vs {len(test_records)} test records")
            n = min(len(generated), len(test_records))
            generated = generated[:n]
            test_records = test_records[:n]

        results = {}
        if args.mode in ("hypogen", "both"):
            results["hypogen"] = evaluate_hypogen(generated, test_records)
        if args.mode in ("internal", "both"):
            results["internal"] = evaluate_internal(generated, test_records)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
