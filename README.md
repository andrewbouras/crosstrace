# CrossTrace

**A Cross-Domain Dataset of Grounded Scientific Reasoning Traces for Hypothesis Generation**

CrossTrace is a dataset of 1,389 grounded scientific reasoning traces spanning biomedical research (518), AI/ML (605), and cross-domain work (266). Each trace captures the structured reasoning chain from established knowledge through intermediate logical steps to a novel hypothesis, with every step grounded in source paper text.

## Dataset

CrossTrace records follow an **Input/Trace/Output** schema:

- **Input**: Field context, conventional assumption, key prior work
- **Trace**: 3-6 ordered reasoning steps, each grounded in a source quotation
- **Output**: Core insight (spark), hypothesis, experimental approach

### Files

| File | Records | Description |
|------|---------|-------------|
| `data/all_traces.jsonl` | 1,389 | Full dataset with metadata and grounding |
| `data/train_ours_only.jsonl` | 1,180 | Training split (CrossTrace only) |
| `data/train_balanced.jsonl` | 4,720 | Balanced training (CrossTrace 2x + HypoGen capped) |
| `data/val.jsonl` | 102 | Validation split |
| `data/test_ours.jsonl` | 107 | CrossTrace test split |
| `data/test_hypogen.jsonl` | 50 | HypoGen test split (for cross-domain eval) |
| `data/manifest.json` | - | Split statistics and metadata |

### Domain Distribution

| Domain | Count | % |
|--------|-------|---|
| AI/ML | 605 | 43.6% |
| Biomedical | 518 | 37.3% |
| Cross-domain | 266 | 19.2% |

## Training

Fine-tuning uses QLoRA on Qwen2.5-7B-Instruct via the Axolotl framework.

### Configs

- `configs/qwen_qlora.yml` - Base config (LoRA r=16, alpha=32)
- `configs/qwen_qlora_r32.yml` - Higher capacity (LoRA r=32, alpha=64)

### Scripts

```bash
# Prepare training data (with optional HypoGen augmentation)
python scripts/prepare_data.py

# Run evaluation
python scripts/evaluate.py eval \
    --model-output generated.jsonl \
    --test-set data/test_hypogen.jsonl \
    --mode hypogen

# Google Colab training notebook
# See scripts/colab_train.py
```

## Results

Fine-tuning on CrossTrace yields substantial improvements over the untuned Qwen2.5-7B-Instruct baseline:

| Metric | Baseline | Run 1 (Ours) | Run 2b (Balanced) |
|--------|----------|--------------|-------------------|
| IAScore (GPT-4o) | 0.828 | 0.912 | **0.968** |
| IAScore (Claude) | 0.716 | 0.849 | **0.888** |
| Spark Cosine Sim | 0.221 | - | **0.620** |
| Structural Compliance | 0% | 100% | **100%** |

Balanced cross-domain training outperforms single-domain training, providing evidence that scientific reasoning patterns transfer across disciplines.

## Extraction Prompt

The full extraction prompt used to generate reasoning traces is in `prompts/extraction_prompt.md`. All extractions used Claude Sonnet 4 (`claude-sonnet-4-20250514`) with temperature 0.

## Schema

```
Record = (Input, Trace, Output, Metadata)
Input  = (field_context, conventional_assumption, key_prior_work)
Trace  = [step_1, step_2, ..., step_k]  where 3 <= k <= 6
Output = (spark, hypothesis, approach)
```

Each reasoning step in the Trace is paired with a `{step, source}` grounding object linking it to a specific section and quotation from the source paper.

## Citation

```bibtex
@article{bouras2026crosstrace,
  title={CrossTrace: A Cross-Domain Dataset of Grounded Scientific Reasoning Traces for Hypothesis Generation},
  author={Bouras, Andrew},
  year={2026}
}
```

## License

This dataset is released under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.
