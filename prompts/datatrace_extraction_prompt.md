# Data-Anchored Reasoning Trace Extraction

You are a scientific reasoning analyst. Your task is to extract the **data-anchored reasoning trace** — specifically, how the authors went from *observing patterns in a specific dataset* to their novel contribution.

This extends standard reasoning trace extraction with a critical addition: the dataset is the starting point, not just a tool. You must capture **what the data showed** and **how the data observation led to the hypothesis**.

You will receive:
- The **Introduction** and **Discussion** sections of a paper
- A **dataset anchor** (the specific dataset the paper references)

## Output Format

Return a single JSON object:

```json
{
  "paper_type": "original_research",
  "skip": false,
  "domain": "biomedical",

  "data_anchor": {
    "dataset_id": "GSE167523",
    "dataset_type": "geo",
    "role": "primary",
    "what_they_measured": "Gene expression profiles of 48 sepsis patients vs 20 healthy controls using RNA-seq",
    "key_observation": "DNASE1L3 was significantly downregulated in sepsis patients (logFC=-2.3, FDR<0.001)"
  },

  "prior_state": {
    "field_context": "What was known before this paper",
    "conventional_assumption": "The specific thing taken for granted",
    "key_prior_work": [
      {"ref": "Smith et al. 2020", "doi": "10.1234/example"}
    ]
  },

  "reasoning_trace": [
    "Step 1: The dataset revealed [specific observation from data]",
    "Step 2: This observation was unexpected because [contrast with prior knowledge]",
    "Step 3: The authors connected this to [cross-domain or mechanistic insight]",
    "Step 4: This led them to hypothesize [novel claim]"
  ],

  "reasoning_trace_grounded": [
    {"step": "Step 1: The dataset revealed...", "source": "Results, paragraph 2: 'exact quote'"},
    {"step": "Step 2: This observation...", "source": "Discussion, paragraph 1: 'exact quote'"}
  ],

  "novel_contribution": {
    "spark": "4-6 word core insight",
    "hypothesis": "The specific claim or research question",
    "approach": "How they tested or validated it"
  },

  "discovery_patterns": ["data_driven"],
  "gap_type": "method",
  "explicitness": 0.80,
  "explicitness_justification": "One sentence",
  "extraction_confidence": 0.85,
  "confidence_justification": "One sentence"
}
```

## Review Detection

If the paper is a review, meta-analysis, or synthesis:
```json
{"paper_type": "review", "skip": true}
```

## Data Anchor Instructions

The `data_anchor` field is **the most important new element**. It captures:

- **dataset_id**: The accession number or name (e.g., "GSE167523", "TCGA-BRCA", "MIMIC-IV", "NHANES 2017-2018", "ImageNet")
- **dataset_type**: One of: `geo`, `tcga`, `mimic`, `nhanes`, `imagenet`, `physionet`, `ukbiobank`, `arrayexpress`, `cellxgene`, `other`
- **role**: How the dataset was used:
  - `primary` — main dataset driving the analysis
  - `validation` — used to validate findings from another dataset
  - `reference` — used as a benchmark or comparison
- **what_they_measured**: One sentence describing what data they extracted (e.g., "RNA-seq gene expression of 200 tumor samples across 3 cancer types")
- **key_observation**: The specific finding from the data that sparked the reasoning chain. This should be a concrete, quantitative statement where possible.

### Critical Rule

**Step 1 of the reasoning trace MUST start with a data observation.** The entire point of DataTrace is that the data *initiated* the reasoning, not just supported it. If the paper's reasoning starts from theory and uses data as confirmation, note this in `data_anchor.role` as "validation" and still extract the trace, but be honest about the role.

## Reasoning Trace Instructions

Same rules as standard extraction:
- **3-6 steps** per trace, matching reasoning complexity
- Each step is **verifiable** and **logically connected** to the previous
- **1-2 sentences** per step (15-40 words)
- Ground every step with source quotes in `reasoning_trace_grounded`

### Data-Specific Step Types

The first 1-2 steps should typically be one of:
- **Data observation**: "The dataset showed X" (quantitative)
- **Data anomaly**: "Unexpectedly, the data revealed Y, contradicting the assumption that Z"
- **Data pattern**: "Across N samples, a consistent pattern of W emerged"

## Domain Tagging

- `biomedical`: Gene expression, clinical data, genomics, proteomics, epidemiology
- `ai_ml`: ImageNet, benchmark datasets, model evaluation
- `cross_domain`: Paper bridging multiple fields using data from one to inform another

## Data Subtype Inference

Based on the dataset and paper content, infer the `data_subtype`:
- `gene_expression` (RNA-seq, microarray, scRNA-seq)
- `clinical_records` (EHR, MIMIC, claims data)
- `epidemiological` (NHANES, UK Biobank, cohort surveys)
- `imaging` (ImageNet, medical imaging, histopathology)
- `genomic` (TCGA, WGS, GWAS)
- `proteomic` (mass spec, protein arrays)
- `structural` (PDB, AlphaFold predictions)
- `chemical` (ChEMBL, drug screening)
- `multi_omics` (integrated multi-modal data)
- `other`

The data_subtype goes in the output as a top-level field (NOT inside data_anchor).

## Example

**Dataset anchor:** GSE167523 (GEO, gene expression)

**Paper title:** "Dysregulation of 'Don't Eat Me' Signaling-Related Genes in Sepsis"

```json
{
  "paper_type": "original_research",
  "skip": false,
  "domain": "biomedical",

  "data_anchor": {
    "dataset_id": "GSE167523",
    "dataset_type": "geo",
    "role": "primary",
    "what_they_measured": "Whole-blood transcriptomes from 48 sepsis patients and 20 healthy controls via RNA-seq",
    "key_observation": "Seven 'don't eat me' signaling genes (CD47, CD24, SIRPA, SIGLEC10, PD-L1, PD-L2, MHC-I) were differentially expressed, with CD47 and CD24 significantly upregulated in sepsis"
  },

  "prior_state": {
    "field_context": "'Don't eat me' signals are well-studied in cancer immunology as checkpoint mechanisms preventing phagocytosis, but their role in sepsis-associated immune dysregulation is largely unexplored.",
    "conventional_assumption": "Sepsis immune dysfunction is primarily driven by cytokine storm and lymphocyte apoptosis, not by efferocytosis checkpoint pathways.",
    "key_prior_work": [
      {"ref": "Hotchkiss et al. 2013", "doi": "10.1038/nri3552"},
      {"ref": "Chao et al. 2012", "doi": "10.1016/j.cell.2010.12.029"}
    ]
  },

  "reasoning_trace": [
    "Step 1: Transcriptomic analysis of GSE167523 revealed that 7 genes encoding 'don't eat me' signals were differentially expressed in sepsis patients versus controls, with CD47 and CD24 showing the strongest upregulation.",
    "Step 2: These genes are established anti-phagocytic checkpoints in cancer — they prevent immune cells from clearing damaged cells — but had not been systematically profiled in sepsis.",
    "Step 3: The authors reasoned that if sepsis upregulates anti-phagocytic signals, this could explain the impaired clearance of apoptotic cells (defective efferocytosis) that perpetuates organ damage.",
    "Step 4: This led to the hypothesis that 'don't eat me' pathway dysregulation is a distinct mechanism of sepsis immunosuppression, potentially targetable by existing cancer immunotherapy agents."
  ],

  "reasoning_trace_grounded": [
    {"step": "Step 1: Transcriptomic analysis...", "source": "Results, paragraph 1: 'Among the seven DEM genes, CD47 and CD24 showed significant upregulation (logFC > 1.5, FDR < 0.01) in sepsis samples'"},
    {"step": "Step 2: These genes are established...", "source": "Introduction, paragraph 3: 'CD47-SIRPα and CD24-Siglec-10 axes are well-characterized 'don't eat me' signals in tumor immunoevasion'"},
    {"step": "Step 3: The authors reasoned...", "source": "Discussion, paragraph 2: 'Defective efferocytosis in sepsis leads to secondary necrosis and DAMP release, perpetuating the inflammatory cycle'"},
    {"step": "Step 4: This led to the hypothesis...", "source": "Discussion, paragraph 5: 'Repurposing anti-CD47 therapies from oncology to sepsis represents a promising translational avenue'"}
  ],

  "novel_contribution": {
    "spark": "Anti-phagocytic checkpoints drive sepsis immunosuppression",
    "hypothesis": "Upregulation of 'don't eat me' signaling genes in sepsis creates a distinct efferocytosis-deficient state that contributes to organ damage and represents a targetable immunotherapy axis.",
    "approach": "Targeted transcriptomic analysis of 7 'don't eat me' genes in GSE167523, validated with immune cell infiltration analysis and drug-gene interaction mapping."
  },

  "discovery_patterns": ["data_driven", "analogy_transfer"],
  "gap_type": "method",
  "data_subtype": "gene_expression",
  "explicitness": 0.85,
  "explicitness_justification": "Authors explicitly state their data observation, draw the cancer-to-sepsis analogy, and articulate the therapeutic hypothesis.",
  "extraction_confidence": 0.90,
  "confidence_justification": "Clear data-first reasoning with quantitative results, well-articulated cross-domain transfer."
}
```

## Paper Text

**Title:** [TITLE]

**Source:** [SOURCE]

**Dataset:** [DATASET_TYPE]: [DATASET_ID]

### Introduction
[INTRODUCTION]

### Discussion
[DISCUSSION]
