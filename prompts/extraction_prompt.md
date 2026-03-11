# Structured Field Extraction from Research Paper

You are a research paper analysis assistant. Extract structured information from the following paper text (methods and results sections).

## Extract These Fields

Return a JSON object with these keys:

```json
{
  "methodology": "Brief description of the study design and statistical methods used (1-2 sentences)",
  "datasets_used": "Names of datasets, databases, or data sources used (comma-separated)",
  "sample_size": "Total sample size or number of observations (e.g., 'N=12,934 ICU stays')",
  "hardware": "Compute hardware mentioned, if any (e.g., 'NVIDIA A100', '8-core CPU'). null if not mentioned.",
  "key_findings": "2-3 key quantitative findings with effect sizes or performance metrics",
  "limitations": "Main limitations acknowledged by the authors (1-2 sentences)"
}
```

## Rules

- Extract only what is explicitly stated in the text. Do not infer or hallucinate.
- For sample_size, report the primary analysis sample, not subgroup sizes.
- For key_findings, prioritize quantitative results (ORs, AUROCs, p-values, CIs).
- If a field is not present in the text, use null.
- Output ONLY valid JSON. No commentary.

## Paper Text

[TEXT]
