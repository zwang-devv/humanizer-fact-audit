# Humanized but Still True? Auditing Facts and Detection in AI-Text Humanizers

Code and per-item results for the paper *Humanized but Still True? Auditing Facts and Detection in AI-Text Humanizers*
(Ziyuan Wang, Genliang Zhu, Xiaobo Chen, 2026). arXiv link: to be added.

This repository contains:

- `factalign.py`: the rule-based checker that extracts checkable units from a source text (numbers and dates,
  person and organization names, quoted spans of at least three words) and decides whether a rewrite keeps each one.
- `data/`: the per-item unit judgments and detector scores behind Tables 1 and 2 of the paper.
- `reproduce_tables.py`: recomputes Table 1, the H2a correlation and Table 2 (including the article-level
  bootstrap intervals) from `data/`.

Source texts, AI drafts and rewrites are not redistributed. Public inputs can be retrieved from their original
releases (see *Sources*); the SHA-256 hashes in `data/passages.csv` identify the exact texts.

## Quick start

```bash
pip install -r requirements.txt
python reproduce_tables.py          # needs numpy only
```

Use the checker on your own pairs:

```python
from factalign import align
units = align(source_text, rewritten_text)
altered = [u for u in units if not u["kept"]]
```

or from the command line, with one JSON object per line containing `source` and `rewrite`:

```bash
python factalign.py --in pairs.jsonl --out units.jsonl
```

The checker is English-only and uses spaCy `en_core_web_sm` 3.8.0 for names. Numbers and dates are compared by value,
names by surname or organization variant, and quoted spans verbatim after ignoring case, punctuation and whitespace.
See Section 3.2 of the paper for the rules and their validation.

## Data

`data/fact_units.jsonl`, one line per (tool, document), 731 lines:

| field | meaning |
| --- | --- |
| `set` | `commercial` (AI Humanizer Benchmark outputs) or `open-12b` (open 12B humanizer on federal texts) |
| `tool`, `aihb_cycle`, `sample_id`, `category` | commercial rows: tool name, benchmark cycle (`2026-10`), sample and category in the benchmark release |
| `aihb_mean_human_prob` | commercial rows: mean human probability over the benchmark's seven commercial detectors (the evasion score) |
| `model_file`, `passage_id`, `corpus` | open-12b rows: GGUF file (`bf16`, `Q8_0`, `Q3-QAT`, `IQ2_XS-QAT`), passage and corpus |
| `units` | list of `{group, surface, kept}`; `group` is `num`, `ent` or `quote` |

`data/passages.csv`, the 92 federal passages: `passage_id`, `corpus` (`gohumanize-test` or `new2026`),
`article_url`, `draft_generator`, and SHA-256 hashes of the human original and the AI draft (UTF-8).

`data/detection_scores.csv`, 552 rows (92 passages x human original, AI draft and four 12B rewrites):

| column | direction | detector |
| --- | --- | --- |
| `binoculars` | lower = more AI-like | Binoculars, observer `tiiuae/falcon-7b@ec89142`, performer `tiiuae/falcon-7b-instruct@8782b5c` |
| `fast_detectgpt_crit` | higher = more AI-like | Fast-DetectGPT criterion, same model pair |
| `roberta_p_fake` | higher = more AI-like | `openai-community/roberta-base-openai-detector@6cba99c` |
| `radar_p_ai` | higher = more AI-like | `TrustSafeAI/RADAR-Vicuna-7B@4ff1f23` |
| `jev_p_ai` | higher = more AI-like | Jev `typesafe/jev-1.13`, snapshot 20260917, one zero-shot choice question (Appendix A of the paper) |

The 12B rewrites were produced with `jialinyyzz/humanizer` revision `2e57b65` and llama.cpp `b11865`.

## Sources and licenses

- AI Humanizer Benchmark, revision `6306b2043736dd958cf39ee1d9a64956c54dd78a`, cycle 2026-10, CC BY 4.0:
  https://huggingface.co/datasets/ai-humanizer-benchmark/ai-humanizer-benchmark
- GoHumanize Open Humanizer dataset, revision `cc066df21227f5fdb73b513c330181f946ce82ef`, CC BY 4.0
  (human texts are U.S. federal public-domain prose): https://doi.org/10.5281/zenodo.22843083
- NEW2026 human passages are public-domain U.S. federal agency news releases; see `article_url`.

Code (`factalign.py`, `reproduce_tables.py`) is released under the MIT License (`LICENSE`).
The files in `data/` are released under CC BY 4.0 and include material derived from the CC BY 4.0 sources above;
please credit those sources as well when you reuse them.

## Citation

```bibtex
@misc{wang2026humanized,
  title  = {Humanized but Still True? Auditing Facts and Detection in AI-Text Humanizers},
  author = {Wang, Ziyuan and Zhu, Genliang and Chen, Xiaobo},
  year   = {2026},
  note   = {arXiv preprint}
}
```
