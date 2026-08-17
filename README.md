<div align="center">

# When Scaffold Design Outweighs Model Adaptation

### A Controlled Quality–Cost Comparison of Socratic AI Math Tutors

[![ICCE 2026](https://img.shields.io/badge/ICCE-2026-1f4e79?style=flat-square)](https://icce2026.org)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-2F5233?style=flat-square)](LICENSE)
[![Reproduction](https://img.shields.io/badge/Artifacts-Prompts%20·%20Configs%20·%20Scripts-8B7355?style=flat-square)](#repository-layout)

*ICCE 2026 · Submission 206 · Camera-ready reproduction package*

[Design](#design-at-a-glance) · [Layout](#repository-layout) · [Quick start](#quick-start) · [Pipeline](#reproducing-the-pipeline) · [Config locks](#key-configuration-locks) · [Metrics](#evaluation-dimensions) · [Data](#data-provenance-not-redistributed) · [Citation](#citation)

</div>

<br/>

This repository releases the **executable experimental stack** behind a controlled comparison of four Socratic math-tutor deployment choices. The study holds the dialogue protocol, student simulator, problem set, and evaluation pipeline fixed, and varies only the adaptation / serving decision.

| Condition | Model | Adaptation | Serving |
|:---|:---|:---|:---|
| **C1** PE-Llama | Llama 3.1–8B-Instruct | Prompt only | Local |
| **C2** SFT-Llama | Llama 3.1–8B-Instruct | LoRA SFT | Local |
| **C3** DPO-Llama | Llama 3.1–8B-Instruct | LoRA DPO ← SFT | Local |
| **C4** PE-Qwen | Qwen3-Max | Prompt only | API |

Scaffolding is ablated separately (**No / Weak / Full**) under the same PE-Llama backbone.

---

## Design at a glance

```text
                    ┌─────────────────────────────────────────┐
                    │         Held constant across C1–C4      │
                    │  problems · simulator · Socratic prompt │
                    │  max turns · judges · composite weights │
                    └───────────────────┬─────────────────────┘
                                        │
          ┌──────────────┬──────────────┼──────────────┬──────────────┐
          ▼              ▼              ▼              ▼              │
       C1 PE          C2 SFT         C3 DPO         C4 API            │
     (prompt)        (LoRA)       (pref. opt.)     (Qwen)             │
          └──────────────┴──────────────┴──────────────┘              │
                                        │                             │
                                        ▼                             │
                         six-dimension scoring → composite            │
                         QQ · SD · SLR · EDT · DC · MCVerified        │
```

Composite weights (study lock): **QQ 20% · SD 20% · SLR 25% · EDT 15% · DC 10% · MCVerified 10%**.

---

## Repository layout

```text
.
├── configs/                 # models, splits, gates, condition YAMLs, prompts
│   ├── conditions/          # C1–C4 + generation / simulator / evaluators
│   └── prompts/             # tutor scaffolds · simulator · judge prompts
├── rubrics/                 # scoring weights (YAML)
├── src/
│   ├── data/                # discovery, isolation, schema mapping
│   ├── training/            # SFT LoRA · DPO · prompt inference
│   ├── simulation/          # student simulator train / run
│   ├── generation/          # four-condition dialogue generation
│   ├── evaluation/          # six metrics · composite · judges · anchors
│   ├── analysis/            # RQ1–RQ5 contrasts, cost, turn, reliability
│   ├── reporting/           # table export · reproducibility manifests
│   └── utils/
├── scripts/                 # ordered pipeline entry points (run_00 …)
├── cost_model/              # parameterised break-even calculator
├── results/                 # machine-readable aggregate tables (CSV)
├── requirements.txt
└── LICENSE
```

No manuscript sources, Word documents, or local model weights are redistributed. Upstream datasets must be obtained from their official releases; rebuild splits with the data scripts below.

---

## Quick start

```bash
git clone https://github.com/XianghuiMeng-1020/socratic-math-tutor-scaffold-comparison.git
cd socratic-math-tutor-scaffold-comparison

python -m venv .venv
# Windows
.venv\Scripts\activate
# Unix
# source .venv/bin/activate

pip install -r requirements.txt
```

Environment variables:

| Variable | Used for |
|:---|:---|
| `PROJECT_ROOT` | Absolute path to this repository (overrides hardcoded defaults) |
| `QWEN_API_KEY` | C4 inference + optional judge calls (DashScope-compatible) |
| `OPENAI_API_KEY` | Optional multi-judge ensemble |
| `HF_TOKEN` | Only if using the gated Meta Llama repo instead of the Unsloth mirror |

```bash
# PowerShell
$env:PROJECT_ROOT = (Get-Location).Path
$env:QWEN_API_KEY = "..."
```

---

## Reproducing the pipeline

Scripts are ordered. Dry-runs skip GPU / API loads where supported.

```bash
python scripts/run_00_data_audit.py
python scripts/run_01_split_and_isolation.py
python scripts/run_02_train_sft.py
python scripts/run_03_build_dpo_pairs.py
python scripts/run_04_train_dpo.py
python scripts/run_05_train_simulator.py
python scripts/run_06_generate_dialogues.py
python scripts/run_07_evaluate_metrics.py
python scripts/run_08_run_analyses.py
python scripts/run_09_build_reports.py
python scripts/run_10_all_gates.py

# Ablation / reliability / figures
python scripts/run_11_multi_judge_scoring.py
python scripts/run_12_scaffold_ablation.py
python scripts/run_14_simulator_validation.py
python scripts/run_15_per_profile_analysis.py
```

Cost model (no GPU):

```bash
python cost_model/cost_calculation.py
```

Precomputed aggregate CSVs used in the camera-ready audit live under [`results/`](results/).

---

## Key configuration locks

| Item | Value |
|:---|:---|
| Base LM (C1–C3) | `unsloth/Meta-Llama-3.1-8B-Instruct` |
| LoRA | rank 16 · α 32 · dropout 0.05 · `{q,k,v,o}_proj` |
| SFT | 3 epochs · lr `2e-4` · cosine · bs 1 · accum 32 · 4-bit |
| DPO | β `0.1` · 1 epoch · lr `5e-5` · init from SFT · 970 retained pairs |
| Decoding | temperature `0.0` · max tokens 1024 · max turns 10 |
| C4 API | `qwen3-max` · DashScope compatible mode |
| Seed policy | `fixed_per_problem_profile_turn` |

Full detail: [`configs/models.yaml`](configs/models.yaml), [`configs/conditions/`](configs/conditions/).

---

## Evaluation dimensions

| Code | Construct | Weight |
|:---:|:---|---:|
| QQ | Question quality | 0.20 |
| SD | Scaffolding depth | 0.20 |
| SLR | Solution-leak resistance (higher = better) | 0.25 |
| EDT | Error-diagnosis targeting | 0.15 |
| DC | Dialogue coherence | 0.10 |
| MCVerified | Symbolic / numeric verification | 0.10 |

Judge prompts: [`configs/prompts/judge_*.txt`](configs/prompts/).  
Weights: [`rubrics/scoring_weights.yaml`](rubrics/scoring_weights.yaml).

> **MCVerified note.** When no verifiable numeric answer can be extracted, the metric returns `0.5` as a *non-identifiable default*, not as verified correctness.

---

## Data provenance (not redistributed)

| Role | Upstream source | Use in this study |
|:---|:---|:---|
| SocraTeach-style dialogues | SocraticLM | SFT supervision |
| Tutoring dialogues | MathDial | DPO pairs · simulator · held-out evaluation |
| Auxiliary | MATH / MathTutorBench | Filtering & validation support |

Provide official dataset paths under `data/` (see `configs/paths.yaml`), then run the split / isolation scripts. Only **problem-ID lists** and **derived aggregate tables** are included here.

---

## Citation

If these artifacts are useful, please cite the ICCE 2026 paper (camera-ready version):

```bibtex
@inproceedings{icce2026socraticscaffold,
  title     = {When Scaffold Design Outweighs Model Adaptation:
               A Controlled Quality--Cost Comparison of Socratic AI Math Tutors},
  author    = {{ICCE 2026 Submission 206}},
  booktitle = {Proceedings of the International Conference on Computers in Education},
  year      = {2026},
  note      = {Camera-ready}
}
```

---

## License & third-party terms

- Code in this repository: [MIT](LICENSE)
- Llama, Qwen, and upstream datasets remain under their original licenses
- Do **not** re-upload gated checkpoints or restricted raw corpora via this project

---

<div align="center">

<sub>
Reproduction tag <code>icce2026-paper206-camera-ready-0.1</code>
· maintained for camera-ready auditability
</sub>

</div>
