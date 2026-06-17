# AURA: Autonomous Vector-Space Auditing Framework


https://github.com/user-attachments/assets/71b4fe97-0ec8-46ad-aa53-18e505586973







AURA is a local privacy auditing utility designed to stress-test vector embedding data streams against information leakage vulnerabilities, aligning with the OWASP LLM08 (Embedding Weaknesses) security standard.

Vector embeddings are often treated as secure, one-way transformations. AURA demonstrates that embedding vectors retain strong semantic coordinates, allowing an attacker to reconstruct raw plaintext strings. The framework simulates a realistic black-box surrogate attack, applies Gaussian noise perturbations as a defense strategy, and charts the resulting privacy-utility frontier directly in the terminal.

---

## Core Features

* **Black-Box Surrogate Inversion:** Uses continuous relaxation optimization via PyTorch to map raw embedding coordinates back to a vocabulary grid using an independently misaligned shadow model.
* **Synchronized Telemetry Tracking:** Aligns optimization step metrics directly with the true clean data vector, ensuring that running logs and final report matrix values are mathematically consistent.
* **Automated Multi-Trial Noise Sweep:** Runs sequential Monte Carlo trials across varying noise thresholds to calculate mean token recovery rates and standard deviations.
* **Terminal Frontier Visualization:** Generates a text-based ASCII scatter plot mapping Token Recovery against RAG Search Utility to pinpoint the optimal configuration sweet spot.

---

## Live In-Action Walkthrough

When initializing the central orchestration pipeline, the system captures user text, expands the vocabulary matrix on disk, runs the dual inversion phases, and benchmarks the privacy curve.

```text
[AURA SETUP] Ready for comprehensive privacy analysis.
📝 Enter a 3 or 4 word secret string to test: my api token

[AUDIT] Launching Baseline Black-Box Attacker Optimization Run...
 Step 000 │ Loss: 0.9658 │ Cos Sim: 0.0342 │ Current Target Vector Estimate: token_72 admin var_129
 Step 025 │ Loss: 0.0076 │ Cos Sim: 0.9924 │ Current Target Vector Estimate: token_8 node_49 node_44
 Step 050 │ Loss: 0.0009 │ Cos Sim: 0.9991 │ Current Target Vector Estimate: my api token_69
 Step 100 │ Loss: 0.0000 │ Cos Sim: 1.0000 │ Current Target Vector Estimate: my api token_69

[AUDIT] Launching Defended Run (Noise Sigma = 0.15)...
 Step 000 │ Loss: 1.0126 │ Cos Sim: 0.0332 │ Current Target Vector Estimate: var_49 node_67 encryption
 Step 025 │ Loss: 0.0105 │ Cos Sim: 0.5867 │ Current Target Vector Estimate: node_67 token_18 var_72
 Step 050 │ Loss: 0.0009 │ Cos Sim: 0.5851 │ Current Target Vector Estimate: var_106 token_18 token_134
 Step 100 │ Loss: 0.0000 │ Cos Sim: 0.5844 │ Current Target Vector Estimate: var_106 token_18 token_134

================================================================================
📊 RUNNING MULTI-METRIC NOISE SWEEP FRONTIER
================================================================================
                                       PRIVACY VS UTILITY DEGRADATION MATRIX
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃             ┃ Raw Embedding Similarity ┃ Token Recovery Rate ┃ Discrete Token Similarity ┃ RAG Utility Retention ┃
┃ Noise Sigma ┃   (Published vs Clean)   ┃    (Mean ± Std)     ┃    (Decoded vs Clean)     ┃   (Search Match %)    ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│    0.00     │      100.0% Cos Sim      │     53.3% ± 16.3%   │       55.4% Cos Sim       │      100.0% Match     │
│    0.02     │       97.6% Cos Sim      │     53.3% ± 16.3%   │       56.7% Cos Sim       │       73.3% Match     │
│    0.05     │       87.4% Cos Sim      │     23.3% ± 21.3%   │       26.3% Cos Sim       │       60.0% Match     │
│    0.10     │       67.1% Cos Sim      │      3.3% ± 10.0%   │        5.5% Cos Sim       │       43.3% Match     │
│    0.15     │       51.9% Cos Sim      │      0.0% ± 0.0%    │        0.5% Cos Sim       │       36.7% Match     │
│    0.20     │       41.7% Cos Sim      │      0.0% ± 0.0%    │        0.1% Cos Sim       │       36.7% Match     │
└─────────────┴──────────────────────────┴─────────────────────┴───────────────────────────┴───────────────────────┘

================================================================================
📈 PRIVACY-UTILITY FRONTIER CURVE (ASCII GRAPH)
   [R] = Token Recovery Rate (%)  │  [U] = RAG Search Utility (%)
================================================================================
100% │  [U]   ·    ·    ·    ·    ·
 90% │   ·    ·    ·    ·    ·    ·
 80% │   ·    ·    ·    ·    ·    ·
 70% │   ·   [U]   ·    ·    ·    ·
 60% │   ·    ·   [U]   ·    ·    ·
 50% │  [R]  [R]   ·    ·    ·    ·
 40% │   ·    ·    ·   [U]  [U]  [U]
 30% │   ·    ·    ·    ·    ·    ·
 20% │   ·    ·   [R]   ·    ·    ·
 10% │   ·    ·    ·    ·    ·    ·
  0% │   ·    ·    ·   [R]  [R]  [R]
     └──────────────────────────────
       0.00  0.02  0.05  0.10  0.15  0.20  (Noise Sigma)
```

---

## Directory Blueprint

```text
├── .github/workflows/   # CI automation workflows
├── core/                # Mathematical framework modules
│   ├── defense.py       # Gaussian perturbation logic
│   ├── inversion.py     # Continuous relaxation gradient descent engine
│   └── metrics.py       # Validation trackers & search utility functions
├── data/                # Vocabulary allocation tables
│   └── vocabulary.txt   # Local token map text file
├── utils/               # Display and UI formatters
│   └── display.py       # Console layout controllers
├── aura_cli.py          # Unified framework orchestration entry point
├── pyproject.toml       # Modern Python build/metadata package declaration
└── REQUIREMENTS.txt     # Locked deployment software dependencies
```

---

## Installation & Local Execution

### Prerequisites
* Python 3.10 or higher
* Pip package manager

### 1. Setup Environment
Clone the repository workspace and install the required dependencies:
```bash
pip install -r REQUIREMENTS.txt
```

### 2. Execute Audit
Initialize the localized command-line auditing workflow:
```bash
python aura_cli.py
```
