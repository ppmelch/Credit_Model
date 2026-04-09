# Credit Model
### Credit Models project

**Author:** 
  - José Armando Melchor Soto
  - Rolando Fortanell Canedo
  - David Campos Ambriz 


**Course:** Credit Models  
**Institution:** ITESO  

---

## Table of Contents



---
## Overview

---

## Architecture

### Loan Lifecycle

```mermaid
graph LR
    A([Applicant]) --> B[Personal Loan Application]

    subgraph AP [Analysis Phase]
        B --> C{Credit Assessment\nRisk Based Pricing}
        C --> D[Credit History\nCredit Bureau]
        C --> E[Capacity to Pay\nIncome / DTI]
        C --> F[Credit Score\nFICO / VantageScore]
        D --> G{Credit Decision}
        E --> G
        F --> G
    end

    subgraph LL [Loan Lifecycle]
        G -->|Approved| H[Capital Disbursement\nTotal Amount]
        H --> I[Fixed Term Contract\n12 to 84 months]
        I --> J[Fixed Periodic Payments\nPrincipal + Interest + Fees]
        J --> K{Payment Compliance?}
        K -->|Yes| L[Positive Report\nCredit Bureau]
        L --> M([Loan Closure])
        K -->|No| N[Debt Collection Management\nPenalties / Negative Reports]
        N --> O{Recovery?}
        O -->|Yes| J
        O -->|No| P([Loss / Write-off\nLoss Given Default])
    end

    G -->|Rejected| Q([Application Denied])
```


### Project Structure
```mermaid
flowchart TD
    ROOT[Credit_Model/]

    ROOT --> DATA[data/]
    ROOT --> SRC[src/]
    ROOT --> NB[notebooks/]
    ROOT --> DOCS[docs/]
    ROOT --> MAIN[main.py]
    ROOT --> REQ[requirements.txt]
    ROOT --> README[README.md]

    DATA --> dataset[dataset.csv]
    DATA --> results[results.csv]

    SRC --> DATAR[data/]
    SRC --> MODELING[modeling/]
    SRC --> VIZ[visualization/]
    SRC --> UTILS[utils/]
    SRC --> MODELS[models/]

    DATAR --> dp[data_preparation.py]
    DATAR --> ds[data_splitter.py]

    MODELING --> bm[base_model.py]
    MODELING --> bl[business_logic.py]
    MODELING --> cm[classification_model.py]
    MODELING --> cfg[config.py]
    MODELING --> me[model_evaluation.py]
    MODELING --> mod[model.py]
    MODELING --> rc[risk_calculator.py]

    VIZ --> viz[viz.py]

    UTILS --> prints[prints.py]

    MODELS --> rf[random_forest.pkl]

    NB --> notebook[notebook.ipynb]

    DOCS --> pdf[Credit_Model.pdf]

```

### Functional Architecture

### OOP Architecture
```mermaid



```
---


## Installation

```bash
# 1. Clone the repository
git clone https://github.com/ppmelch/Credit_Model.git
cd Credit_Model

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

---

## Results

---

## Discussion

---

## Assumptions

---

## Limitations

---


## Conclusions

---


## Output

---


## Documentation

The full project report is available at:

- [Credit Model Report](docs/Credit_Model.pdf)

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.
