# DemandPilot  
### Personal Educational Exercise on Retail Demand Forecasting, Zero-Sale Classification, and LLM-Oriented Decision Support

---

## Overview

This repository contains a **personal, non-commercial educational exercise** built to study and implement an end-to-end retail demand analytics workflow using the **M5 Forecasting** competition dataset structure.

The project was developed exclusively as a **learning and portfolio exercise** in data analysis, forecasting, machine learning, and LLM-oriented decision support.  
It does **not** constitute a commercial product, a client deliverable, or an official competition submission.

The repository focuses on:

- retail demand exploration
- intermittent demand analysis
- price dynamics analysis
- supervised feature engineering
- zero-sale classification with PyTorch
- next-day unit forecasting
- structured context generation for later LLM use

---

## Important Notice

This repository is published **for personal educational purposes only**.

- It is **not affiliated with**, **endorsed by**, or **sponsored by** Kaggle, Walmart, or the University of Nicosia.
- It is **not** an official M5 competition repository.
- It is **not** intended for commercial use.
- It is **not** intended to redistribute competition data.

All dataset ownership, trademarks, and related intellectual property remain with their respective owners.

---

## Educational Purpose Statement

This project was created as a private personal exercise to learn how to:

1. audit a real retail forecasting dataset  
2. transform raw time-series data into analysis-ready tables  
3. engineer forecasting features  
4. compare baseline forecasting models  
5. build a binary zero-sale classifier using PyTorch  
6. prepare structured outputs that can later be interpreted by a language model  

The goal is **learning, experimentation, and portfolio development only**.

---

## Dataset Origin

This work is based on the **M5 Forecasting Accuracy** competition framework.

**Competition title:** M5 Forecasting Accuracy  
**Hosting platform:** Kaggle  
**Competition sponsor:** University of Nicosia  
**Competition start date:** March 2, 2020  
**Entry deadline:** June 23, 2020  
**Final submission deadline:** June 30, 2020  
**Prize pool:** \$50,000  
**Winner license type:** Open-source  
**Data access and use:** Competition Use, Academic, Non-Commercial Use

The original competition page can be found here:

**https://www.kaggle.com/c/m5-forecasting-accuracy**

This repository does **not** include the original competition data.

---

## Data Usage and Rights

### What is included in this repository
This repository includes:

- Python source code
- project structure
- reproducible analysis pipeline
- model training code
- instructions for local execution

### What is intentionally NOT included
This repository does **not** include:

- raw competition CSV files
- redistributed Kaggle data
- private competition data
- manually shared competition datasets
- proprietary source datasets

### Why the data is excluded
According to the competition terms, the competition data is subject to **competition use, academic use, and non-commercial use**, and redistribution is restricted.  
For that reason, this repository is designed so that users must **manually obtain the dataset from the official source** and place it locally in the expected folder structure.

---

## Copyright and Ownership

All rights to the original competition dataset, competition materials, trademarks, and source competition assets belong to their respective owners, including but not limited to:

- Kaggle
- University of Nicosia
- Walmart
- any other original rights holders associated with the competition

This repository claims **no ownership** over the original dataset. Any output, .csv and file made by the them OR made by me, using their data and my .py, won't be uploaded.

This repository only contains **original personal code and documentation** created as part of a self-study exercise.

---

## Non-Commercial Disclaimer

This repository is shared strictly as:

- a personal exercise
- a study project
- a technical learning artifact
- a portfolio demonstration of methods and workflow

It is **not** intended for:

- commercial deployment
- resale
- commercial analytics services
- redistribution of restricted data
- competition misuse
- unauthorized publication of original competition assets

---

## No Warranty

This project is provided **as is**, without warranty of any kind.

The code, analysis, and outputs are intended only for educational exploration.  
No guarantee is made regarding:

- accuracy
- completeness
- fitness for a particular purpose
- regulatory compliance
- production-readiness
- business suitability

Any use of this repository is at the user's own risk.

---

## Project Objective

The project studies a retail forecasting pipeline with three main goals:

### 1. Understand the data
- dataset audit
- structure validation
- missing-value and consistency checks
- store/category/state analysis

### 2. Predict future demand
- zero-sale classification
- next-day demand forecasting
- comparison of naive baselines vs machine learning

### 3. Support interpretation
- transform model outputs into structured business context
- prepare machine-readable summaries for later LLM-based explanation

---

## Core Questions Addressed

This exercise was built around the following practical questions:

- Which retail series are highly intermittent?
- Which products are more likely to record zero sales tomorrow?
- Which forecasting approach performs better than simple baselines?
- How can model outputs be translated into business-facing summaries?
- How can a structured retail forecasting workflow be made reproducible?

---

## Repository Structure

```text
DemandPilot/
├── Data/
├── outputs/
│   ├── figures/
│   ├── llm/
│   ├── models/
│   └── tables/
├── 00_environment_check.py
├── 01_data_audit.py
├── 02_build_analysis_tables.py
├── 03_detailed_eda.py
├── 04_intermittency_analysis.py
├── 05_price_dynamics.py
├── 06_build_model_dataset.py
├── 07_zero_sale_classifier_pytorch.py
├── 08_baseline_forecast.py
├── 09_llm_context_builder.py
├── 10_top_up_down_latest_day.py
├── project_utils.py
├── requirements.txt
└── README.md
