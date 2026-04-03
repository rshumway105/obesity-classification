# Estimation of Obesity Levels Based on Eating Habits and Physical Condition

> DS5006: Machine Learning for Engineering and Science Applications — Spring 2026, WPI

## Team

- Kathryn Ault
- Gabe Colebourn
- Nina Hernandez
- Robbie Shumway
- Cam Walters

## Overview

This project uses a [UCI Machine Learning Repository dataset](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition) to classify individuals into one of seven obesity levels based on eating habits, physical activity, and demographic features. The dataset contains 2,111 instances (23% real survey data from Mexico, Peru, and Colombia; 77% synthetically generated via SMOTE) with 16 features.

The goal is to identify which lifestyle and physical factors are most predictive of obesity category and to compare multiple classification approaches.

## Repository Structure

```
├── data/
│   ├── raw/                        # Original dataset (not tracked by git)
│   └── processed/                  # Cleaned/encoded data (not tracked)
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 02_preprocessing.ipynb      # Encoding, scaling, train/test split
│   ├── 03_modeling.ipynb           # Model training and comparison
│   └── 04_results.ipynb            # Final results, plots, and tables
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Load and validate the dataset
│   ├── preprocessing.py            # Encoding, scaling, splitting utilities
│   ├── models.py                   # Model definitions and training
│   └── evaluation.py               # Metrics, confusion matrices, CV
├── models/
│   └── saved/                      # Saved model artifacts (not tracked)
├── figures/                        # Exported plots for the report/slides
├── config.py                       # Central project configuration
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/obesity-classification.git
cd obesity-classification

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

Download the dataset from [UCI](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition) and place the Excel file in `data/raw/`:

```
data/raw/ObesityDataSet_raw_and_data_sinthetic.xlsx
```

## Target Variable

The target (`NObeyesdad`) has 7 classes:

| Class | Label |
|-------|-------|
| 1 | Insufficient Weight |
| 2 | Normal Weight |
| 3 | Overweight Level I |
| 4 | Overweight Level II |
| 5 | Obesity Type I |
| 6 | Obesity Type II |
| 7 | Obesity Type III |

## Features (16)

**Continuous:** Age, Height, Weight, FCVC (vegetable consumption), NCP (number of meals), CH2O (water intake), FAF (physical activity), TUE (technology use time)

**Categorical:** Gender, family_history_with_overweight, FAVC (high-calorie food), CAEC (snacking), SMOKE, SCC (calorie monitoring), CALC (alcohol), MTRANS (transportation)

## Methodology

1. **EDA** — Distribution analysis, correlation, class balance assessment
2. **Preprocessing** — One-hot and ordinal encoding of categoricals, feature scaling, stratified train/test split
3. **Modeling** — Compare at minimum:
   - Logistic Regression (baseline)
   - K-Nearest Neighbors
   - Decision Tree / Random Forest
   - Neural Network (stretch goal)
4. **Evaluation** — Accuracy, precision, recall, F1-score (macro & per-class), confusion matrices, hierarchical cross-validation

## Results

_To be completed._

## Key Concerns

- 77% of the data is synthetically generated — results should be interpreted with caution
- Height and Weight may dominate predictions; experiments with and without these features are planned

## Deliverables

| Deliverable | Due Date |
|-------------|----------|
| Project Proposal | April 1, 2026 |
| Presentation Slides (10–20 min) | April 22, 2026 |
| Final Report (10–15 pages) | April 22, 2026 |

## References

- [UCI Dataset](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)
- [CDC Obesity Statistics (2024)](https://www.cdc.gov/nchs/products/databriefs/db508.htm)
