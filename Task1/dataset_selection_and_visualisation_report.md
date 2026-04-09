# Task 1 — Dataset Selection and Visualisation (Healthcare)

## 1) Aim
This task demonstrates how an open healthcare dataset can be selected, imported, explored, and visualised to support evidence-based decision making in health analytics.

## 2) Open dataset search and shortlist
I reviewed multiple open-data sources that are commonly used in healthcare analytics:

1. **World Health Organization (WHO) Global Health Observatory**  
   https://www.who.int/data/gho
2. **Data.gov.uk**  
   https://data.gov.uk/
3. **Office for National Statistics (ONS)**  
   https://www.ons.gov.uk/
4. **Google Dataset Search**  
   https://datasetsearch.research.google.com/

### Candidate datasets considered
- **WHO/GHO population-level indicators** (strong for international trend analysis, but less suitable for patient-level risk-factor visualisation).
- **ONS/Data.gov.uk public health indicators** (excellent for UK regional comparisons, but many are aggregated and less useful for individual-level risk modelling visuals).
- **Stroke prediction healthcare dataset (patient-level tabular data)** already available in this repository (`healthcare-dataset-stroke-data.csv`), with demographic and clinical risk-factor fields.

## 3) Final dataset selected
**Selected dataset:** `healthcare-dataset-stroke-data.csv`.

### Why this dataset was selected
- It is **healthcare-relevant** and directly tied to a high-impact clinical event (stroke).
- It has a **patient-level structure**, enabling richer visualisation than aggregate-only indicators.
- It includes common screening variables used in epidemiology and preventive care:
  - age
  - hypertension
  - heart disease
  - BMI
  - average glucose level
  - smoking status
- It is suitable for a progression from visual analytics to future predictive modelling.

## 4) Visualisation tooling and justification
Because this environment does not have external plotting libraries installed, visualisation was implemented using:

- **Python standard library (`csv`, `statistics`, `collections`)** for ingestion and manipulation.
- **Programmatically generated SVG charts** for reproducible visual outputs.

### Why this tool choice is valid and relevant
- **Reproducibility:** scripts can be re-run on new data.
- **Portability:** no third-party dependency requirement.
- **Transparency:** transformation and plotting logic are explicit in code.
- **Healthcare relevance:** clinical analysts often require auditable pipelines; script-based steps support this requirement.

## 5) Visualisation process
The workflow implemented in `Task1/visualise_stroke.py`:

1. **Import data** from CSV.
2. **Compute key descriptive statistics**:
   - total rows, columns
   - stroke prevalence
   - mean age by stroke class
   - missing BMI count
3. **Create charts**:
   - `bar_hypertension_stroke.svg`: stroke counts split by hypertension status.
   - `hist_age.svg`: age distribution in 10-year bins.
   - `scatter_age_glucose_stroke.svg`: age vs average glucose, coloured by stroke outcome.
4. **Export summary text** to `summary.txt`.

## 6) Key findings from visual outputs
From the generated summaries and charts:
- Stroke prevalence is low relative to the total sample (class imbalance).
- Average age is substantially higher in the stroke-positive group.
- Hypertension-positive patients show a higher stroke count profile than hypertension-negative patients.
- Age and glucose scatter shows concentration differences between stroke/non-stroke groups, useful for hypothesis generation.

## 7) Relevance to healthcare sector
These visualisations align with healthcare analytics use cases:
- **Risk stratification:** identify high-risk groups by age/comorbidities.
- **Screening prioritisation:** support targeted preventive interventions.
- **Public health communication:** convert complex tabular data into interpretable graphics.
- **Model-readiness checks:** reveal class imbalance and feature distributions before machine learning.

## 8) How to run
From the repository root:

```bash
python Task1/visualise_stroke.py
```

Outputs are written to:

- `Task1/outputs/summary.txt`
- `Task1/outputs/bar_hypertension_stroke.svg`
- `Task1/outputs/hist_age.svg`
- `Task1/outputs/scatter_age_glucose_stroke.svg`

