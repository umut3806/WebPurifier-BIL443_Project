# WebPurifier: DOM-Node Classification for Robust Web Content Extraction

**Student Name:** Umut Bayram  
**No:** 221101012  
**Department:** Computer Engineering  
**Email:** [u.bayram@etu.edu.tr](mailto:u.bayram@etu.edu.tr)

---

## Abstract

Web pages contain a mixture of meaningful article content and extraneous noise elements such as advertisements, navigation menus, and footers. Automatically distinguishing these two categories is critical for information retrieval, web scraping, and knowledge extraction pipelines. 

**WebPurifier** is a supervised machine-learning system that classifies individual DOM nodes as either `content` or `noise` using five lightweight structural and lexical features. 

We construct a ground-truth dataset of 54,865 nodes crawled from 118 diverse real-world URLs, with a severe 16.77 : 1 class imbalance. Six classifier families—Decision Tree, Gaussian Naive Bayes, K-Nearest Neighbors, Random Forest, LightGBM, and XGBoost—were trained with and without SMOTE oversampling under 5-fold stratified cross-validation. 

**Result Highlights:** Random Forest without SMOTE achieves the best balance: test F1 = 0.8718, accuracy = 98.59 %, and ROC-AUC = 0.9958, outperforming gradient-boosting alternatives while delivering the highest precision (0.8917). Our results confirm that ensemble tree methods with class-weight compensation are superior to synthetic oversampling for this task.

---

## Features Extracted

WebPurifier extracts the following five key lightweight structural and lexical features to differentiate between node types:
1. **Link Density:** Ratio of anchor text length to total text length in the node.
2. **Text-to-Tag Ratio (TTR):** Ratio of text length to the number of HTML tags present.
3. **Keyword Score:** A score calculated by matching node CSS classes and IDs against common content/noise identifiers (e.g., `article`, `sidebar`, `footer`).
4. **Stop Word Density:** Ratio of stop words to total words inside the node text.
5. **Text Length:** The character length of the normalized text present natively in the node.

---

## Project Structure

- `prepare_dataset/` : Contains scripts to collect, crawl, and form the core DOM-node dataset.
- `figures/` : Hosts various visualization charts (e.g., Confusion Matrices, ROC-AUC comparisons, Feature Importances, Radar Charts) representing our findings.
- `webpurifier_dataset.csv` : The collected ground-truth dataset with 54,865 labeled DOM nodes.
- `training.py` : Script to build, hyper-parameter-tune, and evaluate robust models (with and without SMOTE). Saves model pipelines to `trained_models/`.
- `infer.py` : An inference script that leverages `playwright` to render dynamic websites, apply the extracted trained model, and output purified, deduplicated web content locally.
- `visualize_results.py` : Generates the various graphs and performance comparisons located in the `figures/` directory.

---

## Installation & Setup

1. **Clone the repository and enter the directory:**
   ```bash
   git clone <repository_url>
   cd WebPurifier
   ```

2. **Install Python details from `requirements.txt`:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Playwright Browsers (Required for Inference):**
   ```bash
   playwright install chromium
   ```

---

## Usage Guide

### 1. Training the Models
To train the multi-model architecture, perform Randomized-Search cross validation, and compare different models:

```bash
python training.py
```
This script will output `comparison_results.csv` and multiple serialized pipelines into the `trained_models/` directory. The best pipeline will explicitly be copied as the default inference model.

### 2. Inferring / Purifying a webpage
To run purification on an live unstructured webpage:

```bash
python infer.py
```
You will be prompted to enter a URL. The system will dynamically render the page, extract text nodes, predict `content` vs. `noise` classifiers, and save the aggregated clean text in a dedicated `purified_outputs/` directory.

### 3. Generating Visualization Figures
To regenerate graphs evaluating SMOTE impact, performance metrics, and model benchmarks:
```bash
python visualize_results.py
```
Visualizations will be neatly built and saved in the localized `figures/` directory.
