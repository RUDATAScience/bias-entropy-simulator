# Informational Health Validation: Entropy Decay and the Pathology of Data Preprocessing

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository is a Python simulation that verifies the irreversible destructive process that "social desirability bias" (peer pressure and conformity) inflicts on information systems during surveys and large-scale data collection.

It proves a fatal paradox: outlier removal (data cleansing), which is considered "standard preprocessing" in modern machine learning and data science, actually accelerates "informational corruption" under biased conditions.

---

## 📌 Background

In the field of data analysis, extreme minority opinions (e.g., Rating 1) are generally removed as "noise" or "outliers" before model training. However, when "peer pressure (conformity)" influences respondents' decision-making, this procedure leads to fatal consequences.

This simulation quantitatively proves the following three facts:

1. **Phase Diagram of Signal Disappearance**: Identifies the boundary where true signals evaporate in a "phase transition-like" manner due to the interference between respondents' confidence and conformity.
2. **Informational Heat Death**: Demonstrates the process by which systemic diversity (Shannon entropy) rapidly decays and becomes rigid as conformity increases.
3. **The Pathology of Data Preprocessing**: Exposes the structural flaw where "outlier removal (e.g., the 2–3 sigma rule)", executed with good intentions by practitioners, systematically obliterates the only remaining warning signals and pulls the entire dataset further toward a "false consensus."

---

## 🧮 Mathematical Model

An individual's final utility ($U_{\text{total}}$) is defined as a linear combination of their intrinsic "true intention" ($U_{\text{true}}$) and extrinsic "conformity" ($U_{\text{target}}$). The selection probability is then calculated via a Softmax function.

$$U_{\text{total}} = (1 - v_2) U_{\text{true}} + v_2 U_{\text{target}}$$

* **$v_2$**: The weight of social desirability (conformity). 0 represents complete honesty, while 1 represents complete conformity.
* **$\beta$**: The respondent's confidence level, which controls the sensitivity of the Softmax function.

---

## 📊 Outputs

Executing the script creates a `rigorous_validation_results` directory, generating the following high-resolution graphs (PNG) and raw data (CSV).

* **Fig A: Phase Diagram of Signal Evaporation (Heatmap)**: A phase diagram visualizing the "cliff" boundary where warning signals evaporate within the two-dimensional parameter space of confidence ($\beta$) and conformity ($v_2$).
* **Fig B: Information Entropy Decay**: A transition graph showing how "Shannon entropy," representing data diversity, rapidly decays and homogenizes the system as conformity increases.
* **Fig C: How 'Data Cleaning' Amplifies Bias**: A comparison between the raw data and the filtered data after statistical outlier removal. This proves the paradox where data cleansing pushes the mean further away from the truth, thereby amplifying the bias.

---

## 🚀 Usage

This code is optimized for execution in **Google Colaboratory**.

1. Upload `validation_sim.py` (or the Jupyter Notebook format) to Google Colab.
2. Run all cells.
3. Upon completion, a `rigorous_validation_archive.zip` containing the graphs and CSVs will be downloaded automatically.

### Note on Local Environment Execution
When running in a local Python environment (VSCode, JupyterLab, etc.), please disable Colab-specific modules. By deleting or commenting out the `from google.colab import files` line at the beginning and the `files.download(...)` line at the end of the script, the ZIP file will be generated in your current working directory.

```bash
# Install dependencies
pip install -r requirements.txt

# Run the script
python validation_sim.py
