# Which Chemical Concepts Are Captured in ChemBERTa's Attention?

This repository contains the code accompanying the manuscript:

**"Which Chemical Concepts Are Captured in ChemBERTa's Attention?"**

The project investigates how chemical concepts, functional groups, and structural information are represented within the attention mechanisms of the pretrained ChemBERTa-77M-MLM model.

The repository includes implementations of:

- **AMPC-Chem-FG**: Analysis of attention-head specialization for chemical concepts and functional groups.
- **AMPC-Struct**: Analysis of attention-head alignment with molecular structural information represented by Coulomb matrices.
- **Downstream prediction experiments** using representations constructed from AMPC-identified attention heads.

---

## Repository Contents

### `Attention_Analysis.ipynb`

Main analysis notebook containing:

- Chemical concept analysis (CRE, CRA, CB, Ch)
- Functional-group analysis
- Coulomb-matrix-based structural analysis
- Conformational robustness analysis
- Visualization of attention-head specialization

### `AMPC experiments.py`

Implementation of the downstream prediction experiments reported in the manuscript.

The script compares four representation strategies:

1. **AMPC heads**
2. **random_no_AMPC heads**
3. **frozen mean pooling**
4. **fine-tuned CLS token**

for:

- BACE bioactivity classification
- LogP regression

### `data/raw/bace.csv`

Dataset used for the downstream prediction experiments.

---

## Additional Data

The attention-analysis experiments require precomputed resources that are not included in this repository due to size limitations.

These files can be downloaded from:

[Google Drive](https://drive.google.com/drive/folders/1xH1y8HN2OwwptN5jqSKX_1Mbhfw1YOQp?usp=sharing)

Required files:

- `all_smiles`
- `smiles_attention_data.pkl.gz`
- `smiles with data`
- `map_chiral_index.pkl`

Place the downloaded files in the appropriate working directory before running the notebook.

---

## Installation

```bash
pip install torch transformers rdkit-pypi deepchem scikit-learn pandas numpy tqdm matplotlib imbalanced-learn
```

---

## Running the Attention Analysis

Open and execute:

```text
Attention_Analysis.ipynb
```

The notebook reproduces the AMPC-Chem-FG and AMPC-Struct analyses described in the manuscript.

---

## Running the Downstream Prediction Experiments

Place `bace.csv` inside:

```text
data/raw/
```

and run:

```bash
python "AMPC experiments.py"
```

The script automatically performs:

- BACE classification
- LogP regression

and compares:

- AMPC heads
- random_no_AMPC heads
- frozen mean pooling
- fine-tuned CLS token

The resulting performance tables are saved as:

```text
bace_results.csv
logp_results.csv
```

---

## Model

All experiments use:

**DeepChem/ChemBERTa-77M-MLM**

available through the Hugging Face ecosystem.

---

## Citation

If you use this repository in your research, please cite the associated manuscript.
