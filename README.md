# Chemical Attention Analysis

Code for the paper "Analysis of Attention Mechanisms in Chemical Language Models" submitted to JCIM.

## Dataset Access

The required data files are too large for GitHub and are stored on [Google Drive](https://drive.google.com/drive/folders/1xH1y8HN2OwwptN5jqSKX_1Mbhfw1YOQp?usp=sharing). 

Download these files and place them in the project directory:
- `all_smiles` - Complete SMILES dataset
- `smiles_attention_data.pkl.gz` - Compressed attention data  
- `smiles with data` - Additional SMILES metadata
- `map_chiral_index.pkl` - Chiral center mappings

## Quick Start

```bash
pip install torch rdkit-pypi matplotlib numpy tqdm
```

```python
from google.colab import drive
drive.mount('/content/drive')

# Load your data from Google Drive path
# Run the analysis notebook
```

The notebook analyzes attention patterns for chemical features including numeric tokens, ring structures, and chiral centers.
