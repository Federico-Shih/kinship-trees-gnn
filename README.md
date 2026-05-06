# Two step link prediction

This repository contains a suite of tools and notebooks designed for relationship prediction and kinship analysis using Deep Graph Infomax (DGI) and Heterogeneous Graph Transformers (HGT).

---

### 📂 Notebooks Overview

The project is structured around several key Jupyter notebooks, each handling a specific stage of the machine learning pipeline:

* **`main.ipynb`**
    The primary entry point for the project. Likely used for orchestrating the overall workflow, integrating data processing, and high-level testing.
* **`hgt_model.ipynb`**
    Contains the architecture and training logic for the **Heterogeneous Graph Transformer (HGT)**. This model is specifically designed to handle graph data with different types of nodes and edges (e.g., different family roles).
* **`dgi_model.ipynb`**
    Focuses on the **Deep Graph Infomax (DGI)** implementation. This is typically used for unsupervised representation learning to generate robust node embeddings from the kinship graph.
* **`Relationship_Prediction_Training (1).ipynb`**
    Main two step training code, containing GaitingGIN model for link prediction on family trees and FamilyRelationGNN for link classification.
---

### 🛠 Project Structure

* **`data/`**: Directory containing raw and processed kinship datasets.
* **`checkpoints/` & `runs/`**: Storage for model weights and TensorBoard logs during training.
* **`fiw_processor.py`**: A helper script for processing the Families in the Wild (FIW) dataset.
* **`.pt` files**: Pre-trained model weights and saved embeddings for the HGT, DGI, and Gating models.
* **`requirements.txt`**: List of Python dependencies required to run the notebooks.

---

### 🚀 Getting Started

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Ensure your data is placed in the `data/` directory.
3.  Start with `main.ipynb` or `Relationship_Prediction_Training.ipynb` to explore the existing models and embeddings.