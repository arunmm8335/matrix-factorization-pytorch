# Recommender Systems Lab (PyTorch)

This repository contains a progression of Recommender System implementations using PyTorch, moving from classic Matrix Factorization to modern Deep Learning and Transformer-based architectures.

The goal of this project was to reproduce classic algorithms, benchmark their performance on the **MovieLens Small** dataset, and critique their limitations in a "Research Engineering" context.

---

##  Project 1: Matrix Factorization (The Baseline)
**Location:** `src/mf/`

An implementation of the SVD++ algorithm described in the classic paper **"Matrix Factorization Techniques for Recommender Systems"** by Koren et al. (2009) [cite_start][cite: 1].

* **Key Concept:** Modeling users and items as vectors in a latent factor space. [cite_start]The interaction is modeled as a dot product plus bias terms ($b_u$, $b_i$) to account for systematic tendencies (e.g., some users rate everything high)[cite: 153, 168].
* [cite_start]**Optimization:** Stochastic Gradient Descent (SGD) minimizing Regularized Squared Error[cite: 116, 128].

###  Results (50 Epochs)
| Metric | Result | Notes |
| :--- | :--- | :--- |
| **Final RMSE** | **0.8900** | Outperforms standard baseline (~0.95). |
| **Training Loss** | 0.7250 | Converged around Epoch 45. |

###  Critique & Analysis
1.  [cite_start]**The Cold Start Problem:** The model failed completely for new users (`IndexError`), confirming the paper's note that collaborative filtering struggles with new entities[cite: 42].
2.  **Popularity Bias (The "Harry Potter" Effect):**
    * **Test:** Compared recommendations for two distinct users.
    * **Result:** **1.00 Correlation** (Identical recommendations).
    * **Conclusion:** The learned bias terms ($b_i$) dominated the interaction ($q_i^T p_u$). The model effectively became a "Most Popular" list, failing to personalize.

---

##  Project 2: Neural Collaborative Filtering (The Fix)
**Location:** `src/ncf/`

An implementation of **Neural Collaborative Filtering (NCF)** (He et al., 2017). This project addresses the linearity limitation of Matrix Factorization.

* **Key Concept:** Replaces the simple Dot Product with a **Multi-Layer Perceptron (MLP)**. This allows the model to learn non-linear interactions between users and items.
* **Loss Function:** Binary Cross Entropy (Log Loss) using Negative Sampling (predicting probability of interaction).

###  Results (10 Epochs)
| Metric | Result | Notes |
| :--- | :--- | :--- |
| **Final Loss** | **0.3180** | Dropped from initial random guessing (0.69). |

###  Critique & Analysis
* **Did it fix the Bias?**
    * **Correlation:** **0.9886** (User A vs User B).
    * **Conclusion:** Success. Unlike the MF model (1.00), the Neural Network produced *slightly* different probabilities for different users. It learned to personalize, though the small dataset size still resulted in high similarity.

---

##  Project 3: Sequential Recommendation (The Transformer)
**Location:** `src/sasrec/`

An implementation of **SASRec (Self-Attentive Sequential Recommendation)** using a Transformer Encoder.

* **Key Concept:** Treats user history as a **sequence** (Time Series) rather than a bag of items. Uses Self-Attention to predict the *next* movie based on the order of previous movies.
* **Architecture:** Embedding Layer + Positional Embeddings + Transformer Block.

###  Results (20 Epochs)
| Metric | Result | Notes |
| :--- | :--- | :--- |
| **Final Loss** | **4.4580** | Multi-class Cross Entropy over 9,000 items. |

###  Critique & Analysis
* **Does Order Matter?**
    * **Test:** Fed sequence `[A, B]` vs `[B, A]` to see if predictions changed.
    * **Result:** **Identical Output.**
    * **Conclusion:** The model failed to learn sequential dependencies on the MovieLens Small dataset (100k ratings). Transformers are data-hungry; without millions of interactions, they often degrade into "Bag of Words" models, ignoring time dynamics.

---

## 🚀 How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Matrix Factorization:**
    ```bash
    cd src/mf
    python train.py
    ```

3.  **Run Neural Collaborative Filtering:**
    ```bash
    cd src/ncf
    python train.py
    ```

4.  **Run SASRec (Transformer):**
    ```bash
    cd src/sasrec
    python train.py
    ```

## 📜 Citations
1.  Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. *Computer*, 42(8), 30-37.
2.  He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). Neural Collaborative Filtering. *WWW*.
3.  Kang, W. C., & McAuley, J. (2018). Self-Attentive Sequential Recommendation. *ICDM*.
