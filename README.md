# Implementation of Matrix Factorization for Recommender Systems

This repository contains a PyTorch implementation of the classic paper **"Matrix Factorization Techniques for Recommender Systems"** by Yehuda Koren et al. (2009).

The goal of this project was to reproduce the results of the SVD++ algorithm on the MovieLens dataset and perform a critique of the model's limitations in a modern context.

## The Paper
* **Title:** Matrix Factorization Techniques for Recommender Systems  
* **Authors:** Yehuda Koren, Robert Bell, Chris Volinsky  
* **Key Concept:** Modeling users and items as vectors in a latent factor space, optimizing the dot product interaction with added bias terms.

## Implementation Details
* **Framework:** PyTorch  
* **Algorithm:** SVD with Bias Terms (Eq. 4 in the paper)  
* **Optimization:** Stochastic Gradient Descent (SGD) with Weight Decay (L2 Regularization)  
* **Dataset:** MovieLens Small (100k ratings)

## Results

I trained the model for 50 epochs on an NVIDIA T4 GPU.

| Metric | Result | Notes |
|-------|--------|-------|
| **Final RMSE** | **0.8900** | Outperforms standard baseline (0.95) |
| **Training Loss** | 0.7250 | Converged around Epoch 45 |

## Critique & Analysis

After achieving the target accuracy, I stress-tested the model to find its "Research Flaws."

### 1. The Cold Start Problem
The model uses `nn.Embedding` layers, which are static and require pre-defined user/item IDs.  
* **Test:** Attempted to predict for a new `user_id=99999`.  
* **Result:** `IndexError: index out of range`  
* **Conclusion:** This architecture cannot serve new users without a full re-train. Modern systems require hybrid approaches or feature-based inputs.

### 2. Popularity Bias (The "Harry Potter" Effect)
I compared recommendations for two distinct users (User A vs User B).  
* **Correlation:** **1.00** (Perfect Correlation)  
* **Observation:** Both users received identical ranking scores for the top 50 movies.  
* **Conclusion:** The learned item bias terms ($b_i$) dominated the latent interaction terms ($q_i^T p_u$).  
  The model effectively became a "Most Popular" list, failing to capture niche user tastes.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt

