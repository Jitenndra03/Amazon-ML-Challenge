# ML Challenge 2025: Smart Product Pricing Solution Template

**Team Name:** Nocturnals
**Team Members:**  
Jitendra Singh[Leader]
Raghav Agarwal
Shubham Srivastava
Tanishka Gupta

**Submission Date:** [13-10-2025]

---

## 1. Executive Summary

Our approach integrates multimodal learning by combining structured, textual, and visual information to improve predictive accuracy. We generate manual features from metadata, semantic text embeddings using SBERT, and visual embeddings using CLIP, then fuse them into a unified feature space for modeling with LightGBM. The key innovation lies in this feature-level fusion of language and vision representations, enabling the model to capture rich cross-domain patterns beyond traditional tabular learning.

---

## 2. Methodology Overview

### 2.1 Problem Analysis

We interpreted the pricing challenge as a multimodal regression problem, where the goal is to predict optimal or realistic product prices based on diverse information sources — product descriptions, images, and catalog metadata. During exploratory data analysis (EDA), we observed that price distributions were highly right-skewed, motivating the use of a log transformation on the target variable. Additional insights included strong correlations between certain extracted attributes (like brand, quantity, and product category) and price, while text and image features revealed semantic clusters of similar products with consistent pricing patterns, highlighting the value of combining visual and textual cues for better generalization.

**Key Observations:**

### 2.2 Solution Strategy

Our high-level approach is based on a multimodal learning framework that integrates information from three complementary data sources — structured tabular features, textual descriptions, and product images. We independently extract embeddings from text using SBERT and from images using CLIP, while engineered features are derived from structured catalog data through regex-based parsing. These heterogeneous feature sets are fused at the feature level and used to train a LightGBM regressor, leveraging its strength in handling high-dimensional and mixed-type data. This design effectively captures relationships across modalities, improving robustness and prediction accuracy.

**Approach Type:** Hybrid Multimodal Learning  
**Core Innovation:**  
A feature-level fusion of structured manual features, SBERT-based text embeddings, and CLIP-based image embeddings combined into a unified representation for regression modeling using LightGBM. This hybrid design enables the model to capture rich cross-modal relationships that improve pricing accuracy.

---

## 3. Model Architecture

### 3.1 Architecture Overview

              ┌───────────────────────────────────────────────┐
              │                 RAW INPUT DATA                │
              │             (train.csv / test.csv)            │
              └───────────────────────────────────────────────┘
                                        │
                                        ▼
              ┌───────────────────────────────────────────────┐
              │              DATA PREPROCESSING               │
              │        (e.g., Target Log Transform)           │
              └───────────────────────────────────────────────┘
                                        │
                                        ▼
       ┌───────────────────────────────────────────────────────────────┐
       │                 PARALLEL FEATURE GENERATION                   │           └────────────┬──────────────────────┬───────────────────────┬───┘
                    │                      │                       │
                    ▼                      ▼                       ▼
┌──────────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────────┐
│ (A) MANUAL FEATURES      │   │ (B) TEXT EMBEDDINGS      │   │ (C) IMAGE EMBEDDINGS     │
│    (from catalog_content)│   │     (from cleaned text)  │   │     (from image_link)    │
│ • Text Cleaning          │   │ • SBERT Model            │   │ • Image Download         │
│ • Regex Extraction       │   │   ('all-MiniLM-L6-v2')   │   │ • CLIP Model             │
│   (Brand, IPQ, Qty, etc.)│   │ • Output: 384-dim vector │   │ ('clip-vit-base-patch32')│
└──────────────────────────┘   └──────────────────────────┘   └──────────────────────────┘
                    │                       │                      │
                    ▼                       ▼                      ▼
          ┌───────────────────────────────────────────────────────────────┐
          │                        FEATURE FUSION                         │
          │     (Concatenate A, B, and C into a single DataFrame)         │
          └───────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
          ┌───────────────────────────────────────────────────────────────┐
          │                   FINAL DATA PREPARATION                      │
          │  (e.g., Convert booleans to int, objects to 'category' dtype) │
          └───────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
          ┌───────────────────────────────────────────────────────────────┐
          │                    PREDICTIVE MODELING                        │
          │                   (LightGBM Regressor)                        │
          └───────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
          ┌───────────────────────────────────────────────────────────────┐
          │                    INVERSE TRANSFORMATION                     │
          │                        (np.expm1())                           │
          └───────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
          ┌───────────────────────────────────────────────────────────────┐
          │                        FINAL PREDICTION                       │
          │                     (submission.csv)                          │
          └───────────────────────────────────────────────────────────────┘



### 3.2 Model Components

**Text Processing Pipeline:**

- [ ] Preprocessing steps:[To reduce noise, we used regular expressions to strip structural boilerplate text (e.g., 'item', 'value') from the raw catalog_content.]
- [ ] Model type: [Our hybrid approach combined regex for extracting structural features (like brand and IPQ) with a pre-trained SBERT model (all-MiniLM-L6-v2) for generating semantic embeddings.]
- [ ] Key parameters: [Embedding size = 384]

**Image Processing Pipeline:**

- [ ] Preprocessing steps: [Image download, resizing to 224×224, normalization, and RGB conversion]
- [ ] Model type: [CLIP (clip-vit-base-patch32)]
- [ ] Key parameters: [Embedding size = 512]

---

## 4. Model Performance

### 4.1 Validation Results

- **SMAPE Score:** 40.69%



## 5. Conclusion

Our hybrid multimodal approach successfully combined structured, textual, and visual features into a unified learning framework for price prediction. The fusion of SBERT and CLIP embeddings with LightGBM regression significantly improved generalization across diverse product categories. Key lessons learned include the importance of target log transformation, balanced feature scaling, and leveraging pretrained models for robust multimodal representations.

---

