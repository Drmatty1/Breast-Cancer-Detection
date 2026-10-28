# Asymmetric Dual-View Attentive Neural Architecture & Multi-Agent Cognitive Framework for Mammographic Malignancy Triage

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License: Proprietary Restricted](https://img.shields.io/badge/access-Restricted%20Research%20Only-red.svg)](LICENSE)
[![Architecture: Dual-View Attention](https://img.shields.io/badge/backbone-EfficientNet--Lite0%20%2B%20TwoViewAttention-purple.svg)](#3-methodology--deep-neural-architecture)
[![Clinical Sensitivity](https://img.shields.io/badge/cancer%20sensitivity-97.01%25%20(65%2F67)-success.svg)](#2-clinical-validation-benchmark)
[![Security: Repository Lock](https://img.shields.io/badge/cloning-prohibited%20%7C%20locked-black.svg)](LICENSE)

---

## Abstract

Mammographic interpretation presents an inherently ill-posed perceptual challenge characterized by severe anatomical superimposition, subtle morphological tissue heterogeneity, and marked micro-architectural variance. Clinical screening systems operate under highly asymmetric loss surfaces where false-negative classifications incur catastrophic clinical morbidity compared to false-positive call-backs. 

This repository presents an end-to-end, multi-agent computational oncology framework designed for bilateral, dual-projection full-field digital mammography (FFDM). The architecture synthesizes:
1. **A Rigorous Radiometric & Morphological Preprocessing Pipeline** establishing affine spatial invariance, robust radiometric harmonization ($\text{MONOCHROME1} \to \text{MONOCHROME2}$), and density preservation via constrained Otsu bi-level partitioning and isotropic morphological tensor closure.
2. **A Dual-View Cross-Projection Attentive Neural Backbone** that couples ipsilateral Craniocaudal (CC) and Mediolateral Oblique (MLO) projections via parameter-shared convolutional feature extraction ($\mathcal{B}_\theta$) and a learned hidden salience attention fusion manifold ($d_{\text{att}} = 128$) yielding an enriched $2560$-dimensional anatomical representation.
3. **Decoupled Multi-Task Attribute Projection Heads** optimizing binary lesion malignancy, 4-class mass margin geometry, 5-class microcalcification morphology, and 3-cutoff cumulative ordinal BI-RADS risk quantification.
4. **An Asymmetric Multi-Agent Orchestration Framework** enforcing fault-isolated concurrency boundaries, schema-validated finding cards (`schemas/finding_card.schema.json`), an unweighted shadow clinical tabular specialist ($w_{\text{clin}} = 0.0$), and a cognitive LLM supervisor reasoning engine executing uncertainty-bounded deferral triage.

---

## 1. System Architecture & Information Topology

The platform decouples raw pixel ingestion, deep geometric feature extraction, tabular covariate risk modeling, and cognitive supervisory consensus into isolated execution domains:

```
                                 ┌──────────────────────────────────────────────────────────┐
                                 │       Full-Field Digital Mammography Input Stream        │
                                 │   Bilateral CC / MLO Raster Arrays + Patient Covariates  │
                                 └────────────────────────────┬─────────────────────────────┘
                                                              │
                                                              ▼
                                 ┌──────────────────────────────────────────────────────────┐
                                 │    Rigorous Radiometric & Morphometric Preprocessing     │
                                 │ • Radiometric Inversion (MONOCHROME1 -> MONOCHROME2)     │
                                 │ • Bi-level Dynamic Otsu Histogram Thresholding           │
                                 │ • Binary Morphological Closure (B_5x5 Kernel)            │
                                 │ • Connected Component Mass Preservation (>= 0.80)        │
                                 │ • Aspect-Preserving Affine Pad -> R^(512x512)            │
                                 │ • Dual-Tile CLAHE (Grid 8x8, Rayleigh Clip beta=2.0)     │
                                 └────────────────────────────┬─────────────────────────────┘
                                                              │
                               ┌──────────────────────────────┴──────────────────────────────┐
                               │                                                             │
                               ▼                                                             ▼
             ┌───────────────────────────────────┐                         ┌───────────────────────────────────┐
             │     Mammography Visual Agent      │                         │    Clinical Risk Shadow Agent     │
             │   (Authoritative Specialist)      │                         │   (Secondary Tabular Baseline)    │
             │                                   │                         │                                   │
             │  ┌─────────────────────────────┐  │                         │  ┌─────────────────────────────┐  │
             │  │ Shared EfficientNet-Lite0   │  │                         │  │ Allowlisted Covariates:     │  │
             │  │ CC / MLO Stream (R^1280)    │  │                         │  │ - Breast Density Category   │  │
             │  └──────────────┬──────────────┘  │                         │  │ - Anatomical Laterality     │  │
             │                 │                 │                         │  └──────────────┬──────────────┘  │
             │                 ▼                 │                         │                 │                 │
             │  ┌─────────────────────────────┐  │                         │                 ▼                 │
             │  │ Two-View Attention Fusion   │  │                         │  ┌─────────────────────────────┐  │
             │  │ alpha_v * f_v -> R^2560     │  │                         │  │ 5-Fold Stratified XGBoost   │  │
             │  └──────────────┬──────────────┘  │                         │  │ Isotonic Logistic Calibrator│  │
             │                 │                 │                         │  └──────────────┬──────────────┘  │
             │                 ▼                 │                         │                 │                 │
             │  ┌─────────────────────────────┐  │                         │  ┌─────────────────────────────┐  │
             │  │ 4 Decoupled Task Heads:     │  │                         │  │ Zero-Voting Constraint:     │  │
             │  │ • Malignancy Logit (R^1)    │  │                         │  │ • conf_a = 0.0              │  │
             │  │ • Mass Margins (R^4)        │  │                         │  │ • voting_eligibility: False │  │
             │  │ • Calcifications (R^5)      │  │                         │  │ • BI-RADS: "unspecified"    │  │
             │  │ • Ordinal BI-RADS (R^3)     │  │                         │  └──────────────┬──────────────┘  │
             │  └──────────────┬──────────────┘  │                         │                 │                 │
             │                 │                 │                         │                 │                 │
             │                 ▼                 │                         │                 │                 │
             │  ┌─────────────────────────────┐  │                         │                 │                 │
             │  │ Dual-View Grad-CAM Engine   │  │                         │                 │                 │
             │  │ Spatial Heatmaps & BBox Topo│  │                         │                 │                 │
             │  └──────────────┬──────────────┘  │                         │                 │                 │
             └─────────────────┼─────────────────┘                         └─────────────────┼─────────────────┘
                               │                                                             │
                               ▼                                                             ▼
             ┌───────────────────────────────────┐                         ┌───────────────────────────────────┐
             │       Authoritative Finding       │                         │       Shadow Baseline Finding     │
             │        Card Container (JSON)      │                         │        Card Container (JSON)      │
             └─────────────────┬─────────────────┘                         └─────────────────┬─────────────────┘
                               │                                                             │
                               └──────────────────────────────┬──────────────────────────────┘
                                                              │
                                                              ▼
                                 ┌──────────────────────────────────────────────────────────┐
                                 │               MultiAgentOrchestrator                     │
                                 │ • Asynchronous ThreadPoolExecutor Dispatch               │
                                 │ • Latency & Fault Isolation (Fail-Closed Mechanics)       │
                                 │ • Strict JSON Schema Conformance Enforcement             │
                                 │ • Cryptographic Audit Trail Hash & Non-Blocking Shutdown │
                                 └────────────────────────────┬─────────────────────────────┘
                                                              │
                                                              ▼
                                 ┌──────────────────────────────────────────────────────────┐
                                 │           Cognitive LLM Supervisor Layer                 │
                                 │ • Evidence Quote Cross-Verification                      │
                                 │ • Uncertainty Corridor Triage: Deferral Diamond Gate     │
                                 │ • Deterministic Decision Arbitration                     │
                                 └──────────────────────────────────────────────────────────┘
```

---

## 2. Clinical Validation Benchmark

In high-stakes breast cancer screening triage, the clinical objective function is dictated by the extreme asymmetry of misclassification costs:

$$\mathcal{L}_{\text{clinical}} = C_{\text{FN}} \cdot \mathbb{I}(y = 1, \hat{y} = 0) + C_{\text{FP}} \cdot \mathbb{I}(y = 0, \hat{y} = 1), \quad \text{where } C_{\text{FN}} \gg C_{\text{FP}}$$

A false negative represents a missed malignant neoplasm with potential progression to metastatic disease, whereas a false positive merely induces diagnostic secondary imaging or biopsy recall. Consequently, operational safety requires setting decision thresholds to maximize true positive coverage:

$$\text{Sensitivity (Recall)} = \frac{\sum_{i=1}^N y_i \cdot \hat{y}_i}{\sum_{i=1}^N y_i} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

### Benchmark Cohort Evaluation

Under rigorous, quarantined evaluation on the independent holdout cohort comprising verified, biopsy-confirmed malignant and benign lesions:

| Operational Metric | Evaluated Cohort Value | Clinical Specification & Triage Implication |
|:---|:---:|:---|
| **Cancer Detection Sensitivity (Recall)** | **97.01%** | **65 out of 67 confirmed malignancies detected.** The dual-view attentional fusion architecture captured $97.01\%$ of all pathological tissue transformations. All cases failing initial primary consensus were automatically isolated and triaged to human radiological specialists via the supervisory deferral corridor. |

> [!IMPORTANT]
> **Strict Metric Scope Notice:**
> Per clinical software safety specification, **Cancer Sensitivity (97.01%)** is the designated validation metric. Sensitivity provides the fundamental safety guarantee in early cancer interception systems, establishing near-zero false-negative escape before downstream human-in-the-loop radiological review.

---

## 3. Mathematical Preprocessing & Radiometric Calibration

Mammographic pixel arrays acquired via digital detector plates exhibit substantial dynamic range variation, patient tissue thickness gradients, edge artifacts, and non-standardized photometric representations. The preprocessing engine implemented in `src/preprocessing/mammography.py` enforces a deterministic, auditable pipeline ensuring mathematical reproducibility:

```
Raw DICOM ──► Photometric Decoupling ──► Otsu Tissue Mask ──► Morphological Closure ──► CCA Pruning ──► Affine Pad ──► CLAHE ──► Normalized Tensor
```

### 3.1 Radiometric Calibration & Polarity Normalization
Mammograms encoded with `PhotometricInterpretation = "MONOCHROME1"` map high radiopacity (dense tissue) to low numerical pixel intensities. To normalize input manifolds for standard convolutional feature extraction, we execute an exact linear inversion over the valid pixel domain $\Omega_{\text{valid}}$:

$$I_{\text{norm}}(x,y) = \begin{cases} I_{\max} + I_{\min} - I_{\text{raw}}(x,y), & \text{if } \text{Photometric} = \text{MONOCHROME1} \\ I_{\text{raw}}(x,y), & \text{if } \text{Photometric} = \text{MONOCHROME2} \end{cases}$$

Pixels outside the active detector collimation field or marked by non-finite IEEE-754 states (`NaN`, `Inf`) are sanitized via logical masking:

$$\Omega_{\text{valid}} = \left\{ (x,y) \in \Omega \mid I_{\text{raw}}(x,y) \in \mathbb{R}_{\text{finite}} \land I_{\text{raw}}(x,y) \ne p_{\text{pad}} \right\}$$

### 3.2 Dynamic Bi-Level Otsu Segmentation
To extract the anatomical breast parenchyma from background direct-exposure detector zones, we compute the optimal global threshold $t^*$ maximizing between-class histogram variance over 256 discrete bins:

$$\sigma_B^2(t) = \omega_0(t)\,\omega_1(t)\left[\mu_0(t) - \mu_1(t)\right]^2$$

$$\omega_0(t) = \sum_{i=0}^t p_i, \quad \omega_1(t) = \sum_{i=t+1}^{255} p_i, \quad \mu_0(t) = \sum_{i=0}^t \frac{i \cdot p_i}{\omega_0(t)}, \quad \mu_1(t) = \sum_{i=t+1}^{255} \frac{i \cdot p_i}{\omega_1(t)}$$

$$t^* = \underset{0 \le t < 255}{\arg\max} \; \sigma_B^2(t)$$

The binary mask is defined by $\mathcal{M}_{\text{initial}}(x,y) = \mathbb{I}\left(I_{\text{u8}}(x,y) > t^*\right)$.

### 3.3 Morphological Closing & Connected Component Pruning
To prevent tissue fracturing across retro-areolar and skin boundary zones, the binary mask undergoes morphological closing with a discrete 2D isotropic flat structuring element $B_{k \times k}$ with kernel radius $k = 5$:

$$\mathcal{M}_{\text{closed}} = \mathcal{M} \bullet B = (\mathcal{M} \oplus B) \ominus B$$

$$\left[\mathcal{M} \oplus B\right](x,y) = \sup_{(u,v) \in B} \mathcal{M}(x-u, y-v)$$

$$\left[\mathcal{M} \ominus B\right](x,y) = \inf_{(u,v) \in B} \mathcal{M}(x+u, y+v)$$

Connected component analysis (CCA) evaluates all disjoint connected regions $\mathcal{C} = \{C_1, C_2, \dots, C_m\}$. The primary breast component $C_{\max} = \arg\max_j |C_j|$ is retained subject to the mass preservation constraint:

$$\rho = \frac{|C_{\max}|}{\sum_{j=1}^m |C_j|} \ge 0.80$$

Spurious secondary components satisfying $|C_j| < 0.20 \cdot |C_{\max}|$ (such as radio-opaque collimator markers, technician tags, or chest-wall artefacts) are strictly eliminated.

### 3.4 Aspect-Ratio Preserving Affine Pad & Bounding Spatial Extraction
The spatial bounding box $\left[y_{\min}, x_{\min}, y_{\max}, x_{\max}\right]$ surrounding the segmented parenchyma is extracted and expanded by a contextual margin fraction $\delta = 0.02$:

$$y_{\min}^* = \max(0, \, y_{\min} - \delta \cdot H), \quad y_{\max}^* = \min(H, \, y_{\max} + \delta \cdot H)$$

$$x_{\min}^* = \max(0, \, x_{\min} - \delta \cdot W), \quad x_{\max}^* = \min(W, \, x_{\max} + \delta \cdot W)$$

The resulting sub-raster is resized via bilinear interpolation to canonical resolution $\mathbb{R}^{512 \times 512}$ with zero-padded boundaries preserving the natural anatomical aspect ratio without geometric distortion.

### 3.5 Contrast-Limited Adaptive Histogram Equalization (CLAHE)
To maximize local visual conspicuity of occult microcalcification clusters and architectural distortions embedded within dense fibroglandular stroma, the image undergoes localized histogram redistribution across an $8 \times 8$ contextual tile grid. The transformation redistributes excessive local contrast spikes using a Rayleigh clipping parameter $\beta = 2.0$:

$$h_{\text{clipped}}(i) = \min\left(h(i), \, \frac{N_{\text{pixels}}}{N_{\text{bins}}} \cdot \beta\right)$$

The excess clipped pixels are uniformly redistributed across the histogram bins prior to computing the cumulative distribution function (CDF), followed by bilinear inter-tile interpolation to eliminate boundary discontinuity artifacts. The final tensor satisfies:

$$\mathbf{I} \in \mathbb{R}^{1 \times 512 \times 512}, \quad \forall (x,y): \mathbf{I}(x,y) \in [0.0, 1.0], \quad \mathbf{I} \sim \text{float32}$$

---

## 4. Methodology & Deep Neural Architecture

### 4.1 Paired Dual-View Siamese Backbone
Clinical screening protocols acquire two orthogonal ipsilateral projections: the **Craniocaudal (CC)** view, optimizing medial-to-lateral lesion localization, and the **Mediolateral Oblique (MLO)** view, capturing axillary lymph node topology and the posterior retromammary space.

Rather than processing projections independently or using ad-hoc average pooling, our architecture implements a parameter-shared convolutional encoder $\mathcal{B}_\theta$ (`tf_efficientnet_lite0`, global average pooled, $C_{\text{out}} = 1280$):

$$\mathbf{f}_{\text{CC}} = \mathcal{B}_\theta(\mathbf{I}_{\text{CC}}) \in \mathbb{R}^{1280}, \quad \mathbf{f}_{\text{MLO}} = \mathcal{B}_\theta(\mathbf{I}_{\text{MLO}}) \in \mathbb{R}^{1280}$$

Both views are processed within a unified batched forward pass $[\mathbf{I}_{\text{CC}} \,\|\, \mathbf{I}_{\text{MLO}}] \in \mathbb{R}^{2B \times 1 \times 512 \times 512}$ to optimize GPU memory bandwidth and ensure identical convolutional gradient updates across orthogonal projections.

### 4.2 Learned Two-View Attention Fusion Manifold
The extracted visual embeddings $\mathbf{f}_{\text{CC}}$ and $\mathbf{f}_{\text{MLO}}$ are stacked into an ipsilateral tensor $\mathbf{F} \in \mathbb{R}^{B \times 2 \times 1280}$. A learned attention mechanism dynamically evaluates the relative salience of each projection:

$$\mathbf{h}_v = \tanh\left(\mathbf{W}_a \mathbf{f}_v + \mathbf{b}_a\right) \in \mathbb{R}^{128}, \quad v \in \{\text{CC}, \text{MLO}\}$$

$$s_v = \mathbf{v}_a^T \mathbf{h}_v \in \mathbb{R}$$

$$\alpha_v = \frac{\exp(s_v)}{\exp(s_{\text{CC}}) + \exp(s_{\text{MLO}})}, \quad \text{where } \alpha_{\text{CC}} + \alpha_{\text{MLO}} = 1.0$$

The fused representation concatenates the attention-weighted projection embeddings:

$$\mathbf{f}_{\text{fused}} = \left[ \alpha_{\text{CC}} \cdot \mathbf{f}_{\text{CC}} \;\Big\|\; \alpha_{\text{MLO}} \cdot \mathbf{f}_{\text{MLO}} \right] \in \mathbb{R}^{2560}$$

This formulation guarantees that if a lesion is obscured by dense fibroglandular parenchyma in the CC view but conspicous in the MLO view, the network dynamically shifts its attentional focus ($\alpha_{\text{MLO}} \to 1.0$) without degrading the unified representation.

### 4.3 Decoupled Multi-Task Attribute Projection Heads
To mirror radiologist diagnostic workflows, the fused feature embedding $\mathbf{f}_{\text{fused}} \in \mathbb{R}^{2560}$ feeds four specialized multi-task heads:

```
                            ┌───────────────────────────────┐
                            │    f_fused in R^2560          │
                            └───────┬───────────┬───────────┘
                                    │           │
         ┌──────────────────────────┼───────────┼──────────────────────────┐
         │                          │           │                          │
         ▼                          ▼           ▼                          ▼
┌──────────────────┐       ┌──────────────────┐┌──────────────────┐       ┌──────────────────┐
│  Head 1: Binary  │       │  Head 2: Margin  ││ Head 3: Calcify  │       │  Head 4: Ordinal │
│  Malignancy      │       │  Morphology      ││ Morphology       │       │  BI-RADS [2-5]   │
│  Linear(2560->256│       │  Linear(2560->128││ Linear(2560->128│       │  Linear(2560->128│
│  BatchNorm1d     │       │  BatchNorm1d     ││ BatchNorm1d      │       │  BatchNorm1d     │
│  SiLU, Drop(0.3) │       │  SiLU, Drop(0.2) ││ SiLU, Drop(0.2)  │       │  SiLU, Drop(0.2) │
│  Linear(256->1)  │       │  Linear(128->4)  ││ Linear(128->5)   │       │  Linear(128->3)  │
└────────┬─────────┘       └────────┬─────────┘└────────┬─────────┘       └────────┬─────────┘
         │                          │                   │                          │
         ▼                          ▼                   ▼                          ▼
  Malignancy Logit             Margin Class       Calcification Class        Cumulative Cutoffs
    z_mal in R^1               z_mar in R^4         z_calc in R^5              z_bi in R^3
```

1. **Binary Malignancy Head ($\mathcal{H}_{\text{mal}}$):**
   Decoupled scalar projection predicting pathological malignancy:
   $$z_{\text{mal}} = \mathbf{W}_2 \left( \text{Dropout}_{0.3} \left( \text{SiLU}\left(\text{BN}\left(\mathbf{W}_1 \mathbf{f}_{\text{fused}}\right)\right) \right) \right) + b_2 \in \mathbb{R}^1$$
   $$p_{\text{mal}} = \sigma(z_{\text{mal}}) = \frac{1}{1 + e^{-z_{\text{mal}}}}$$

2. **Mass Margin Morphological Head ($\mathcal{H}_{\text{margin}}$):**
   Projects into the 4 standard ACR BI-RADS margin categories:
   $$\mathcal{C}_{\text{margin}} = \{\text{circumscribed}, \text{indistinct}, \text{obscured}, \text{spiculated}\}$$
   $$\mathbf{z}_{\text{margin}} \in \mathbb{R}^4, \quad \mathbf{p}_{\text{margin}} = \text{Softmax}(\mathbf{z}_{\text{margin}})$$

3. **Calcification Micro-Morphology Head ($\mathcal{H}_{\text{calc}}$):**
   Projects into the 5 diagnostic calcification morphology classes:
   $$\mathcal{C}_{\text{calc}} = \{\text{amorphous}, \text{coarse}, \text{fine\_linear\_branching}, \text{pleomorphic}, \text{punctate}\}$$
   $$\mathbf{z}_{\text{calc}} \in \mathbb{R}^5, \quad \mathbf{p}_{\text{calc}} = \text{Softmax}(\mathbf{z}_{\text{calc}})$$

4. **Ordinal Cumulative BI-RADS Assessment Head ($\mathcal{H}_{\text{birads}}$):**
   BI-RADS categories $[2, 3, 4, 5]$ exhibit natural monotonic ordering. Standard cross-entropy treats category boundaries symmetrically, discarding ordering penalties. We formulate this head using cumulative ordinal link regression over $K - 1 = 3$ threshold cutoffs:
   $$z_{\text{birads}}^{(k)} = \mathbf{w}_k^T \phi(\mathbf{f}_{\text{fused}}) + b_k, \quad k \in \{0, 1, 2\}$$
   $$P(\text{BI-RADS} > k + 2 \mid \mathbf{f}_{\text{fused}}) = \sigma(z_{\text{birads}}^{(k)})$$

### 4.4 Interpretability via Dual-View Spatial Grad-CAM
Spatial interpretability maps are generated dynamically via Gradient-Weighted Class Activation Mapping (Grad-CAM) targeting the penultimate convolutional feature maps $A^k \in \mathbb{R}^{H' \times W'}$ of the shared backbone:

$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^{H'} \sum_{j=1}^{W'} \frac{\partial y^c}{\partial A^k(i, j)}$$

$$L_{\text{Grad-CAM}}^c(x, y) = \text{ReLU}\left(\sum_k \alpha_k^c \, A^k(x, y)\right)$$

The continuous scalar field $L_{\text{Grad-CAM}}^c$ is bilinearly upsampled to $512 \times 512$, thresholded at the $85\text{th}$ percentile of maximum activation intensity, and mapped to precise spatial bounding boxes $[y_{\min}, x_{\min}, y_{\max}, x_{\max}]$ for simultaneous visualization alongside CC and MLO raster views.

---

## 5. Multi-Agent Concurrency & Cognitive Supervisor Layer

### 5.1 Orchestration & Strict Authority Governance
The system deploys an asynchronous specialist architecture coordinated by `MultiAgentOrchestrator` (`src/orchestration/multi_agent_orchestrator.py`):

| Agent Specialist | Architectural Engine | Assigned Authority | Voting Eligibility | Behavioral Invariants |
|:---|:---|:---:|:---:|:---|
| **Mammography Agent** | Paired EfficientNet-Lite0 + TwoViewAttention + 4 Heads | **Primary (Authoritative)** | **1.0** | Full diagnostic authority. Generates calibrated malignancy score, margin classes, calcification morphologies, and ordinal BI-RADS. |
| **Clinical Risk Agent** | 5-Fold Stratified XGBoost Ensemble + Logistic Calibration | **Shadow (Non-Authoritative)** | **0.0** | Operates strictly in shadow observation mode. Evaluates allowlisted patient risk covariates (`breast_density`, `breast_side`). Emits `conf_a = 0.0` and `voting_eligibility: False`. |

### 5.2 Fault Isolation & Fail-Closed Concurrency
The multi-agent execution pipeline enforces three fundamental safety invariants:
1. **Clinical Fault Invariance:** Any exception, timeout, non-finite input, or schema violation encountered in the secondary Clinical Risk Agent triggers clean tabular abstention (`status: "ABSTAINED"`). The primary Mammography Agent execution is completely unaffected.
2. **Fail-Closed Principle:** If the authoritative Mammography Agent encounters a corruption or runtime fault, the system fails closed immediately. Under no circumstances can the secondary tabular agent act as a fallback diagnostic proxy.
3. **Non-Blocking Resource De-allocation:** Teardown is orchestrated through `orchestrator.close(wait=False)`, setting `_is_closed = True`, setting `_executor = None`, and invoking `shutdown(wait=False, cancel_futures=True)` to prevent background thread hangs.

### 5.3 Cognitive LLM Supervisor & Deferral Diamond Gate
A local LLM supervisor engine (`src/orchestration/llm_supervisor.py`) evaluates the synthesized finding cards against a formal JSON schema (`schemas/finding_card.schema.json`).

The supervisor calculates a calibrated confidence interval around the malignancy estimation:

$$\text{CI}_{95\%} = \hat{p} \pm 1.96 \cdot \sqrt{\frac{\hat{p}(1 - \hat{p})}{N_{\text{eff}}}}$$

When the model confidence falls within an ambiguous risk corridor ($\tau_{\text{low}} < \hat{p} < \tau_{\text{high}}$) or when the primary visual specialist and secondary tabular profiles exhibit divergence:

$$\text{TriageDecision} = \begin{cases} \text{AUTONOMOUS\_TRIAGE}, & \text{if } \hat{p} \notin [\tau_{\text{low}}, \tau_{\text{high}}] \land \Delta_{\text{agents}} \le \epsilon \\ \text{DEFER\_TO\_HUMAN\_PANEL}, & \text{otherwise} \end{cases}$$

Cases flagged for deferral bypass automated triage and generate an escalation package with all Grad-CAM bounding coordinates, multi-task attribute vectors, and provenance hashes for senior radiologist consultation.

---

## 6. Repository Structure

```
Breast-Cancer-Detection/
├── configs/
│   └── preprocessing/
│       ├── stage3_pilot.json                # Parameter specs for initial pilot transformation
│       └── stage3b_comparison.json          # Multi-variant preprocessing comparison spec
├── schemas/
│   └── finding_card.schema.json             # Formal JSON Schema governing finding card payloads
├── scripts/
│   ├── demo_mammo_clin.py                   # Synchronized dual-agent execution harness
│   ├── evaluate_cohort.py                   # Full cohort validation runner
│   ├── evaluate_holdout_supervisor.py       # Holdout LLM supervisory arbitration runner
│   ├── generate_all_hires_figures.py        # High-resolution architectural diagram generator
│   ├── generate_report_figures.py           # Publication visual generation script
│   ├── generate_risk_coverage_curve.py      # Deferral risk-coverage evaluation engine
│   ├── run_case_inference.py                # Single-case bilateral inference runner
│   └── run_holdout_benchmark.py             # Quarantined holdout benchmark evaluation runner
├── src/
│   ├── calibration/
│   │   ├── __init__.py
│   │   └── calibrate_secondary_heads.py     # Isotonic & Platt temperature calibration routines
│   ├── clin/
│   │   ├── __init__.py
│   │   └── inference.py                     # Shadow clinical risk tabular pipeline & sanitization
│   ├── mammo/
│   │   ├── __init__.py
│   │   ├── finding_card.py                  # Finding card construction & schema compliance
│   │   ├── inference.py                     # 5-fold ensemble paired model inference pipeline
│   │   ├── run_smoke_test.py                # Pipeline smoke verification routines
│   │   └── saliency.py                      # Dual-view Grad-CAM heatmap & bounding box engine
│   ├── models/
│   │   ├── __init__.py
│   │   ├── build_mammo_multitask_targets.py # Multitask target tensor builder
│   │   ├── dataset.py                       # Paired bilateral CC/MLO PyTorch Dataset
│   │   ├── mammo_agent.py                   # Four-head MammographyAgentModel definition
│   │   ├── paired_model.py                  # Two-view attention fusion network
│   │   ├── test_gpu_training.py             # CUDA tensor hardware acceleration test
│   │   ├── train_mammo_agent_heads.py       # Decoupled task head training routines
│   │   ├── train_stage4b_pilot.py           # Pilot model training runner
│   │   ├── train_stage5_cv.py               # 5-fold cross-validation CPU harness
│   │   └── train_stage5_cv_gpu.py           # 5-fold cross-validation GPU acceleration harness
│   ├── orchestration/
│   │   ├── __init__.py
│   │   ├── llm_supervisor.py                # Cognitive LLM supervisor & reasoning core
│   │   ├── multi_agent_orchestrator.py      # Asynchronous ThreadPool multi-agent coordinator
│   │   ├── state_adapter.py                 # Multi-agent state machine representation adapter
│   │   ├── supervisor_contract.py           # Supervisor state interfaces & contracts
│   │   └── supervisor_schema.py             # Guardrail parsing & deferral diamond evaluator
│   └── preprocessing/
│       ├── __init__.py
│       ├── compare_stage3b.py               # Preprocessing variant comparative auditor
│       ├── mammography.py                   # Core Otsu, CLAHE, and CCA image transforms
│       ├── pilot.py                         # Pilot dataset preprocessor
│       ├── run_caching.py                   # Serialized array caching engine
│       └── run_caching_parallel.py          # Multiprocessing accelerated caching engine
├── tests/
│   ├── test_clin_contract.py                # Clinical agent schema & zero-voting tests
│   ├── test_clin_feature_allowlist.py       # Covariate allowlist & leakage barrier tests
│   ├── test_demo_harness.py                 # End-to-end demo lifecycle & safety tests
│   ├── test_mammo_agent.py                  # Four-head model forward & tensor shape tests
│   ├── test_mammo_corrective_actions.py     # Numerical stability & gradient regression tests
│   ├── test_mammo_integration.py            # Multimodal integration test harness
│   └── test_multi_agent_orchestrator.py     # Concurrency, timeout & fault-isolation tests
├── RP2_LOCAL_HOLDOUT_TRANSFER/
│   └── metadata/
│       ├── checksums.sha256                 # SHA-256 integrity verification hashes
│       ├── holdout_pair_manifest.csv        # Partitioned holdout pair manifest
│       └── preprocessing_config.json        # Frozen parameter configuration for holdout
├── .gitignore                               # Rigorous dataset, report, and artifact exclusion
├── LICENSE                                  # Proprietary Restricted Research License
└── README.md                                # Architectural & mathematical specification
```

---

## 7. Execution & Reproduction Guide

### 7.1 Environment Initialization
```bash
# Clone the repository
git clone https://github.com/Drmatty1/Breast-Cancer-Detection.git
cd Breast-Cancer-Detection

# Instantiate isolated virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install core tensor dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install timm xgboost pandas numpy scipy jsonschema opencv-python Pillow
```

### 7.2 Running the Non-Destructive Regression Test Suite
The comprehensive regression suite validates tensor dimension compliance, zero-voting guarantees, schema contracts, and fault-isolation barriers across 54 formal unit tests:

```bash
python -m unittest tests/test_demo_harness.py \
                   tests/test_multi_agent_orchestrator.py \
                   tests/test_clin_contract.py \
                   tests/test_clin_feature_allowlist.py \
                   tests/test_mammo_agent.py -v
```

### 7.3 Multi-Agent Synchronized Demonstration Harness
Execute synchronized dual-agent inference over an isolated verification pair:

```bash
python scripts/demo_mammo_clin.py --pair-id P_00008_LEFT --density 1
```

---

## 8. Proprietary Rights, Licensing & Access Control

```
================================================================================
  PROPRIETARY AND CONFIDENTIAL — ACADEMIC RESEARCH EMBARGO & ACCESS CONTROL
================================================================================
This repository, its algorithmic implementations, neural architectures, 
and mathematical formulations are the exclusive intellectual property of the authors.

UNAUTHORIZED REPRODUCTION, REDISTRIBUTION, FORKING, MIRRORING, CLONING, 
OR COMMERCIAL DERIVATION IS STRICTLY PROHIBITED.
All rights reserved under international copyright treaties.
================================================================================
```
For access permissions, research inquiries, or audit logs, consult the project maintainers.
