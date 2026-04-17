# GNSS Spoofing Detection: AI-Driven Defense Architecture
> **Technical General Championship 2026** <br>
> **Problem Statement:** Generative AI powered by NyneOS <br>
> **Team:** The Overfitters <br> 
> **Team Code:** 12110 <br>
> **Bhawan:** Rajendra Bhawan

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-PyTorch-orange.svg)
![XGBoost](https://img.shields.io/badge/Machine%20Learning-XGBoost-green.svg)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Bidirectional%20LSTM-pink.svg)

## 1. Abstract & Project Overview
This repository contains our team's Round 1 submission for the AI-driven GNSS Spoofing Detection problem statement. Unmanned Aerial Vehicles (UAVs) rely heavily on unencrypted civilian GNSS signals, making them highly vulnerable to signal falsification (spoofing). 

Recognizing the complexity of multi-level spoofing—from brute-force power overpowering to sophisticated "matched-phase" slow drifts—we took an iterative, multi-model approach. We began with strict, unsupervised physical heuristics and progressively advanced to a deep sequential sequence model to maximize our Weighted F1 Score and ensure hardware robustness.

---
## 2. Dataset: Spatial GNSS Spoofing

- 1.⁠ **⁠Data Sourcing:**<br>
The raw baseband data originates from the publicly available Mendeley repository: A DATASET for GPS Spoofing Detection on Unmanned Aerial System. It consists of recordings from an 8-channel GPS receiver mounted on a drone during dynamic flights (0-60 mph), capturing both authentic satellite signals and active spoofing attacks.

- 2.⁠ **⁠Transformation & Reshaping:**<br>
The original repository distributed the data across three separate, 3D wide-matrix files. To make this data compatible with standard spatial machine learning models, we consolidated the files and applied a spatial flattening pipeline. We pivoted the 8 parallel hardware channels (ch0 through ch7) so that every individual row in the final CSV represents a single, frozen microsecond "snapshot" of the entire receiver state. Each snapshot is mapped to a binary target variable: 0 (Authentic Flight) or 1 (Spoofed Attack).
---

## 3. Our Iterative Approaches

### Approach 1: Physics-Informed Unsupervised Anomaly Detection (Isolation Forest)
**Objective:** To detect spoofing using purely first-principle reasoning without relying on labeled training data.<br>
**Methodology:** We hypothesized that while an attacker can spoof a location, they struggle to replicate flawless temporal physics. We engineered true rate-of-change ($\Delta$) features per PRN (Satellite) for kinematics ($\Delta$ Doppler, $\Delta$ Pseudorange) and signal power ($\Delta$ CN0). We fed these physical deltas into an **Isolation Forest** and aggregated the anomaly scores into a receiver-level probability using fractional voting.

**Quantitative Results & Learnings:**

Performance validation relied on an external labeled database. Consequently, the model's efficacy was heavily dependent on the assumed anomaly ratio within the data, making hyperparameter tuning highly volatile for the generalized test dataset.

* **Normal Data (Level 0):** 99.3% correctly ignored (Extremely low False Positive Rate).
* **Crude Attacks (Level 1):** ~16% detection rate.
* **Covert Attacks (Level 2 & 3):** < 1% detection rate.
* **Conclusion:** While highly precise on authentic data, the unsupervised model was overly conservative. It successfully caught violent physical jumps but was completely blind to sophisticated, slow-drifting attacks. The architecture's suboptimal recall is likely the result of a `contamination` hyperparameter mismatch between the validation dataset and the test dataset. 

> *Note: We utilized this model as our fundamental baseline, evaluating subsequent architectures against its performance.*

### Approach 2: Domain-Adversarial Neural Network (DANN) with 1D-ResNet
**Objective:** To solve the "Domain Gap" problem. Lab-trained models often fail in real-world deployments because different GNSS hardware receivers possess unique internal biases. This approach aimed to build a hardware-agnostic model capable of detecting complex time-series attacks on the unseen NyneOS test drone.

**Methodology:** We engineered a robust Deep Learning pipeline that fused physical heuristics with Unsupervised Domain Adaptation (UDA) and Self-Supervised Learning.

1. **Physics-Informed Preprocessing:** We converted raw static values into dynamic features. We calculated Correlator Ratios (EC/PC, LC/PC) to isolate signal wave geometry distortion, and Temporal Derivatives ($\Delta$ Phase, $\Delta$ Doppler) to capture impossible physical kinematics. 
2. **Temporal Windowing:** Data was grouped by PRN and parsed into 10-timestep sliding windows, allowing the network to observe "slow-drift" spoofing attacks over time.
3. **1D-ResNet Backbone:** We utilized a 1D-Convolutional Neural Network with residual connections. Unlike LSTMs, 1D Convolutions act as mathematical derivative calculators across time, effectively compressing the 11 physical features into a dense 128-dimensional latent vector.
4. **Unsupervised Domain Adaptation (The GRL):** We attached a Domain Discriminator featuring a Gradient Reversal Layer (GRL). By feeding both labeled training data and unlabeled test data simultaneously, the GRL mathematically penalized the backbone if it could distinguish between the two receivers. This forced the network to erase hardware noise and learn universal signal physics.
5. **Contrastive Learning:** To build resilience against natural atmospheric interference, we applied weak (Gaussian jitter) and strong (variance scaling) augmentations, utilizing Contrastive Loss to force the model to recognize that both distorted signals represented the identical underlying physics.

**Quantitative Results & Learnings:**
* **Weighted F1 Score:** ~0.7570 (Converged at Epoch 15)
* **Conclusion:** While an F1 of 0.7570 initially appeared promising, critical analysis led us to pivot away from this model. We realized that this computationally heavy Domain-Adversarial architecture did not significantly outperform a properly tuned baseline. The Isolation Forest's low recall was simply an artifact of a rigid `contamination` mismatch, not a failure of our physical logic. Ultimately, the DANN proved too computationally expensive for the marginal improvements it offered.

> *Note: This approach shows strong theoretical potential, however, further hyperparameter tuning and scaling were limited due to strict hackathon time constraints.*

### Approach 3: Deep Bidirectional LSTM with External Data Synthesis (Winning Model)
**Objective:** To shatter the performance ceiling encountered in previous iterations by feeding a high-capacity sequential model with a massive, highly diverse, and perfectly structured dataset.

**Methodology:** We recognized that our primary bottleneck was no longer the model architecture, but the sheer volume and structure of the training data. We executed a two-part strategy:

1. **3D-to-2D Data Flattening:** We sourced a comprehensive external [GNSS spoofing dataset](https://data.mendeley.com/datasets/z7dj3yyzt8/3). Originally, this data was 3-dimensional (with satellite channels separated into isolated matrices). We engineered a transformation pipeline to "flatten" this 3D data into a 2D tabular format perfectly mirroring the schema of the NyneOS test dataset. We appended a strict binary `Output` column (`0` for normal, `1` for any level of spoofing), providing our model with an incredibly rich, labeled environment matching our target deployment.
2. **Bidirectional LSTM Architecture:** We fed this scaled data into a Deep Bidirectional LSTM network. 
    * **Architecture:** Two stacked BiLSTM layers (64 units $\rightarrow$ 32 units) designed to capture the deep, non-linear temporal relationships between correlator outputs and kinematic features. 
    * **Regularization:** We aggressively applied Dropout layers (`0.3`) between the LSTM blocks and utilized `EarlyStopping` (patience=5) monitoring the validation loss to guarantee the model would not overfit to the training set.
    * **Classification Head:** A dense layer (16 units) feeding into a final Sigmoid activation for binary classification.

**Quantitative Results & Learnings:**
* **Validation F1 Score:** **0.9726**
* **Conclusion:** This was the undisputed breakthrough of the project. By successfully mapping external channel-level data to our exact test-set domain and passing it through a high-capacity BiLSTM, we achieved an exceptional F1 score of >0.97. The Bidirectional nature of the network allowed it to look forward and backward across the time-series window, effortlessly distinguishing between clean physical noise and the subtle, unnatural feature distributions of a spoofing attack. 

> *Note: This is currently our best-performing model, although the project still offers substantial opportunities for further research and optimization.*

---
## 4. Model Comparison & Evaluation Metrics

To track our progress and justify architectural pivots, we evaluated each model against F1 metrics. 

| Model / Approach | Validation F1-Score |
| :--- | :---: |
| **Approach 1:** Isolation Forest | *N/A (Evaluated via Recall)* |
| **Approach 2:** DANN (1D-ResNet) | **0.7570** |
| **Approach 3:** Deep BiLSTM *(Final)* | **0.9726** |

---
## 5. Literature & Research Acknowledgment

Our feature engineering pipeline and architectural pivots were heavily guided by recent advancements in GNSS cybersecurity research. We specifically adapted methodologies from the following papers to build our defense system:

**1. Machine Learning Modeling of GPS Features with Applications to UAV Location Spoofing Detection and Classification**
> *Nayfeh, M., et al. (Computers & Security, 2023)*

**2. Exploring Multi-Channel GPS Receivers for Detecting Spoofing Attacks on UAVs Using Machine Learning**
> *Mouzai, M., et al. (Sensors, 2025)*

**3. Comparative Analysis of Deep Learning-Based Anomaly Detection Models for GPS Spoofing Detection**
> *Mirzakhaninafchi, H. (South Dakota State University, 2024)*

