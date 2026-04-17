# GNSS Spoofing Detection: AI-Driven Defense Architecture
**Team:** [Your Team Name] | **Event:** NyneOS x IIT Roorkee Technical General Championship 2026

## 📌 1. Project Overview
This repository contains our team's Round 1 submission for the AI-driven GNSS Spoofing Detection problem statement. Recognizing the complexity of multi-level spoofing (from crude jamming to covert trajectory drifting), we took an iterative, multi-model approach. We began with strict, unsupervised physical heuristics and progressively advanced to our final model to maximize our Weighted F1 Score.

---

## 🔬 2. Our Iterative Approaches

### Approach 1: Physics-Informed Unsupervised Anomaly Detection (Isolation Forest)
**Objective:** To detect spoofing using purely first-principle reasoning without relying on labeled training data.
**Methodology:** We hypothesized that while a spoofer can fake a location, they struggle to fake flawless temporal physics. We engineered true rate-of-change ($\Delta$) features per PRN (Satellite) for kinematics ($\Delta$ Doppler, $\Delta$ Pseudorange) and signal power ($\Delta$ CN0). We then fed these physical deltas into an **Isolation Forest** and aggregated the anomaly scores to a receiver-level probability using fractional voting.

**Quantitative Results & Learnings:**

The results depended on an external database with labels given to verify how the model worked, and thus it was dependant on the amount of anomalies in the data which made the tuning of parameters very difficult for the general test dataset provided.

* **Normal Data (Level 0):** 99.3% correctly ignored (Extremely low False Positive Rate).
* **Crude Attacks (Level 1):** ~16% detection rate.
* **Covert Attacks (Level 2 & 3):** < 1% detection rate.
* **Conclusion:** While highly precise on normal data, the unsupervised model was far too conservative. It successfully caught violent physical jumps but was completely blind to the sophisticated, slow-drifting attacks. This justified our pivot to other approaches to improve recall. The architecture's bad performance is most likely the result of a contamination hyperparameter mismatch between the validation dataset and the test dataset.

### Approach 2: Domain-Adversarial Neural Network (DANN) with 1D-ResNet
**Objective:** To solve the "Domain Gap" problem. Lab-trained models often fail in the real world because different GNSS hardware receivers have different internal biases. This approach aimed to build a hardware-agnostic model capable of detecting complex time-series attacks on the unseen NyneOS test drone.

**Methodology:** We built a robust Deep Learning pipeline that fused physical heuristics with Unsupervised Domain Adaptation (UDA) and Self-Supervised Learning.

1. **Physics-Informed Preprocessing:** We converted raw static values into dynamics. We calculated Correlator Ratios (EC/PC, LC/PC) to isolate signal wave geometry distortion, and calculated Temporal Derivatives ($\Delta$ Phase, $\Delta$ Doppler) to capture impossible physical kinematics. 
2. **Temporal Windowing:** Data was grouped by PRN and parsed into 10-timestep sliding windows to allow the network to "see" slow-drift spoofing attacks over time.
3. **1D-ResNet Backbone:** We utilized a 1D-Convolutional Neural Network with residual connections. Unlike LSTMs, 1D Convolutions act as mathematical derivative calculators across time, compressing the 11 physical features into a dense 128-dimensional latent vector.
4. **Unsupervised Domain Adaptation (The GRL):** We attached a Domain Discriminator with a Gradient Reversal Layer (GRL). By feeding both labeled training data and unlabeled test data simultaneously, the GRL mathematically penalized the backbone if it could tell the difference between the two receivers. This forced the network to erase hardware noise and learn pure, universal signal physics.
5. **Contrastive Learning:** To build resilience against natural atmospheric interference, we applied weak (Gaussian jitter) and strong (variance scaling) augmentations, using Contrastive Loss to force the model to recognize that both distorted signals represent the same underlying physics.


**Quantitative Results & Learnings:**
* **Weighted F1 Score:** ~0.7570 (Converged at Epoch 15)
* **Conclusion:** While an F1 of 0.7570 initially appeared to be an upgrade, critical analysis led us to reject this model. We realized that this heavy Domain-Adversarial architecture wasn't actually superior to our baseline. Our Isolation Forest's low recall was simply an artifact of a rigid `contamination` hyperparameter mismatch, not a failure of our physical logic. Ultimately, the DANN proved to be computationally expensive and offered only debatable improvements over a properly tuned unsupervised model. This realization pushed us to rethink our architecture entirely for our final, definitive approach.

### Approach 3: Deep Bidirectional LSTM with External Data Synthesis (Winning Model)
**Objective:** To shatter the performance ceiling encountered in previous approaches by feeding a high-capacity sequential model with a massive, highly diverse, and perfectly structured dataset.

**Methodology:** We realized that our bottleneck was no longer just the model architecture, but the sheer volume and structure of the training data. We executed a two-part strategy:

1. **3D-to-2D Data Flattening (The Breakthrough):** We sourced an expansive external GNSS spoofing dataset. Originally, this data was 3-dimensional (with all satellite channels separated into isolated matrices). We engineered a transformation pipeline to "flatten" this 3D data into a 2D tabular format that exactly mirrored the schema of the NyneOS test dataset. We appended a strict binary `Output` column (`0` for normal, `1` for any level of spoofing). This gave our model an incredibly rich, labeled environment to learn from that perfectly matched our target deployment.
2. **Bidirectional LSTM Architecture:** We fed this scaled data into a Deep Bidirectional LSTM network. 
    * **Architecture:** Two stacked BiLSTM layers (64 units $\rightarrow$ 32 units) designed to capture the deep, non-linear relationships between the correlator outputs and kinematic features. 
    * **Regularization:** We aggressively applied Dropout layers (`0.3`) between the LSTM blocks and utilized `EarlyStopping` (patience=5) monitoring the validation loss to guarantee the model would not overfit to the training set.
    * **Classification Head:** A dense layer (16 units) feeding into a final Sigmoid activation for binary classification.

**Quantitative Results & Learnings:**
* **Validation F1 Score:** **0.9726**
* **Conclusion:** This was the undisputed breakthrough of the project. By successfully mapping external channel-level data to our exact test-set domain and passing it through a high-capacity BiLSTM, we achieved an exceptional F1 score of >0.97. The Bidirectional nature of the network allowed it to effortlessly distinguish between clean physical noise and the subtle, unnatural feature distributions of a spoofing attack. 