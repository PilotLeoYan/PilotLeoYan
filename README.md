<hr style="height:2px;border-width:0;color:gray;background-color:gray">

# <p align="center">White-Box Deep Learning & Computational Systems</p>
### <p align="center">Mathematical Foundations, Empirical Falsifiability, and Reproducible Engineering</p>

<p align="center">
  <b>Leonardo Fabyan Ortega Rivera</b><br>
  <sub>Department of Computer Science · Universidad de Guadalajara (CUCEI)</sub><br>
  <sub>Guadalajara, Jalisco, México</sub><br><br>
  <a href="mailto:leofabyano@gmail.com"><img src="https://img.shields.io/badge/Email-leofabyano%40gmail.com-blue?style=flat-square&logo=gmail" alt="Email"></a>
  <a href="https://www.linkedin.com/in/leonardo-fabyan-ortega-rivera-8b0460346/"><img src="https://img.shields.io/badge/LinkedIn-Leonardo_Ortega-0077B5?style=flat-square&logo=linkedin" alt="LinkedIn"></a>
  <a href="https://orcid.org/0009-0004-0497-2808"><img src="https://img.shields.io/badge/ORCID-0009--0004--0497--2808-A6CE39?style=flat-square&logo=orcid" alt="ORCID"></a>
  <a href="https://codeforces.com/profile/LeoYan955"><img src="https://img.shields.io/badge/Codeforces-LeoYan955-1F8ACB?style=flat-square&logo=codeforces" alt="Codeforces"></a>
  <a href="cv-harvard-eng-v.pdf"><img src="https://img.shields.io/badge/Curriculum_Vitae-PDF-red?style=flat-square&logo=adobeacrobatreader" alt="Curriculum Vitae (PDF)"></a>
</p>

<hr style="height:2px;border-width:0;color:gray;background-color:gray">

> **Abstract** — *I am a Computer Engineering graduate candidate at Universidad de Guadalajara (100% coursework completed, GPA: 91.42/100) specializing in deep learning mathematical foundations, spatiotemporal modeling, and robust computational systems. My work is anchored in **white-box engineering**: understanding and deriving neural architectures from mathematical first principles—such as analytical Jacobians via tensor contractions (`torch.einsum`) without automated differentiation (0.00% MAPE vs. native PyTorch)—auditing and reproducing peer-reviewed scientific literature (uncovering published equation errata in Elsevier 2024), and enforcing rigorous software craftsmanship (strict static typing with Mypy, deterministic environments with Astral uv, and automated security CI/CD). Actively seeking early-career opportunities as a **Junior Machine Learning Engineer**, **Computer Vision Engineer**, or **Systems / Python Developer**.*

---

## 1. Selected Research & Engineering Contributions

### 🧠 [Inside Deep Learning](https://github.com/PilotLeoYan/inside-deep-learning)
*White-Box Deep Learning and Analytical Gradients from First Principles*  
[![Repo](https://img.shields.io/badge/GitHub-inside--deep--learning-black?style=flat-square&logo=github)](https://github.com/PilotLeoYan/inside-deep-learning)
[![PyTorch](https://img.shields.io/badge/PyTorch-No_Autograd-EE4C2C?style=flat-square&logo=pytorch)](https://github.com/PilotLeoYan/inside-deep-learning)
[![MAPE](https://img.shields.io/badge/MAPE-0.00%25-brightgreen?style=flat-square)](https://github.com/PilotLeoYan/inside-deep-learning)

- **First-Principles Derivation:** Derived and implemented backpropagation, multi-dimensional tensor transformations, and exact analytical Jacobian derivatives without relying on `torch.autograd.backward()`.
- **Einstein Summation:** Leveraged `torch.einsum` for tensor contractions and vector-Jacobian products (VJPs), achieving **0.00% Mean Absolute Percentage Error (MAPE)** verified numerically against native PyTorch functional layers:
  $$\text{MAPE} = \frac{1}{n} \sum_{i=1}^n \left| \frac{y_{\text{analytic}} - y_{\text{autograd}}}{y_{\text{autograd}}} \right| = 0.00\%$$
- **Pedagogical Rigor:** Authored companion mathematical derivations in MyST Markdown documenting computational graph topologies, gradient flow, and memory layouts.

---

### 🔬 [Temporal Heterogeneity in Cognitive Architectures](https://github.com/PilotLeoYan/Temporal-Heterogeneity-in-Cognitive-Architectures)
*Scientific Reproduction & Mathematical Audit — Elsevier (Cognitive Systems Research)*  
[![Repo](https://img.shields.io/badge/GitHub-Temporal--Heterogeneity-black?style=flat-square&logo=github)](https://github.com/PilotLeoYan/Temporal-Heterogeneity-in-Cognitive-Architectures)
[![Audit](https://img.shields.io/badge/Audit-typos.md-orange?style=flat-square)](https://github.com/PilotLeoYan/Temporal-Heterogeneity-in-Cognitive-Architectures)
[![Ablations](https://img.shields.io/badge/Ablations-12_Runs_%7E10.6h-blue?style=flat-square)](https://github.com/PilotLeoYan/Temporal-Heterogeneity-in-Cognitive-Architectures)

- **Formal Replication:** Independently reproduced the evolutionary multi-agent model of Sandoval-Arrayga et al. (*Cognitive Systems Research*, Elsevier 2024), confirming that multiscale temporal processing ($a=3$) yields up to a **+151.8% fitness advantage** over synchronous baselines across 4,500 generations.
- **Mathematical Audit & Errata:** Uncovered critical arithmetic inconsistencies in the published publication equations (e.g., $0.0833\%$ vs. $0.833\%$), documented formally in [`typos.md`](https://github.com/PilotLeoYan/Temporal-Heterogeneity-in-Cognitive-Architectures/blob/main/typos.md).
- **Empirical Rigor:** Designed and executed a 12-run parallel ablation suite (~10.6 hours compute) verifying fitness distributions, behavioral convergence, and evolutionary stability.

---

### 🎥 [Recurrent Convolutional Neural Networks](https://github.com/PilotLeoYan/Recurrent-Convolution-NN)
*Spatiotemporal Modeling & Multi-Frame Video Prediction*  
[![Repo](https://img.shields.io/badge/GitHub-Recurrent--Convolution--NN-black?style=flat-square&logo=github)](https://github.com/PilotLeoYan/Recurrent-Convolution-NN)
[![Tensors](https://img.shields.io/badge/Tensors-5D_(T%2CB%2CC%2CH%2CW)-blueviolet?style=flat-square)](https://github.com/PilotLeoYan/Recurrent-Convolution-NN)
[![Loss](https://img.shields.io/badge/Loss-MSE_%2B_SSIM-informational?style=flat-square)](https://github.com/PilotLeoYan/Recurrent-Convolution-NN)

- **Architecture Benchmarking:** Engineered and evaluated spatiotemporal architectures (`RCNN2d`, `Conv2dGRU`, and non-recurrent baseline CNN) for sequence-to-sequence video prediction on Moving MNIST using 5D tensors $(T, B, C, H, W)$.
- **Multi-Objective Perceptual Loss:** Formulated a composite loss function balancing pixel-level error (MSE) and structural perceptual coherence (SSIM) with autoregressive scheduled sampling:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \lambda \, (1 - \text{SSIM})$$
- **Recurrence Analysis:** Analyzed memory retention vs. gradient attenuation in recurrent convolutions across variable rollout horizons.

---

### 🚀 [RecycleNet MLOps Pipeline](https://github.com/PilotLeoYan/RecycleNet-K8s)
*Production-Grade Computer Vision & Automated CI/CD Lifecycle*  
[![Repo](https://img.shields.io/badge/GitHub-RecycleNet--K8s-black?style=flat-square&logo=github)](https://github.com/PilotLeoYan/RecycleNet-K8s)
[![Packaging](https://img.shields.io/badge/Packaging-Astral_uv-blue?style=flat-square)](https://github.com/PilotLeoYan/RecycleNet-K8s)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?style=flat-square&logo=githubactions)](https://github.com/PilotLeoYan/RecycleNet-K8s)
[![Quality](https://img.shields.io/badge/Security-Bandit_SAST-red?style=flat-square)](https://github.com/PilotLeoYan/RecycleNet-K8s)

- **Vision Backbone:** Transfer learning with MobileNetV3 for 6-class solid waste sorting, integrating deterministic CUDA seeding and multi-class One-vs-Rest ROC-AUC evaluation.
- **Modern Python Tooling:** Managed strictly through Astral `uv` for reproducible virtual environments and lightning-fast dependency resolution.
- **Engineering Craftsmanship:** Automated CI/CD pipelines in GitHub Actions enforcing strict static typing with `mypy`, formatting/linting with `ruff`, and automated security vulnerability auditing with `bandit`.

---

### ⚡ [LeoYan Solutions](https://github.com/PilotLeoYan/leoyan-solutions)
*Algorithmic Problem Solving & Competitive Programming*  
[![Repo](https://img.shields.io/badge/GitHub-leoyan--solutions-black?style=flat-square&logo=github)](https://github.com/PilotLeoYan/leoyan-solutions)
[![Codeforces](https://img.shields.io/badge/Platform-Codeforces-1F8ACB?style=flat-square&logo=codeforces)](https://codeforces.com/profile/LeoYan955)
[![C++17](https://img.shields.io/badge/Language-C%2B%2B17-00599C?style=flat-square&logo=c%2B%2B)](https://github.com/PilotLeoYan/leoyan-solutions)

- **Algorithmic Base:** Rigorous problem solving in **C++17** and Python focusing on core data structures, greedy approaches, dynamic programming, binary search, and graph traversals.
- **Structured Explanations:** Every problem is documented with step-by-step reasoning, asymptotic complexity analysis ($\mathcal{O}(N \log N)$ time, $\mathcal{O}(1)$ space), and edge-case validation.

---

## 2. Methodological & Technical Arsenal

```
┌──────────────────────────────────┬──────────────────────────────────┐
│ Category                         │ Technologies & Competencies      │
├──────────────────────────────────┼──────────────────────────────────┤
│ 💻 Core Languages                │ Python 3, C++ (C++17 Algorithms) │
│                                  │ SQL, POSIX Bash                  │
├──────────────────────────────────┼──────────────────────────────────┤
│ 📐 Deep Learning & Numerics      │ PyTorch, Tensor Algebra (einsum) │
│                                  │ NumPy, Computer Vision, Optuna   │
│                                  │ Scikit-Learn, PINN Formulation   │
├──────────────────────────────────┼──────────────────────────────────┤
│ 🛠️ Software Craftsmanship       │ Astral uv, Strict Mypy, Ruff     │
│    & MLOps                       │ GitHub Actions CI/CD, Git        │
│                                  │ Bandit (SAST), Linux (Debian)    │
├──────────────────────────────────┼──────────────────────────────────┤
│ 🗄️ Databases & Systems           │ SQLite3 (WAL Concurrency), MySQL │
│                                  │ Relational Modeling, LAN Systems │
└──────────────────────────────────┴──────────────────────────────────┘
```

---

## 3. Academic Milestones & Research Affiliations

- **🎓 B.S. in Computer Engineering — Universidad de Guadalajara (CUCEI)**
  - Coursework: **100% completed** · GPA: **91.42 / 100** (Academic standing in high percentiles).
  - Specialization: Machine Learning, Numerical Methods, and Systems Programming.
- **🛰️ Scientific Research Exchange — Programa DELFÍN (Colombia)**
  - *Institución Universitaria de Envigado (2025)*: Architected the relational schema (OUL-DB) to catalog multi-spectral FITS astronomical images and satellite telemetry.
- **🌌 Latin Lunar Lobby Congress — Universidad Complutense de Madrid**
  - Speaker on *Dynamic NeRFs for Exospheric Reconstruction*: 4D temporal parameterization of non-rigid diurnal particle density shifts for the OUL satellite mission.
- **💼 Industry Experience — Power Industrial (Software & Systems Developer)**
  - Engineered warehouse inventory management software handling 3,600+ industrial SKUs with Python and SQLite3 (WAL mode) across multi-workstation local area networks.

---

## 4. Citation

If you find these implementations or mathematical notes useful in your work or research, please consider citing:

```bibtex
@misc{ortega2026whitebox,
  author       = {Ortega Rivera, Leonardo Fabyan},
  title        = {White-Box Deep Learning and Computational Systems: Mathematical Foundations, Empirical Falsifiability, and Reproducible Engineering},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub Profile Repository},
  howpublished = {\url{https://github.com/PilotLeoYan}}
}
```

---

<p align="center">
  <sub>Open to technical discussions, collaborations, and engineering roles:</sub><br>
  <b><a href="mailto:leofabyano@gmail.com">leofabyano@gmail.com</a></b> · <b><a href="https://www.linkedin.com/in/leonardo-fabyan-ortega-rivera-8b0460346/">LinkedIn</a></b> · <b><a href="cv-harvard-eng-v.pdf">Curriculum Vitae (PDF)</a></b>
</p>
