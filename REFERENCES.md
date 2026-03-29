# Research Reference List

This document lists the foundational research papers and methodologies used in the development of the **Epidemic Spread Prediction** project. The project's architecture, specifically the **Hybrid Physics-Informed Attention-SEIR-LSTM**, is rooted in the following peer-reviewed works and preprints:

## 1. Core Model Architecture (Hybrid SEIR-LSTM)
*   **She, B., Xin, L., Paré, P. E., & Hale, M. (2023).** 
    *   *Title*: "Modeling and Predicting Epidemic Spread: A Gaussian Process Regression Approach"
    *   *Link*: [arXiv:2312.09384](https://arxiv.org/abs/2312.09384)
    *   *Contribution*: Provides the methodology for predicting infection rates using log-difference transformations and a Gaussian Process framework for uncertainty estimation, integrated here into the LSTM's output head.

*   **Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).**
    *   *Title*: "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations."
    *   *Publication*: Journal of Computational Physics.
    *   *Contribution*: Foundational methodology for "Physics-Informed" neural networks, which allows our model to respect the biological constraints of the SEIR (Susceptible-Exposed-Infectious-Recovered) dynamic equations.

## 2. Transmission & Network Dynamics
*   **Mercier, A. (2021).**
    *   *Title*: "Contagion-Preserving Network Sparsifiers: Exploring Epidemic Edge Importance Utilizing Effective Resistance"
    *   *Link*: [arXiv:2101.11818](https://arxiv.org/abs/2101.11818)
    *   *Contribution*: Introduces the concept of "Effective Resistance" as a metric for epidemic transmission importance. This informed the weighted node importance and transmission probability calculations in the model's spatial module.

*   **Funk, S., Gilad, E., Watkins, C., & Jansen, V. A. A. (2009).**
    *   *Title*: "The spread of awareness and its impact on epidemic outbreaks."
    *   *Publication*: PNAS (Proceedings of the National Academy of Sciences).
    *   *Contribution*: Theoretical basis for the **Awareness Response** mechanism used in `seir_lstm.py`. The model dynamically reduces the transmission rate ($\beta$) as public awareness (measured by rising infection counts) increases.

## 3. Stochastic Modeling & Inference
*   **Britton, T., & Pardoux, E. (Eds.) (2020).**
    *   *Title*: "Stochastic Epidemic Models with Inference"
    *   *Link*: [arXiv:1808.05350](https://arxiv.org/abs/1808.05350)
    *   *Contribution*: Provides the rigorous statistical framework for stochastic SEIR modeling, foundational for the model's likelihood functions and parameter bounds.

*   **Bracher, J., & Held, L. (2020).**
    *   *Title*: "Statistical modeling of infectious disease surveillance data"
    *   *Link*: [arXiv:1901.03090](https://arxiv.org/abs/1901.03090)
    *   *Contribution*: Influenced the implementation of multi-scale feature engineering, specifically the 30-day adaptive Gaussian smoothing and weighted lag features in `src/data/preprocess.py`.

## 4. Forecasting & Uncertainty
*   **Allen, L. J. S., et al. (2021).**
    *   *Title*: "Stochastic Models for the Spread of Infectious Diseases"
    *   *Link*: [arXiv:2107.03334](https://arxiv.org/abs/2107.03334)
    *   *Contribution*: Refines the branching process approximations used for the early-stage "Quick-Start" estimation and stochastic extinction risk analysis.

---
> [!NOTE]
> Detailed summaries of these papers can be found in the `research_papers/` directory within this repository.
