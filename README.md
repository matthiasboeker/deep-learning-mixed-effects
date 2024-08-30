# Neural and Linear Mixed Effects Models

This repository contains code for exploring the efficacy of both neural network-based and traditional linear mixed effects models in estimating parameters and their covariance matrices. The primary goal is to compare these advanced neural network models against more conventional linear models on simulated datasets.

## Installation

To set up the necessary environment:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. Install the dependencies using Conda:
    ```bash
    conda env create -f environment.yml
    conda activate your-env-name
    ```

## Overview of the Code

The code is structured as follows:

- **Neural Network Models**: Implementations of neural networks with and without random effects to capture the variability in the data linked to specific groups or conditions.
- **Linear Mixed Effects Model**: Utilization of a custom-built model fitting routine that leverages negative log-likelihood for parameter estimation.
- **Training and Evaluation**: Scripts to train these models and evaluate their performance in terms of mean squared error and R-squared values.
- **Visualization**: Code to plot predictions against actual outcomes to visually assess model performance.

The main script can be executed by running:
```bash
python base_comparison_variance_estimate.py
```

## Key Results and Observations

The following key results and observations were noted during model experiments:

- **Parameter Estimation**:

    - The neural network with random effects tends to overestimate the first parameter of the random effects (b1) while closely estimating the second (b2). This issue persists across multiple runs.
    - The linear model provides estimates closer to the ground truth for both parameters, suggesting better stability in simpler model structures for this particular task.

- **Covariance Matrix Estimation:**

    - Both the linear and neural network model struggles to accurately estimate the covariance matrix
    G compared to the linear model, particularly in capturing off-diagonal elements which represent the correlations between random effects.

## Considerations and Challenges

Scale and Magnitude of Parameters: Differences in the scale and magnitude of parameters could be affecting the neural network's ability to learn effectively. Normalization or standardization of input features might help mitigate this issue.
Optimization and Stability: The choice of optimization algorithm, learning rate, and regularization techniques (like dropout and L2 regularization) critically impacts the training dynamics and the resulting model accuracy.

## Contributors

Matthias Boeker