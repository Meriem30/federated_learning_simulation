# Federated Learning Simulation Framework
### Graph-Based Client Selection with Information Theory and Spectral Clustering
[![Python 3.8+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-BSD-green.svg)](LICENSE)
[![Paper Status](https://img.shields.io/badge/IEEE%20IoT%20Journal-Under%20Review-yellow.svg)](https://ieee-iotj.org/)

> **A comprehensive simulation framework for federated learning with advanced client selection strategy based on graph modeling of FL system, information theory, and spectral clustering techniques.**

---


## 📋 Table of Contents
- [X] [Overview](#-overview) 
- [X] [Key Features](#-key-features)
- [X] [Research Context](#-research-context)
- [X] [Architecture & Project Structure](#-architecture-and-project-structure)
- [X] [Installation](#-installation) 
- [X] [Configuration](#-configuration)
- [X] [Quick Start](#quick-start)
    ### TODOs
- [ ] [Detailed Component Guide](#detailed-component-guide)
  - [ ] [Core Library: `federated_learning_simulation_lib`](#1-core-library-federated_learning_simulation_lib)
  - [ ] [Simulation Engine: `federated_learning_simulator`](#2-simulation-engine-federated_learning_simulator)
  - [ ] [Vision Toolkit: `torch_vision`](#3-vision-toolkit-torch_vision)
  - [ ] [Medical Imaging: `torch_medical`](#4-medical-imaging-torch_medical-optional)
  - [ ] [Utility Libraries: `other_libs` & `torch_kit`](#5-utility-libraries-other_libs--torch_kit)
- [ ] [Experimental Workflows](#experimental-workflows)
- [ ] [Results & Visualization](#results--visualization)
- [ ] [Citation](#citation)
- [ ] [Contributing](#contributing)
- [ ] [License](#license)
- [ ] [Contact](#contact)

---

## 🎯 Overview

This repository implements a **state-of-the-art federated learning simulation framework** designed for investigating **intelligent client selection strategies** in heterogeneous federated learning IoT environments. 
Our framework introduces a novel **graph-based client selection mechanism** that combines **information-theoretic similarity measures** with **spectral clustering** to optimize client participation, reduce communication overhead, and improve model convergence.



---

[//]: # ( ### What Makes This Framework Original?)



[//]: # ( - **🔗 Graph-Based FL Modeling**: Represents FL systems as graphs where nodes are clients and edges capture similarity/interaction patterns)
[//]: # (- **📊 Information Theory Integration**: Uses entropy, mutual information, and divergence measures for client selection)
[//]: # (- **🎯 Spectral Clustering**: Applies spectral methods to identify coherent client clusters for efficient federated rounds)
[//]: # (- **🔬 Modular Research Design**: Clean separation between simulation engine, data handling, and model architectures)
[//]: # (- **📈 Comprehensive Evaluation**: Built-in metrics for convergence, communication efficiency, and fairness analysis)


---


## ✨ Key Features

### Core Capabilities

- ✅ **Advanced Client Selection Algorithms**
  - ... *(to be added)*

- ✅ **Flexible Simulation Environment**
  - Configurable network topologies
  - Flexible model architecture: Pytorch, HuggingFace, and custom models
  - Advanced hyperparameters fine-tuning
  - Configurable heterogeneous data distributions (IID/Non-IID)
  - Comprehensive result tracking saved as JSON files.
  - Realistic failure and dropout modeling
  - All to be specified in *yaml* configuration files

- ✅ **Multiple Application Domains**
  - Research benchmark Computer Vision datasets (CIFAR-10, CIFAR-100, ImageNet subsets)
  - Custom dataset (medical imaging) integration support

- ✅ **Comprehensive Metrics & Logging**
  - Multiple log levels (INFO, WARNING, DEBUG) with flexible log saving options.
  - Model accuracy and loss tracking + F1 score, AUC and other metrics if specified.
  - Client participation fairness metrics
  - Convergence speed evaluation (rounds to target accuracy)


---


## 🔬 Research Context

This framework supports the research presented in our paper submitted to **IEEE Internet of Things Journal**:

**"GRAIL-FL: Graph-based Adaptive Information Learning for Efficient Client Selection in Federated IoT Networks"**

### Research Objectives
>... *(to be added)*
>1. **Problem**:
>2. **Solution**:
>3. **Benefits**:

---


## 🏗️ Architecture and Project Structure

The framework is organized into **six modular sub-projects**, each with a specific role in the FL simulation pipeline:

```
federated_learning_simulation/
│
├── federated_learning_simulation_lib/    # ⭐ Core FL algorithms & graph methods
├── federated_learning_simulator/         # ⭐ Main simulation orchestration
├── torch_vision/                         # ⭐ Vision tasks implementation
├── torch_medical/                        # 🏥 Medical imaging support
├── torch_kit/                            # 🔧 PyTorch helper functions
└── other_libs/                           # 🛠️ Shared utilities
```

### Component Overview

| Component                                                                                                                                         | Purpose                                                               | Key Features                                                                          |
|---------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| [**`federated_learning_simulation_lib`**](https://github.com/Meriem30/federated_learning_simulation/tree/main/federated_learning_simulation_lib)  | Core FL library with graph-based client selection                     | Worker, GraphWorker, Sampler, Server; Aggregation, Selection and Clustering Algorithms |
| [**`federated_learning_simulator`**](https://github.com/Meriem30/federated_learning_simulation/tree/main/federated_learning_simulator)            | Orchestration engine for running FL experiments (Project Entry Point) | Experiment Configuration, Client and Server Endpoints, Aggregation Method Assignment  | 
| [**`torch_vision`**](https://github.com/Meriem30/federated_learning_simulation/tree/main/torch_vision)                                            | Computer Vision FL Implementations                                    | Dataset Constructors, Transformer Pipelines, Vision Model Constructors                | 
| [**`torch_medical`**](https://github.com/Meriem30/federated_learning_simulation/tree/main/torch_medical)                                          | Medical Imaging Specialization                                        | Medical Dataset Registration, Domain-Specific Transformers, Models                    | 
| [**`torch_kit`**](https://github.com/Meriem30/federated_learning_simulation/tree/main/torch_kit)                                                  | Advanced PyTorch Utility Toolkit (Extensions)                         | Design Abstractions (Training/Optimizers/Metrics), Device Management, Custom layers   | 
| [**`other_libs`**](https://github.com/Meriem30/federated_learning_simulation/tree/main/other_libs)                                                | Helper Utilities to Simplify Python Tasks                             | Data loaders, Log Formatting, Communication Abstractions                              | 

---

## 📦 Installation

### Prerequisites

- **Python 3.11+** (tested with 3.11.5)
- **CUDA 12.1+** (for GPU support, recommended) or CPU-only mode
- **16GB+ RAM** recommended for large-scale simulations
- **Operating Systems**: Windows, Linux, macOS (all Unix distributions supported)

> **Note**: A pre-configured `environment.yaml` file is provided at the root of the repository with pinned package versions and minimum versions specified to prevent dependency conflicts. 
> This file is configured for **CUDA-enabled systems** and is the **recommended installation method**. Directives for CPU-only installation are provided below.
---

### Recommended Installation 

**For CUDA-enabled systems using conda env file:**

#### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/Meriem30/federated_learning_simulation.git
cd federated_learning_simulation
```

#### Step 2: Create Virtual Environment from `environment.yaml`

```bash
# Create environment (includes CUDA support)
conda env create -f environment.yaml

# Activate environment
conda activate FL_projects
```

#### Step 3: Verify Installation 

```bash
# Check CUDA availability and core installations

python -c "import torch, torchvision, torchmetrics; \
print(f'PyTorch version: {torch.__version__}'); \
print(f'TorchVision version: {torchvision.__version__}'); \
print(f'TorchMetrics version: {torchmetrics.__version__}'); \
print(f'CUDA available: {torch.cuda.is_available()}'); \
print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); \
print(f'CUDA device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}'); \
print(f'CUDA device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU mode only\"}')"
```


**Expected output:**

If you have GPU-enabled system, and depending on your setup, you would see something like the messages below:

```bash
PyTorch version: 2.3.1
TorchVision version: 0.18.1
TorchMetrics version: 1.3.0
CUDA available: True
CUDA version: 12.1
CUDA device count: 1
CUDA device name: NVIDIA GeForce RTX 3080 # (your nvidia device, otherwise you'll get: CPU mode only)
```

---

**For CPU-only systems:**

If you want to run on CPU without CUDA support, modify the `environment.yaml` file before creating the environment:

#### Step 2.1 Change the Channels Section
replace the channels in `environment.yaml` with:

```yaml
   channels:
     - anaconda
     - conda-forge
     - defaults
```

#### Step 2.2 Remove CUDA-related Dependencies
replace `pytorch-cuda=12.1` with the following

```yaml
dependencies:
  - ...
  - pytorch=2.3.1 #or >=2.1
```

#### Step 2.3 Create the Environment:
```bash
   conda env create -f environment.yaml
   conda activate FL_projects
```

Now you can do the sanity checks from *[step 3](#step-3:-verify-installation)*

---

#### Step 4 : Running Pre-configured Experiment
If you have made it this far, the next command should run successfully as a quick running experiment check


```bash
# Run a quick sanity-check experiment
python federated_learning_simulator/simulator.py \
    --config-name fed_avg/cifar10.yaml \
    ++fed_avg.round=1 \
    ++fed_avg.epoch=1 \
    ++fed_avg.worker_number=2 \
    ++fed_avg.algorithm_kwargs.node_sample_percent=1 \
    ++fed_avg.debug=True
 ```


## ⚙️ Configuration

The framework uses a **hierarchical configuration system** powered by [Hydra](https://hydra.cc/), enabling flexible experiment management through layered YAML files. Configuration follows a three-level architecture from global settings to algorithm-specific parameters.

---

### Configuration Architecture
```
federated_learning_simulator/
├── conf/
│   ├── global.yaml                    # Level I: Global/Fixed parameters
│   ├── fed_avg/                       # Level II: Algorithm-specific configs
│   │   ├── mnist.yaml
│   │   ├── cifar10.yaml
│   │   └── cifar100.yaml
│   ├── graph_fed_avg/                 # Level II: GRAIL-FL algorithm configs
│   │   ├── mnist.yaml                 # GRAIL-FL configuration for MNIST
│   │   ├── cifar10.yaml               # GRAIL-FL configuration for CIFAR-10
│   │   └── cifar100.yaml              # GRAIL-FL configuration for CIFAR-100
│   └── power_of_choice/               # Level II: Other algorithms...
│       ├── mnist.yaml
│       └── cifar10.yaml
├── log/                               # Generated logs (grouped by exp_name)
└── session/                           # Session checkpoints and JSON results
```

---

### Level I: Global Configuration (`conf/global.yaml`)

The global configuration file defines **experiment-wide settings** and **logging parameters** that remain consistent across different algorithm runs. Here is an example of how it could be configured:
```yaml
# federated_learning_simulator/conf/global.yaml

# Experiment identification
exp_name: cifar10_niid_0.3  # Groups logs/sessions for the same experimental setup

# Logging configuration
log_level: INFO  # Options: DEBUG, INFO, WARNING, ERROR
log_performance_metric: true

# For efficiency
cache_transforms: cpu
```

> **💡 Best Practice**: Keep `exp_name` identical when comparing different algorithms (e.g., FedAvg vs GRAIL-FL) on the same dataset/heterogeneity setting. This ensures logs and sessions are organized under the same experiment group for easy comparison.

---

### Level II: Algorithm-Specific Directories

Each federated learning algorithm has its dedicated directory containing dataset-specific configuration files.

**Available Algorithm Directories:**

| Directory | Algorithm | Description |
|-----------|-----------|-------------|
| `fed_avg/` | FedAvg | Baseline with random client sampling |
| `graph_fed_avg/` | **GRAIL-FL** | Our proposed graph-based selection (named `graph_fed_avg` in code) |
| `power_of_choice/` | Power-of-Choice | Loss-based client selection |
| *(others in development)* | - | Additional state-of-the-art methods |

> **📝 Note**: In our codebase, **GRAIL-FL** is referred to as `graph_fed_avg`. This naming predates the finalized terminology used in our paper. The identifier reflects that the algorithm retains the FedAvg aggregation mechanism while introducing graph-based client selection.
---

### Level III: Dataset-Specific Configuration Files

Within each algorithm directory, separate YAML files configure parameters for different datasets.

### GRAIL-FL Configuration Example

Below is a complete breakdown of parameters in `federated_learning_simulator/conf/graph_fed_avg/cifar10.yaml`:



#### **A. Distributed Learning Parameters**

Parameters controlling the federated learning simulation environment.

| Parameter                   | Value | Description | Possible Values |
|-----------------------------|-------|-------------|-----------------|
| **General**                 |
| `distributed_algorithm`     | `graph_fed_avg` | FL algorithm (GRAIL-FL) | `fed_avg`, `graph_fed_avg`, `power_of_choice` |
| `exp_name`                  | `c510` | Short experiment identifier | Any string (keep consistent with `global.yaml`) |
| `debug`                     | `false` | Enable debug mode | `true`, `false` |
| **Client Configuration**    |
| `worker_number`             | `65` | Total number of clients in federation | Any integer (10-1000+) |
| `node_sample_percent`       | `0.7` | Fraction of clients selected per round | `0.0` - `1.0` |
| `node_random_selection`     | `false` | Use random selection (overrides graph-based) | `true`, `false` |
| **Communication**           |
| `distribute_init_parameters` | `True` | Send initial model to all clients | `True`, `False` |
| `round`                     | `150` | Total federated learning rounds | Any integer (50-500) |

---

#### **B. Training Parameters**

Parameters for local client training and optimization.

| Parameter                       | Value               | Description | Possible Values |
|---------------------------------|---------------------|-------------|-----------------|
| **Local Training**              |
| `batch_size`                    | `64`                | Mini-batch size for local training | 16, 32, 64, 128, 256 |
| `epoch`                         | `3`                 | Local epochs per round | 1-10 |
| `learning_rate`                 | `0.00001`           | Initial learning rate | `1e-5` - `1e-1` |
| **Optimizer**                   |
| `optimizer_name`                | `AdamW` | Optimizer algorithm | `SGD`, `Adam`, `AdamW`, `RMSprop` |
| `optimizer_kwargs`              | (see below) | Optimizer-specific parameters | Varies by optimizer (follows PyTorch API) |
| **Learning Rate Scheduler**     |
| `learning_rate_scheduler_name`  | `ReduceLROnPlateau` | LR scheduler type | `CosineAnnealingLR`, `ReduceLROnPlateau`, `StepLR`, `MultiStepLR` |
| `learning_rate_scheduler_kwargs` | (see below) | Scheduler-specific parameters | Varies by scheduler (follows PyTorch API) |

See the next two subsections for details on `optimizer_kwargs` and `learning_rate_scheduler_kwargs` configuration.
___

<details> 

<summary><b>Optimizer-Specific Parameters</b></summary>

The parameters under `optimizer_kwargs` depend on the chosen `optimizer_name` and follow the **PyTorch optimizer API**. Below are examples for common optimizers:

<details>
<summary><b>AdamW</b> (used in example config)</summary>

```yaml
optimizer_kwargs:
  betas: [0.9, 0.99]      # Exponential decay rates for moment estimates
  eps: 1e-8               # Numerical stability epsilon
  weight_decay: 0.0001    # L2 regularization strength (decoupled)
```

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `betas` | Coefficients for computing running averages | `[0.9, 0.999]`, `[0.9, 0.99]` |
| `eps` | Term for numerical stability | `1e-8`, `1e-10` |
| `weight_decay` | Weight decay (L2 penalty) | `0.0` - `0.01` |

</details>

<details>

<summary><b>SGD</b></summary>

```yaml
optimizer_kwargs:
  momentum: 0.9           # Momentum factor
  weight_decay: 0.0001    # L2 regularization
```

</details>

<details>
<summary><b>Adam</b></summary>

```yaml
optimizer_kwargs:
  betas: [0.9, 0.999]     # Coefficients for moment estimates
  eps: 1e-8               # Numerical stability
  weight_decay: 0.0       # L2 regularization (coupled with gradient)
```

</details>

> **📘 Reference**: See [PyTorch Optimizers Documentation](https://pytorch.org/docs/stable/optim.html) for complete parameter lists for each optimizer.

----


</details>


<details> 


<summary><b>Scheduler-Specific Parameters</b></summary>


The parameters under `learning_rate_scheduler_kwargs` depend on the chosen `learning_rate_scheduler_name` and follow the **PyTorch scheduler API**.

<details>
<summary><b>ReduceLROnPlateau</b> (used in example config)</summary>

```yaml
learning_rate_scheduler_kwargs:
  mode: "max"              # Metric optimization direction
  factor: 0.5              # Factor by which LR is reduced
  patience: 5              # Rounds to wait before reducing LR
```


| Parameter | Description | Values                                 |
|-----------|-------------|----------------------------------------|
| `mode` | Whether to maximize or minimize monitored metric | `"max"` or `"min"`          |
| `factor` | Multiplicative factor of LR decrease | `0.1` - `0.9` (e.g., `0.5` = halve LR) |
| `patience` | Number of rounds with no improvement before reduction | `1` - `20`                             |


> **💡 Tip**: Use `mode: "max"` when tracking **accuracy** or F1-score, and `mode: "min"` when tracking **loss**.

</details>

<details>
<summary><b>CosineAnnealingLR</b></summary>

```yaml
learning_rate_scheduler_kwargs:
  T_max: 150              # Maximum number of iterations (typically = total rounds)
  eta_min: 1e-6           # Minimum learning rate
```

</details>

<details>
<summary><b>StepLR</b></summary>

```yaml
learning_rate_scheduler_kwargs:
  step_size: 30           # Period of LR decay
  gamma: 0.1              # Multiplicative factor of LR decay
```

</details>

<details>
<summary><b>MultiStepLR</b></summary>

```yaml
learning_rate_scheduler_kwargs:
  milestones: [50, 100]   # Rounds at which to reduce LR
  gamma: 0.1              # Multiplicative factor of LR decay
```

</details>

> **📘 Reference**: See [PyTorch LR Schedulers Documentation](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) for complete parameter lists for each scheduler.

---
</details>


#### **C. Model & Dataset Parameters**

| Parameter | Value | Description | Possible Values                                |
|-----------|-------|-------------|------------------------------------------------|
| **Model Configuration** |
| `model_name` | `densenet40` | Model architecture | `resnet18`, `densenet40`,  |
| `model_kwargs.num_classes` | `10` | Number of output classes | Dataset-dependent (10, 100,...)                |
| `model_kwargs.keep_model_cache` | `false` | Cache model states | `true`, `false`                                |
| **Dataset Configuration** |
| `dataset_name` | `CIFAR10` | Dataset identifier | `MNIST`, `CIFAR10`, `CIFAR100`, `ImageNet` |
| `dataset_kwargs.dataset_type` | `Vision` | Dataset category | `Vision`, `Medical`, `Text`                    |
| `dataset_kwargs.dataset_sampling` | `dirichlet_split` | Data partitioning method | `iid`, `dirichlet_split`, |
| `dataset_kwargs.classes` | `[0,1,2,3,4,5,6,7,8,9]` | Class labels to use | List of integers                               |
| **Sampling Configuration** |
| `dataset_sampling_kwargs.concentration` | `0.5` | Dirichlet concentration (α) | `0.1` (high heterogeneity) - `1.0` (near IID)  |

---

#### **D. GRAIL-FL Specific Parameters**

Parameters unique to our graph-based client selection algorithm.

##### **D.1 Graph Representation Parameters**

Controls how the client similarity graph is constructed.

| Parameter | Value | Description                                                   | Possible Values                                 |
|-----------|-------|---------------------------------------------------------------|-------------------------------------------------|
| `graph_worker` | `true` | Enable graph-based representation and selection               | `true`, `false`                                 |
| `graph_type` | `KNN` | Graph construction method                                     | `knn`, `mutual_knn`, `fully_connected`          |
| `num_neighbor` | `2` | Number of nearest neighbors (KNN)                             | 1-10                                            |
| `threshold` | `0.0` | Minimum similarity for constructing the edges                 | `0.0` - `1.0`                                   |
| `similarity_function` | `Gaussian` | Similarity function to be applied on client evaluation matrix | `gaussian`, `cosine`, `euclidean`, `customized` |




##### **D.2 Spectral Clustering Parameters**

Controls the spectral clustering for client grouping and selection.

| Parameter | Value | Description                          | Possible Values                           |
|-----------|-------|--------------------------------------|-------------------------------------------|
| `family_number` | `4` | Number of client clusters (families) | \>2                                       |
| `laplacian_type` | `RandomWalk` | Graph Laplacian normalization        | `RandomWalk`, `Symmetric`, `Unnormalized` |

---



### Complete GRAIL-FL Configuration Example
```yaml
# federated_learning_simulator/conf/graph_fed_avg/cifar10.yaml
---
# General
distributed_algorithm: graph_fed_avg
exp_name: "c510"
debug: false

# Training Parameters
optimizer_name: AdamW
optimizer_kwargs:
  betas: [0.9, 0.99]
  eps: 1e-8
  weight_decay: 0.00001

learning_rate_scheduler_name: ReduceLROnPlateau
learning_rate_scheduler_kwargs:
  mode: "max"
  factor: 0.5
  patience: 5

batch_size: 64
epoch: 3
learning_rate: 0.0001

# Model Configuration
model_name: densenet40
model_kwargs:
  num_classes: 10
  keep_model_cache: false

# Dataset Configuration
dataset_name: CIFAR10
dataset_kwargs:
  dataset_type: Vision
  dataset_sampling: dirichlet_split
  classes: [0,1,2,3,4,5,6,7,8,9]

dataset_sampling_kwargs:
  concentration: 0.5  # α=0.5 → moderate heterogeneity

# Distributed Learning Parameters
worker_number: 10
round: 150
algorithm_kwargs:
  distribute_init_parameters: True
  node_sample_percent: 0.7
  node_random_selection: false

# GRAIL-FL: Graph Representation
graph_worker: true
graph_type: KNN
num_neighbor: 2
threshold: 0.0
similarity_function: Gaussian

# GRAIL-FL: Spectral Clustering
family_number: 4
laplacian_type: RandomWalk
```

---


## Quick Start

### Running with YAML Configuration

```bash
# Use default configuration
python federated_learning_simulator/simulator.py \
    ---config-name fed_avg/cifar10.yaml


# Override specific parameters via command line
python federated_learning_simulator/run_experiment.py \
    ---config-name graph_fed_avg/cifar10.yaml \
    ++graph_fed_avg.worker_number=100 \
    ++graph_fed_avg.algorithm_kwargs.node_sample_percent=0.5 \
    ++graph_fed_avg.family_number=5
```