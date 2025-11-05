# Federated Learning Simulation Framework
### Graph-Based Client Selection with Information Theory and Spectral Clustering
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-BSD-green.svg)](LICENSE)
[![Paper Status](https://img.shields.io/badge/IEEE%20IoT%20Journal-Under%20Review-yellow.svg)](https://ieee-iotj.org/)

> **A comprehensive simulation framework for federated learning with advanced client selection strategy based on graph modeling of FL system, information theory, and spectral clustering techniques.**

---


## 📋 Table of Contents
- [X] [Overview](#overview) 
- [X] [Key Features](#key-features)
- [X] [Research Context](#research-context)
- [X] [Architecture & Project Structure](#architecture--project-structure)
    ### TODOs
- [ ] [Installation](#installation)  
- [ ] [Quick Start](#quick-start)
- [ ] [Detailed Component Guide](#detailed-component-guide)
  - [ ] [Core Library: `federated_learning_simulation_lib`](#1-core-library-federated_learning_simulation_lib)
  - [ ] [Simulation Engine: `federated_learning_simulator`](#2-simulation-engine-federated_learning_simulator)
  - [ ] [Vision Toolkit: `torch_vision`](#3-vision-toolkit-torch_vision)
  - [ ] [Medical Imaging: `torch_medical`](#4-medical-imaging-torch_medical-optional)
  - [ ] [Utility Libraries: `other_libs` & `torch_kit`](#5-utility-libraries-other_libs--torch_kit)
- [ ] [Experimental Workflows](#experimental-workflows)
- [ ] [Configuration](#configuration)
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


## 🏗️ Architecture & Project Structure

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

| Component                                   | Purpose                                                               | Key Features                                                                          |
|---------------------------------------------|-----------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| [**`federated_learning_simulation_lib`**](https://github.com/Meriem30/federated_learning_simulation/tree/main/federated_learning_simulation_lib) | Core FL library with graph-based client selection                     | Worker, GraphWorker, Sampler, Server; Aggregation, Selection and Clustering Algorithms |
| [**`federated_learning_simulator`**](https://github.com/Meriem30/federated_learning_simulation/tree/main/federated_learning_simulator)      | Orchestration engine for running FL experiments (Project Entry Point) | Experiment Configuration, Client and Server Endpoints, Aggregation Method Assignment  | 
| [**`torch_vision`**](https://github.com/Meriem30/federated_learning_simulation/tree/main/torch_vision)                      | Computer Vision FL Implementations                                    | Dataset Constructors, Transformer Pipelines, Vision Model Constructors                | 
| [**`torch_medical`**](https://github.com/Meriem30/federated_learning_simulation/tree/main/torch_medical)                     | Medical Imaging Specialization                                        | Medical Dataset Registration, Domain-Specific Transformers, Models                    | 
| [**`torch_kit`**](https://github.com/Meriem30/federated_learning_simulation/tree/main/torch_kit)                         | Advanced PyTorch Utility Toolkit (Extensions)                         | Design Abstractions (Training/Optimizers/Metrics), Device Management, Custom layers   | 
| [**`other_libs`**](https://github.com/Meriem30/federated_learning_simulation/tree/main/other_libs)                        | Helper Utilities to Simplify Python Tasks                             | Data loaders, Log Formatting, Communication Abstractions                              | 



---