# SIMAP Layer: Simplicial-Map Neural Network Layer

## Overview

SIMAP (Simplicial-Map) is a novel neural network layer designed to enhance the interpretability of deep learning models. The SIMAP layer is an enhanced version of Simplicial-Map Neural Networks (SMNNs), an explainable neural network based on support sets and simplicial maps (functions used in topology to transform shapes while preserving their structural connectivity).

This repository contains experimental implementations and examples demonstrating the use of SIMAP layers in various neural network architectures.

## Key Features

### **Interpretability Enhancement**
- **Explainable AI**: SIMAP layers work in combination with other deep learning architectures as an interpretable layer substituting classic dense final layers
- **Topological Foundation**: Based on simplicial maps from algebraic topology that preserve structural connectivity
- **Decision Justification**: Provides explanations based on similarities and dissimilarities with training data instances

### **Technical Innovation**
- **Barycentric Subdivision**: Unlike SMNNs, the support set is based on a fixed maximal simplex, the barycentric subdivision being efficiently computed with a matrix-based multiplication algorithm
- **Efficient Computation**: Matrix-based algorithms for fast barycentric coordinate computation
- **Modular Design**: Can be integrated into existing deep learning architectures

## Repository Structure

| File  | Description |
| ------------- | ------------- |
| [main_SMNN](https://github.com/Cimagroup/SIMAP-layer/blob/main/main_SMNN.py)  | Main document with all necessary functions |
| [Notebook_Example](https://github.com/Cimagroup/SIMAP-layer/blob/main/Notebook_Example.ipynb)  | Toy example  |
| [Notebook_Synthetic_example](https://github.com/Cimagroup/SIMAP-layer/blob/main/Notebook_Synthetic_example.ipynb)  | Jupyter notebook using a synthetic dataset |
| [synthetic_data_grid](https://github.com/Cimagroup/SIMAP-layer/blob/main/synthetic_data_grid.py)| Experiment with synthetic dataset for different data dimensionalities  |
| [MNIST](https://github.com/Cimagroup/SIMAP-layer/blob/main/MNIST.py)  | Experiment with a convolutional neural network and a SIMAP-layer for the MNIST dataset  |


## Getting Started

### Prerequisites
- Python 3.x
- TensorFlow/PyTorch (depending on implementation)
- NumPy
- Matplotlib (for visualizations)
- Jupyter Notebook (for example notebooks)

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Cimagroup/SIMAP-layer.git
   cd SIMAP-layer
   ```

2. **Explore the toy example**:
   - Open `Notebook_Example` to see a simple implementation
   - This provides an introduction to SIMAP layer concepts

3. **Run synthetic experiments**:
   - Use `Notebook_Synthetic_example` for hands-on experimentation
   - Explore `synthetic_data_grid` for comprehensive dimensionality testing

4. **Try real-world application**:
   - Check the `MNIST` directory for a practical computer vision example

## Technical Background

### Simplicial Maps and Topology

SIMAP layers leverage concepts from algebraic topology:

- **Simplicial Complexes**: Mathematical structures that generalize triangles and tetrahedra to higher dimensions
- **Barycentric Coordinates**: A coordinate system that expresses points as weighted averages of simplex vertices
- **Simplicial Maps**: Functions that preserve the combinatorial structure of simplicial complexes

### How SIMAP Works

1. **Support Set Construction**: Creates a fixed maximal simplex based on the input space
2. **Barycentric Subdivision**: Efficiently subdivides the simplex using matrix operations
3. **Coordinate Mapping**: Maps input data to barycentric coordinates
4. **Interpretable Output**: Generates predictions with topological explanations

### Advantages over Traditional Dense Layers

- **Interpretability**: Each prediction comes with a geometric explanation
- **Structural Preservation**: Maintains topological relationships in the data
- **Efficient Computation**: Matrix-based algorithms for fast processing
- **Modular Integration**: Can replace final dense layers in existing architectures

## Use Cases and Applications

### **Computer Vision**
- Image classification with geometric interpretability
- Feature visualization in high-dimensional spaces
- Explainable convolutional neural networks

### **Synthetic Data Analysis**
- Multi-dimensional dataset exploration
- Topology-aware pattern recognition
- Geometric data understanding

### **Interpretable Machine Learning**
- Model explanation and validation
- Decision boundary visualization
- Trust-building in AI systems

## Experimental Results

The repository includes several experiments demonstrating SIMAP effectiveness:

### Synthetic Data Experiments
- **Multi-dimensional testing**: Performance across various data dimensionalities
- **Comparison studies**: SIMAP vs traditional dense layers
- **Interpretability analysis**: Visualization of decision boundaries

### MNIST Classification
- **Architecture**: CNN + SIMAP layer combination
- **Performance**: Competitive accuracy with enhanced interpretability
- **Visualization**: Topological representation of digit classification

## API Reference

### Core Functions (from `main_SMNN`)

The main implementation file contains essential functions for:

- **Simplex Construction**: Building the underlying topological structure
- **Barycentric Computation**: Efficient coordinate calculation
- **Layer Integration**: Connecting SIMAP with other neural network components
- **Visualization Tools**: Methods for interpreting and displaying results

## Research and Citations

This work is based on the research paper:

**"SIMAP: A simplicial-map layer for neural networks"**
- Authors: Rocio Gonzalez-Diaz, Miguel A. Gutiérrez-Naranjo, Eduardo Paluzo-Hidalgo
- Published: March 2024
- arXiv: [2403.15083](https://arxiv.org/abs/2403.15083)

If you use this code in your research, please cite:

```bibtex
@article{gonzalez2024simap,
  title={SIMAP: A simplicial-map layer for neural networks},
  author={Gonzalez-Diaz, Rocio and Gutiérrez-Naranjo, Miguel A. and Paluzo-Hidalgo, Eduardo},
  journal={arXiv preprint arXiv:2403.15083},
  year={2024}
}
```

## Contributing

We welcome contributions to improve and extend the SIMAP layer implementation:

1. **Bug Reports**: Submit issues for any problems encountered
2. **Feature Requests**: Suggest new capabilities or improvements
3. **Code Contributions**: Submit pull requests with enhancements
4. **Documentation**: Help improve examples and explanations

## License

Please refer to the repository's license file for usage terms and conditions.

## Contact and Support

For questions, suggestions, or collaboration opportunities:

- **Repository**: [https://github.com/Cimagroup/SIMAP-layer](https://github.com/Cimagroup/SIMAP-layer)
- **Issues**: Use GitHub issues for bug reports and feature requests
- **Research Group**: CIMA Group
- **Maintainer**: Eduardo Paluzo-Hidalgo (@EduPH)

---

## Summary

SIMAP layers represent a significant advancement in interpretable machine learning, combining the power of deep learning with the mathematical rigor of algebraic topology. This repository provides practical implementations and examples to help researchers and practitioners integrate topological interpretability into their neural network models.

The modular design allows SIMAP layers to enhance existing architectures while providing geometric insights into model decisions, making it valuable for applications requiring both high performance and interpretability.
