# Mechanistic Interpretability with Transformers

A comprehensive exploration of mechanistic interpretability through transformer implementations from scratch, discovering interpretable structure in random neural networks.

## 🔍 Project Overview

This repository documents a research journey into **mechanistic interpretability** - the field focused on understanding what neural networks learn and how they process information. The key discovery: **random, untrained neural networks already contain interpretable, structured features**.

### Key Research Question
> Do random, untrained neural networks contain interpretable features, or do these only emerge through training?

### Key Findings
1. **Random transformers contain feature detectors** (e.g., neurons that fire for "token 7 at position 3")
2. **Neurons form opposing pairs** that partition the input space systematically
3. **Random features respond consistently** to meaningful patterns without any training
4. **Interpretable structure emerges from architecture + initialization**, not just training

## 📁 Repository Structure

```
mechinterp_with_claude/
├── mechinterp.py                 # Complete mechanistic interpretability analysis
├── transformer.py                # Transformer implementation from scratch
├── transformer_with_backprop.py  # Transformer with backpropagation support
├── session_aug_18_2025.py       # Session notes/experiments
└── session_backup_18_aug_2025.py # Backup session file
```

## 🧠 Core Files

### `mechinterp.py` - Main Research Implementation
The flagship file containing:
- **Complete transformer built from scratch** (no deep learning libraries)
- **5 systematic experiments** discovering interpretable features in random networks
- **Feature detection analysis** finding neurons selective for specific patterns
- **Circuit analysis** revealing how neurons cooperate and oppose each other
- **Comprehensive visualizations** of neuron preferences and behaviors

### `transformer.py` - Clean Transformer Implementation
A pedagogical implementation featuring:
- Multi-head self-attention with causal masking
- Layer normalization and feedforward networks
- Positional encodings
- Complete mini-GPT architecture
- Attention pattern visualization tools

### `transformer_with_backprop.py` - Trainable Version
Extends the base transformer with:
- Full backpropagation implementation
- Training utilities for gradient-based learning
- Support for parameter updates

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy matplotlib
```

### Run the Main Analysis
```bash
python mechinterp.py
```

This will execute all experiments and generate visualizations showing:
- Feature detectors in random networks
- Neuron selectivity analysis
- Circuit discovery (cooperating/opposing neuron pairs)
- Response patterns to structured inputs

### Run Individual Components
```bash
# Basic transformer demo
python transformer.py

# Transformer with backprop
python transformer_with_backprop.py
```

## 🔬 Experiments Overview

The `mechinterp.py` file contains five systematic experiments:

### Experiment 1: Finding Feature Detectors
Discovers that random networks contain neurons that consistently fire for specific input patterns.

### Experiment 2: Testing Specific Hypotheses
Tests whether identified neurons are truly selective for specific token-position combinations.

### Experiment 3: Systematic Neuron Analysis
Maps the specializations of multiple neurons, finding position-selective and token-selective units.

### Experiment 4: Circuit Discovery
Reveals how neurons form structured circuits with strong correlations (>0.8) and oppositions (<-0.8).

### Experiment 5: Structured Input Response
Tests how random features respond to meaningful patterns like sequences, repetitions, and alternations.

## 📊 Key Results

- **10+ interpretable neurons** found in a 32-dimensional random network
- **Strong circuit structure** with correlation coefficients >0.8
- **Consistent feature detection** across different input types
- **Position and token selectivity** emerging from pure randomness

## 🧬 Implications for AI Safety & Interpretability

### For Mechanistic Interpretability Research:
- **Challenge the "learning" paradigm**: Features may be selected/amplified rather than created
- **Require random baselines**: Compare trained vs. untrained networks
- **Separate architectural bias from learning**: Distinguish inherent vs. acquired structure

### For the Lottery Ticket Hypothesis:
- May extend beyond sparse networks to **interpretable features**
- Random initialization contains **structured subnetworks** ready for selection
- Training might be **feature selection** rather than feature creation

### For AI Safety:
- **Interpretable features exist before training** - important for understanding model capabilities
- **Architecture choice significantly impacts** interpretable structure
- **Random baselines essential** for understanding what's truly learned vs. inherited

## 🛠️ Technical Implementation

### Core Components
- **Softmax with numerical stability**
- **Layer normalization for training stability**  
- **Self-attention with causal masking**
- **Feedforward networks with ReLU activation**
- **Complete transformer blocks with residual connections**

### Analysis Tools
- **Activation capture hooks** for intermediate states
- **Neuron selectivity measurement** across positions and tokens
- **Correlation analysis** for circuit discovery
- **Statistical significance testing** for feature validation

## 📈 Future Directions

1. **Track feature evolution** during training
2. **Compare across multiple random initializations**
3. **Test different architectures** for interpretability bias
4. **Investigate feature survival** through training
5. **Scale to larger models** and validate findings

## 🎓 Educational Value

This repository serves as:
- **Complete transformer tutorial** built from NumPy
- **Mechanistic interpretability primer** with hands-on experiments
- **Research methodology demonstration** for feature discovery
- **Foundation for advanced interpretability research**

## 📝 Citation

If you use this work in your research:

```
@misc{mechinterp_with_claude_2025,
  title={Mechanistic Interpretability: Discovering Structure in Random Transformers},
  author={Omar Khursheed},
  year={2025},
  note={Research exploration into interpretable features in untrained neural networks}
}
```

## 📜 License

This project is open source and available under the MIT License.

---

**"The most profound discoveries often come from questioning our most basic assumptions. Here, we discovered that neural networks don't just learn to be interpretable—they start that way."**