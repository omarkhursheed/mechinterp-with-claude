# Mechanistic Interpretability: Feature Persistence in Transformers

Investigating how interpretable features evolve during transformer training - do they emerge from scratch or were they there all along?

## Project Overview

This repo explores mechanistic interpretability (mech interp) of transformers, specifically looking at when interpretable features actually show up. The main question: do random, untrained networks already have interpretable structure, or does it only emerge through training?

We're building tools to peek inside transformers at different stages and see what's actually happening.

### What We're Investigating

**Core Question:** Do random transformers have interpretable features before any training, or do these only emerge through gradient descent?

**Early Observations:**
- Random transformers show some structured attention patterns (not just uniform noise)
- Found neurons in untrained networks that seem to respond to specific token-position combinations  
- Some evidence of systematic activation patterns even without any training
- Architecture and initialization choices seem to matter for what structure exists pre-training

This challenges the usual assumption that interpretability only comes from learning.

## Repository Structure

```
mechinterp_with_claude/
├── do_some_mechinterp.ipynb      # Main research notebook - MI toolkit [ACTIVE]
├── mechinterp.py                 # Complete analysis pipeline (5 experiments)
├── transformer.py                # Minimal transformer from scratch
├── transformer_with_backprop.py  # Transformer with training support
├── torch_transformer.py          # PyTorch implementation
└── session_*.py                  # Previous experiments
```

## What We've Built

### Mechanistic Interpretability Toolkit
Standard mech interp tools for transformer analysis:
- **Activation hooks**: Capture intermediate activations without modifying forward pass
- **Logit lens**: Project hidden states to vocab space to see what each layer "thinks"
- **Attention pattern analysis**: Classify attention heads (previous token, first token, etc.)
- **Activation patching**: Causal interventions to test which activations matter
- **Ablation studies**: Remove neurons/components to measure their importance

### Initial Findings
- Random (untrained) transformers have non-uniform attention patterns
- Some neurons in random networks fire selectively for specific input patterns
- Attention heads show biases toward certain positions even without training
- There's more structure in random initialization than we expected

## Key Files

### `do_some_mechinterp.ipynb` - Main Research [ACTIVE]
The main notebook where we're developing the MI toolkit:
- **TinyTransformer**: Simple PyTorch transformer for quick experiments
- **Activation patching & ablation**: Just implemented causal intervention tools
- **Hook-based analysis**: Clean way to peek inside model without modifying architecture
- Currently exploring feature persistence during training

### `mechinterp.py` - From-Scratch Analysis  
Complete pure NumPy implementation:
- Transformer built without any ML libraries (educational + transparency)
- 5 systematic experiments on random networks
- Found evidence of interpretable structure in untrained models
- Correlation analysis showing neuron relationships

### `torch_transformer.py` - Production Implementation
Standard PyTorch transformer for scaling up:
- Multi-head attention, layer norm, standard architecture
- Ready for training runs and larger-scale experiments

## Getting Started

### Setup
```bash
# Minimal setup
pip install numpy matplotlib torch

# For the main analysis
python mechinterp.py

# For interactive development  
jupyter notebook do_some_mechinterp.ipynb
```

## Experiments in mechinterp.py

Five experiments that systematically probe random (untrained) transformer structure:

1. **Feature Detection**: Look for neurons that fire consistently for specific patterns
2. **Hypothesis Testing**: Validate whether detected features are actually selective 
3. **Systematic Mapping**: Survey multiple neurons to see what they respond to
4. **Circuit Discovery**: Find correlations and oppositions between neurons
5. **Structured Inputs**: Test responses to sequences, repetitions, alternations

The key insight: even random networks show way more structure than expected.

## Results So Far

### From the Notebook (Active Development)
- Random transformers have non-uniform attention patterns (not just noise)
- Attention heads show consistent biases before any training
- Logit lens works on random models - different layers make different "predictions"
- Successfully implemented activation patching and ablation tools

### From mechinterp.py (Previous Analysis)
- Found ~10+ neurons in random 32D network that respond to specific patterns
- Strong correlations (>0.8) and anti-correlations (<-0.8) between neurons
- Position-selective and token-selective responses in untrained networks
- Systematic structure that persists across different random seeds

## Why This Matters

### For Mech Interp Research
- Challenges the assumption that interpretability = learning
- Need random baselines when studying trained models
- Architecture choices create interpretable bias before any gradient steps

### Broader Implications
- Lottery ticket hypothesis might extend to interpretable features
- Training could be more about *selecting* existing structure than creating it
- Pre-training interpretability has implications for understanding model capabilities

This isn't just a curiosity - it changes how we think about what neural networks learn vs. what they start with.

## Next Steps

Currently working on:
- Feature persistence tracking through training (how do random features evolve?)
- Scaling up to larger models to see if patterns hold
- More systematic analysis of different architectures and initializations
- Connecting to lottery ticket hypothesis and feature selection vs. creation

## Technical Notes

**Implementation**: Pure NumPy transformer (mechinterp.py) for transparency, PyTorch version (torch_transformer.py) for scaling, main development in Jupyter notebook.

**Key tools**: Activation hooks, logit lens, attention analysis, activation patching, ablation studies - standard mech interp toolkit.

---

**License**: MIT  
**Contact**: This is research in progress - findings are preliminary but interesting.