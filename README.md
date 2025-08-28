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

### Key Finding: Significant Structure in Random Networks

In statistical testing of a small transformer:
- All tested neurons (16/16) showed statistically significant token selectivity (p < 0.01 with Bonferroni correction)
- All tested neurons (16/16) showed statistically significant position selectivity
- Effect sizes ranged from 0.35-0.76 (proportion of variance explained)
- These patterns appear in untrained, randomly initialized networks

### Possible Sources of Structure

Analysis suggests several contributing factors:
1. Random embeddings show diversity (cosine similarities range from -0.59 to 0.56)
2. Weight matrices show directional preferences (singular value ratios around 45x)
3. Token-position combinations create 80 distinct input patterns
4. Architecture and initialization interact to create non-uniform responses

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

### Statistical Analysis (Current Findings)
- All tested neurons showed statistically significant selectivity (ANOVA, p < 0.01)
- Token selectivity effect sizes: η² = 0.35-0.76
- Position selectivity effect sizes: η² = 0.36-0.45
- Attention weights average to uniform (0.125) but vary by input
- Some neurons show opposing activation patterns

### Visualization Insights
- Clear vertical bands in token selectivity heatmaps
- Strong position gradients and diagonal patterns
- Token-position interaction effects (not just additive)
- Attention variability around uniform baseline suggests structured randomness

### From mechinterp.py (Previous Analysis)
- Found ~10+ neurons in random 32D network that respond to specific patterns
- Strong correlations (>0.8) and anti-correlations (<-0.8) between neurons
- Position-selective and token-selective responses in untrained networks
- Systematic structure that persists across different random seeds

## Why This Matters

### For Mech Interp Research
- Random baselines are important when studying trained models
- Need to account for structure that exists before training
- Statistical testing helps distinguish real patterns from noise

### Research Questions Raised
- Do initial features persist through training or get replaced?
- How much does architecture vs. initialization contribute to this structure?
- Are these initial features useful for downstream tasks?
- Does this pattern hold for larger models and different architectures?

### Potential Implications
- May inform initialization strategies
- Could affect how we think about feature emergence during training
- Suggests investigating the relationship between initial and final features

## Next Steps

### Immediate Research Questions
1. **Feature Persistence**: Do these random features survive training or get overwritten?
2. **Scale Testing**: Does 100% neuron selectivity hold in larger models?
3. **Architecture Comparison**: How do different architectures affect initial feature structure?

### Experiments to Run
- Track specific neuron identities through training (correlation analysis)
- Test with different initialization schemes (Xavier, He, etc.)
- Compare transformer vs. MLP vs. CNN initial features
- Measure feature "usefulness" - are random features helpful for tasks?

### Theoretical Work
- Formalize the "Feature Selection Hypothesis"
- Connect to lottery ticket and network pruning literature
- Develop metrics for "feature quality" at initialization

## Technical Notes

**Implementation**: Pure NumPy transformer (mechinterp.py) for transparency, PyTorch version (torch_transformer.py) for scaling, main development in Jupyter notebook.

**Key tools**: Activation hooks, logit lens, attention analysis, activation patching, ablation studies - standard mech interp toolkit.

---

**License**: MIT  
**Contact**: This is research in progress - findings are preliminary but interesting.