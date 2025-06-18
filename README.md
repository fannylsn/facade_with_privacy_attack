# Facade with Privacy Attack

## Overview

This repository extends the [FACADE](https://github.com/sacs-epfl/facade) decentralized learning (DL) framework by embedding **privacy attacks** directly into the training loop. While the original FACADE algorithm promotes fairness by enabling emergent clustering among clients with similar feature distributions, this work goes a step further by investigating the **privacy vulnerabilities** of such a system **during training**.

In particular, we integrate **membership inference attacks (MIA)** into FACADE’s training pipeline. This allows us to analyze how **cluster formation**, **model specialization**, and **heterogeneity in data distributions** influence privacy leakage over time.

## Key Contributions

- 🔍 **First evaluation of FACADE's vulnerability to privacy attacks**, focusing on real-time MIA during DL training.  
- 🧪 **Integrated experimental pipeline** for launching in-training MIAs in FACADE, enabling temporal analysis of attack success throughout the learning process.  
- ⚖️ **Insight into the fairness–privacy tradeoff**: models trained for fairness—especially toward minority clusters—can exhibit increased susceptibility to privacy leakage.

## Running the Experiments

1. **Configure the experiment parameters**, such as cluster ratios, dataset paths, etc. by editing the configuration file:

   ```bash
   ./facade_with_MIA/tutorial/IDCA/configs/configCIFAR_idca.ini
   ```

2. **Launch the training and attack pipeline** using the provided script:

   ```bash
   ./facade_with_MIA/tutorial/IDCA/run_IDCA.sh
   ```

This will train the FACADE model on CIFAR (or the selected dataset), while periodically executing membership inference attacks.

## Notes

- In this repo, FACADE is refered to as IDCA
- The default setup targets CIFAR-10. To use other datasets, update the dataset path and settings in the configuration file accordingly.

