# Gaussian Mixture Model-based Speaker Recognition

## Overview

This repository details a speaker recognition system using Gaussian Mixture Models (GMM) and Mel-Frequency Cepstral Coefficients (MFCCs), developed at CSIR-CMERI. The project leverages advanced machine learning techniques to analyze and recognize speakers across different acoustic environments.

## Repository Structure

- **`My Project-2`** and **`My Project-2_noise`**:
  - These directories are configured for operations in clean and noisy conditions, respectively. Each includes:
    - **`Sp_Models`**: Trained Gaussian Mixture Models for the respective environment.
    - **`Test-Data`** and **`Train-Data`**: Audio samples used for model evaluation and training.
    - Core scripts responsible for executing the feature extraction, training, and testing phases.
- **`Adhiraj_Banerjee_FinalReport_CSIR-CMERI.pdf`**: A comprehensive report that discusses the methodologies, experimental setup, and insights derived from the project.

## Machine Learning Workflow

### Feature Extraction
The feature extraction process involves several critical steps to prepare audio data for machine learning modeling:
- **Pre-Emphasis**: Enhances the high-frequency parts of the signal to improve the reliability of the feature set.
- **Frame Blocking**: Breaks the audio signal into manageable frames to treat it as quasi-stationary.
- **Windowing**: Applies a window function to minimize edge effects in each frame.
- **Fourier Transform**: Converts frames from the time domain to the frequency domain using the Fast Fourier Transform.
- **Mel-Frequency Wrapping**: Applies a Mel-scaled filter bank to focus on frequencies that are significant for human auditory perception.

### Gaussian Mixture Model and Expectation-Maximization Algorithm
The use of GMMs allows the system to model the voice features with multiple Gaussian distributions:
- **Initialization**: Sets initial parameters for Gaussian components.
- **Expectation Step**: Assigns data points to specific Gaussian components based on their probabilities.
- **Maximization Step**: Adjusts Gaussian parameters to maximize the likelihood of the data.
- **Convergence**: Repeats until the parameters stabilize, ensuring optimal modeling of the speaker's voice characteristics.

## Installation and Setup

Ensure you have Python installed with the necessary libraries to run the scripts:

```bash
pip install numpy scipy librosa scikit-learn matplotlib
```

## Usage

To utilize the scripts within Python IDLE:

1. **Feature Extraction**:
   - Execute `feature_extr.py` from the respective directory to process audio files and extract features.

2. **Model Training**:
   - Run `Training_model.py` to train the Gaussian Mixture Models using the prepared features.

3. **Model Testing**:
   - Start `Test.py` to evaluate the effectiveness of the trained models.

## Acknowledgements

Special acknowledgment is extended to **Dr. Siva Ram Krishna Vadali, Senior Principal Scientist at the Robotics & Automation Division, CSIR-Central Mechanical Engineering Research Institute, Durgapur**. His profound expertise and insightful supervision have been instrumental in shaping both the direction and success of this project. His mentorship was invaluable in integrating complex machine learning techniques with practical applications in speaker recognition.
