# Multimodal Emotion Recognition (多模态情绪识别)
[中文](README.md) | English

> A research suite for multimodal emotion recognition methods based on deep learning, supporting emotion classification and sentiment analysis tasks in text, audio, and video modalities.

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Project Introduction

This project is a research suite for multimodal emotion recognition, supporting the following features:

- 🎯 **Multimodal Fusion**: Supports independent and combined use of text (T), audio (A), and video (V) modalities
- 📊 **Multiple Dataset Support**: Supports mainstream emotion datasets such as MELD, IEMOCAP, and SIMS
- 🚀 **Efficient Training**: Integrates optimization techniques such as caching mechanism, mixed-precision training, and knowledge distillation
- ⚙️ **Flexible Configuration**: Modular design based on TOML configuration files, supporting various experimental configurations
- 📈 **Complete Workflow**: A complete toolchain from data preprocessing to model training and evaluation

For detailed project structure, please refer to the [Project Structure Documentation](docs/structure.md).

## Quick Start

### Environment Requirements

- Python 3.12+
- PyTorch 2.4.1
- CUDA 12.1 (optional, for GPU acceleration)

### Installation

1. Clone the project repository:
```bash
git clone https://github.com/zrr1999/emotion-recognition.git
cd emotion-recognition
```

2. Install dependencies:
```bash
# Use uv to install (recommended)
uv sync --all-extras --dev

# Users in China can accelerate using Tsinghua source
uv sync --all-extras --dev --index-url https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cu121
```

3. Prepare the dataset:
```bash
# Create the dataset directory
mkdir datasets

# Download and link the MELD dataset
ln -s /path/to/MELD datasets/MELD
```

The dataset needs to be organized in the format specified in the [Project Structure Documentation](docs/structure.md#数据集结构).

### Basic Usage

#### Command Line Tools

The project provides two command line tools:

```bash
# Emotion recognition tool
emotion-recognize --help

# Data processing tool
emotion-tool --help
```

#### Training the Model

Use [nanoflow](https://github.com/zrr-lab/nanoflow) for experiments:

```bash
# Run basic experiments
uvx nanoflow run experiments/mdn.toml
uvx nanoflow run experiments/makd.toml

# Run ablation experiments
uvx nanoflow run experiments/mdn-ablation.toml
uvx nanoflow run experiments/makd-ablation.toml
```

## Core Features

### 🚀 Efficient Training Mechanism

#### Feature Caching Mechanism
Since the output of the feature extraction module remains unchanged when it is frozen for the same input, the outputs can be cached to reduce repeated calculations. This project uses [SafeTensors](https://github.com/huggingface/safetensors) format for feature caching, which offers better performance and security compared to other formats.

#### Transparent Model Storage
During experiments, we often save many model checkpoints and train multiple different models by modifying parameters. Generally, a copy of parameters is required for each model, leading to wasted storage space. To address this issue, we use soft links to connect the same model parameters to different model folders, thereby reducing storage space waste. Moreover, this method does not disrupt the original folder structure, making the overall structure clearer.

#### Mixed Precision Training
Integrated automatic mixed precision (AMP) support significantly speeds up the training process and reduces memory usage while maintaining model accuracy.

### ⚙️ Flexible Configuration System

#### Dynamic Configuration Concept
During experiments, it is often necessary to frequently modify parameters to train multiple different models for comparison. This project adopts a configuration file solution that maintains clean code and conveniently modifies parameters. The configuration files support complex data structures such as lists and dictionaries, offering expressive power far beyond command-line parameters.

Modular design implemented using TOML configuration files:

- `configs/encoders/`: Encoder configurations (T, A, V, and their combinations)
- `configs/fusion/`: Fusion strategy configurations
- `configs/losses/`: Loss function configurations
- `configs/dataset/`: Dataset configurations

Technical dependencies: [Pydantic](https://pydantic-docs.helpmanual.io/) for configuration validation and type checking.

### 🔄 Knowledge Distillation Optimization

In the implementation of [TelME](https://github.com/yuntaeyang/TelME), using knowledge distillation to train different modalities requires multiple iterations, leading to excessive training time (even with the caching techniques of this project). To resolve this, we merge the knowledge distillation training of multiple modalities into a single training run, significantly reducing training time.

### 📊 Multiple Dataset Support

Supports mainstream emotion recognition datasets:
- **MELD**: Emotion classification and sentiment analysis
- **IEMOCAP**: Emotion classification
- **SIMS**: Sentiment analysis

### 🔧 Continuous Integration

This project has implemented basic continuous integration, specifics can be found in [train-and-eval.yml](.github/workflows/train-and-eval.yml).

## Project Conventions

### Checkpoint Naming Convention

The checkpoint naming format is `{training category}/{dataset}/{training method}--{batch size}--{classification loss function}/{network summary}/{network hash}--{random seed}`, for example, `training/MELD--E/trainable--2--{loss}/1xE--T/51fe7ba3--114`.

#### Modality Abbreviations
- **T** (Text): Text modality
- **A** (Audio): Audio modality
- **V** (Video): Video modality

#### Training Method Abbreviations
- **T** (Full Tuning): Full parameter fine-tuning
- **L** (LoRA): Low-Rank Adaptation
- **F** (Froze Backbones): Freeze backbone networks

#### Dataset Type Abbreviations
- **E** (Emotion): Emotion classification tasks
- **S** (Sentiment): Sentiment analysis tasks

## Technical Architecture

### Core Dependencies

| Component | Version | Purpose |
|-----------|---------|---------|
| Python    | 3.12+   | Runtime environment |
| PyTorch   | 2.4.1   | Deep learning framework |
| Transformers | 4.48+ | Support for pretrained models |
| Pydantic  | 2.0+    | Configuration validation |
| SafeTensors | 0.4+  | Efficient feature caching |

### Model Architecture

```
Multimodal input → Feature Encoder → Feature Fusion → Classifier → Emotion Prediction
    ↓           ↓          ↓        ↓         ↓
  T/A/V    BERT/Whisper  Deep/MoE  Linear   Classes
```

#### Encoder Support
- **Text (T)**: BERT, RoBERTa, ModernBERT, etc.
- **Audio (A)**: Whisper, Distil-Whisper, etc.
- **Video (V)**: OpenCV feature extraction

#### Fusion Strategies
- **Vanilla Fusion**: Simple concatenation
- **Deep Fusion**: Deep neural networks
- **MoE Fusion**: Mixture of Experts model
- **Attention Fusion**: Self-attention mechanism

### Performance Optimization

- **Video Reading**: OpenCV (0.08s/frame) vs PyAV (0.17s/frame)
- **Feature Caching**: Using SafeTensors format to avoid repeated calculations when frozen
- **Storage Optimization**: Soft link mechanism to reduce model storage space

## Experimental Results

> The following results are based on the emotion classification tasks of the MELD dataset.

### Unimodal Performance

#### Text Modality (Text-only)

| Method | Random Seed | Accuracy | Weighted-F1 |
|--------|-------------|----------|-------------|
| APCL (temp=0.08, β=0.1, γ=0.1) | 43 | 67.74% | 67.04% |
| APCL (temp=0.08, β=0.1, γ=0.1) | 42 | 68.05% | 66.91% |
| APCL (temp=0.08, β=0.1, γ=0.1) | 114 | 67.59% | 66.55% |
| APCL (temp=0.08, β=0.1, γ=0.1) | 0 | 67.93% | 66.92% |
| SPCL (temp=0.08, pool=512, support=64) | 42 | 68.31% | 67.31% |
| SPCL (temp=0.08, pool=512, support=64) | 114 | 67.32% | 66.55% |
| SPCL (temp=0.08, pool=512, support=64) | 0 | 66.63% | 66.50% |

### Multimodal Performance

| Modality Combination | Method | Accuracy | Weighted-F1 | Notes |
|----------------------|--------|----------|-------------|-------|
| T+A+V                | To be supplemented | - | - | To be supplemented |
| T+A                  | To be supplemented | - | - | To be supplemented |
| T+V                  | To be supplemented | - | - | To be supplemented |

## Contribution Guidelines

Contributions of code and ideas are welcome! Please follow these steps:

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Create a Pull Request

### Development Environment

```bash
# Install development dependencies
uv sync --dev

# Run code checks
uv run ruff check src/
uv run pyright src/

# Run formatting
uv run ruff format src/
```

## References

### Pretrained Models
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/pdf/1810.04805)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692)
- [ModernBERT: Modernizing the BERT Architecture](https://github.com/AnswerDotAI/ModernBERT)

### Model Compression Techniques
- [LoRA: Low-Rank Adaptation of Large Language Models](https://huggingface.co/docs/peft/task_guides/lora_based_methods)

### Knowledge Distillation
- [Knowledge Distillation for Deep Learning](https://arxiv.org/pdf/2104.09044)
- [DIST: Distillation with Student-Teacher Networks](https://arxiv.org/pdf/2205.10536)
- [Cross-Modal Knowledge Distillation](https://arxiv.org/pdf/2401.12987v2)

### Mixture of Experts Models
- [TGMoE: A Text Guided Mixture of Experts Model](https://ftp.saiconference.com/Downloads/Volume15No8/Paper_119-TGMoE_A_Text_Guided_Mixture_of_Experts_Model.pdf)

### Speech Recognition
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper)
- [Distil-Whisper: Robust Knowledge Distillation](https://arxiv.org/abs/2311.00430)
- [Distil-Large-V3](https://huggingface.co/distil-whisper/distil-large-v3)
- [BELLE: Be Everyone's Large Language model Engine](https://github.com/LianjiaTech/BELLE)

## Other Resources

### Technical Blogs and Tutorials
- [Overview of Multimodal Emotion Recognition](https://zhuanlan.zhihu.com/p/694747931) - A comprehensive introduction to multimodal emotion recognition technologies
- [Detailed Explanation of Automatic Mixed Precision AMP Training](https://zhuanlan.zhihu.com/p/408610877/) - Explanation and best practices for PyTorch AMP technology

### Datasets and Benchmarks
- [Comparison of Emotion Recognition Datasets](https://paperswithcode.com/task/emotion-recognition-in-conversation) - Related resources on Papers with Code
- [Official MELD Dataset](https://affective-meld.github.io/) - Official website of the MELD dataset
- [IEMOCAP Dataset](https://sail.usc.edu/iemocap/) - IEMOCAP Emotion Database

### Academic Resources
- [Review of Multimodal Learning Methods](https://arxiv.org/abs/2209.05025) - Recent advances in multimodal deep learning
- [Affective Computing International Conference (ACII)](http://acii-conf.org/) - Top conference in the field of affective computing
- [Multimodal Machine Learning Tutorial](https://cmu-multicomp-lab.github.io/mmml-tutorial/) - CMU Multimodal Course

### Tools and Frameworks
- [Transformers Library Documentation](https://huggingface.co/docs/transformers/) - User guide for HuggingFace Transformers
- [Official PyTorch Tutorials](https://pytorch.org/tutorials/) - PyTorch deep learning tutorials
- [Nanoflow Workflow Engine](https://github.com/zrr-lab/nanoflow) - Experiment management tool

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

Thanks to all the students and teachers who provided guidance and ideas for this project!

---

If you have any questions or suggestions, feel free to submit an [Issue](https://github.com/zrr1999/emotion-recognition/issues) or a [Pull Request](https://github.com/zrr1999/emotion-recognition/pulls).
