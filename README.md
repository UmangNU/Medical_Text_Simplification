# 🏥 Medical Text Simplification Using Fine-Tuned GPT-2

An AI-powered system for simplifying complex medical terminology into patient-friendly language using fine-tuned GPT-2 with LoRA parameter-efficient techniques.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK_HERE)
[![Demo](https://img.shields.io/badge/Demo-Gradio-orange)](https://165c62fbeb47f4e1a9.gradio.live/)

## 📋 Project Overview

**Problem:** 88% of adults struggle to understand medical documents, leading to medication errors and poor health outcomes.

**Solution:** Fine-tuned GPT-2 model that automatically transforms complex medical text into clear, accessible language.

**Results:** 32.7% improvement in ROUGE-1 scores over baseline GPT-2.

### Key Features

- ✅ **Multi-Level Simplification**: Three complexity tiers (child, adult, medical student)
- ✅ **13 Medical Specialties**: Covers cardiology, neurology, endocrinology, and more
- ✅ **4 Model Configurations**: Comprehensive hyperparameter optimization
- ✅ **LoRA Implementation**: Advanced parameter-efficient fine-tuning
- ✅ **Interactive Demo**: Live web interface powered by Gradio
- ✅ **Production-Ready**: Complete inference pipeline with <1s latency

## 🎯 Quick Start

### Option 1: Google Colab (Recommended)

1. Click the "Open in Colab" badge above
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Run all cells sequentially
4. Total time: ~4-5 hours (includes training)

### Option 2: Local Setup
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/medical-text-simplification.git
cd medical-text-simplification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook Medical_Text_Simplification.ipynb
```

**Note:** Local training requires CUDA-compatible GPU with 16GB+ memory.

## 📊 Dataset

**Source:** [MedAlpaca Medical Meadow Flashcards](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards)

**Statistics:**
- **Size:** 33,955 medical Q&A pairs
- **Source:** NIH, CDC, peer-reviewed medical literature
- **Specialties:** 13 medical domains
- **Split:** 80% train / 10% validation / 10% test

## 🤖 Model Architecture

**Base Model:** GPT-2 Small (124M parameters)

**Configurations Tested:**

| Config | Learning Rate | Technique | Val Loss | Training Time |
|--------|---------------|-----------|----------|---------------|
| **1 ⭐** | **5e-5** | **Full FT** | **1.279** | **68 min** |
| 2 | 3e-5 | Full FT | 1.326 | 70 min |
| 3 | 2e-5 | Full FT | 1.369 | 68 min |
| 4 | 1e-4 | LoRA (PEFT) | 1.685 | 17 min |

**Winner:** Config 1 (lr=5e-5) - Selected as production model

## 📈 Results

### Quantitative Performance

| Metric | Baseline GPT-2 | Fine-Tuned (Config 1) | Improvement |
|--------|----------------|----------------------|-------------|
| ROUGE-1 | 0.2431 | 0.3226 | **+32.7%** |
| ROUGE-2 | 0.1475 | 0.1796 | **+21.8%** |
| ROUGE-L | 0.2132 | 0.2519 | **+18.2%** |

### Qualitative Examples

**Input:** "Myocardial infarction results from coronary artery occlusion"

**Baseline:** "The condition is caused by a common form of the common cold..." ❌ (Incorrect!)

**Fine-Tuned:** "A heart attack happens when blood flow to the heart is blocked" ✅ (Accurate & Clear!)

## 🏗️ Project Structure
```
Medical_Text_Simplification.ipynb
├── Section 1: Setup & Imports
├── Section 2: Dataset Preparation (12 pts)
│   ├── Load 33,955 examples from Hugging Face
│   ├── Medical specialty classification
│   ├── Multi-level simplification creation
│   └── Train/val/test split (80/10/10)
├── Section 3: Model Selection (10 pts)
│   ├── Load GPT-2 (124M params)
│   └── Tokenization & formatting
├── Section 4: Training & Optimization (22 pts)
│   ├── Config 1: lr=5e-5 (WINNER)
│   ├── Config 2: lr=3e-5
│   ├── Config 3: lr=2e-5
│   ├── Config 4: LoRA technique
│   └── Hyperparameter comparison
├── Section 5: Evaluation (12 pts)
│   ├── ROUGE metrics calculation
│   ├── Baseline comparison
│   └── Performance analysis
├── Section 6: Error Analysis (8 pts)
│   ├── Error categorization
│   ├── Pattern identification
│   └── Improvement suggestions
├── Section 7: Inference Pipeline (6 pts)
│   └── Interactive Gradio demo
└── Section 8: Documentation (10 pts)
    ├── Environment setup guide
    ├── Reproduction instructions
    └── Code documentation
```

## 🚀 Usage

### Training (If running from scratch)
```python
# All training code is in the notebook cells
# Simply run cells 10-13 sequentially
# Training time: ~3 hours on T4 GPU
```

### Inference (Using pre-trained model)
```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load model (replace with your Google Drive path)
model = GPT2LMHeadModel.from_pretrained('/path/to/config1_full/checkpoint-XXX')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Simplify text
def simplify(complex_text):
    prompt = f"Simplify this medical text:\n\nComplex: {complex_text}\n\nSimple:"
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example
result = simplify("Myocardial infarction occurs due to coronary occlusion")
print(result)
```

### Interactive Demo
```bash
# Launch Gradio interface (Cell 17 in notebook)
# Access via generated public URL
# Demo link (temporary): https://165c62fbeb47f4e1a9.gradio.live/
```

## 📊 Technical Specifications

**Training Configuration:**
- **Epochs:** 5 per configuration
- **Batch Size:** 4 (effective: 8 with gradient accumulation)
- **Optimizer:** AdamW (weight_decay=0.01)
- **Learning Rate Schedule:** Linear warmup (400 steps) + decay
- **Mixed Precision:** FP16 enabled
- **Hardware:** NVIDIA Tesla T4 GPU (14.7 GB)

**LoRA Configuration:**
- **Rank:** 8
- **Alpha:** 32  
- **Dropout:** 0.1
- **Target Modules:** Attention layers (c_attn)
- **Trainable Parameters:** 294,912 (0.24% of total)

## 📁 Saved Models

Models are saved to Google Drive at:
```
/content/drive/MyDrive/Medical_Text_Simplification/
├── results/
│   ├── config1_full/         # Best model (lr=5e-5)
│   ├── config3_full/         # Alternative (lr=2e-5)
│   ├── final_model/          # Config 2 (lr=3e-5)
│   └── config_lora/          # LoRA model
└── logs/
    └── [corresponding log files]
```

## 🔍 Error Analysis Summary

**Error Categories Identified:**
1. **Length Control (40%)**: Outputs too long/short
2. **Jargon Retention (25%)**: Medical terms not simplified
3. **Incomplete Outputs (20%)**: Truncated responses
4. **Context Drift (10%)**: Off-topic responses
5. **Over-Simplification (5%)**: Critical details lost

**Proposed Solutions:** Post-processing pipeline, length constraints, medical glossary integration, specialty-specific models.

## 🎓 Educational Value

This project demonstrates:
- ✅ Complete fine-tuning workflow from data to deployment
- ✅ Systematic hyperparameter optimization methodology
- ✅ Advanced techniques (LoRA/PEFT)
- ✅ Rigorous evaluation and error analysis
- ✅ Production deployment considerations

Perfect for learning modern NLP engineering practices!

## 📚 References

1. Radford et al. (2019). "Language Models are Unsupervised Multitask Learners"
2. Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models"
3. Han et al. (2023). "MedAlpaca: Medical Domain Adaptation"
4. Hugging Face Transformers Documentation

## 📝 Citation

If you use this work, please cite:
```bibtex
@misc{medical-text-simplification-2025,
  author = {[Your Name]},
  title = {Medical Text Simplification Using Fine-Tuned GPT-2},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/YOUR_USERNAME/medical-text-simplification}
}
```

## 📄 License

This project is for educational purposes (Prompt Engineering Course Assignment).

## 👤 Author
 Umang Mistry
- Course: Prompt Engineering
- Institution: Northeastern University


## 🙏 Acknowledgments

- Hugging Face for Transformers library and model hosting
- MedAlpaca team for medical dataset
- Google Colab for GPU infrastructure
- Course instructors for guidance and feedback

## 📞 Contact

For questions or collaboration: mistr.um@northeastern.edu

---


