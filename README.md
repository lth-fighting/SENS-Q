```markdown
# SENS‑Q: Sensitivity-Guided Non-Uniform Quantization for Ribosome Stalling Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of the paper:

> **SENS‑Q: Sensitivity‑Guided Non‑Uniform Quantization for Efficient and Biologically Faithful Prediction of Ribosome Stalling with Hybrid Deep Learning Models**  
> Tianhui Li, Huiping Liu, Weiliang Zeng  
> *IEEE/ACM Transactions on Computational Biology and Bioinformatics (TCBB)*

*SENS‑Q compresses a hybrid CNN‑Mamba‑Transformer model by **8.5×** with negligible performance loss (FP32 R²=0.848 → 3‑bit R²=0.843), enabling large‑scale screening on commodity GPUs while preserving critical biological signals.*

---

## 🧠 Overview

Post‑training quantization (PTQ) in biological sequence models faces a unique challenge: **only <0.1% of parameters encode sparse regulatory motifs** (e.g., rare codon bottlenecks, Shine‑Dalgarno sequences) that govern ribosome stalling. Conventional uniform quantization treats all weights equally, destroying these critical signals.

**SENS‑Q** introduces three key innovations:

1. **Signal Protection** – Squared‑gradient sensitivities are computed over calibration data (200 sequences) to identify high‑impact weights. These are preserved in full precision using compressed sparse row (CSR) storage.

2. **Perception‑Driven Compression** – A sensitivity‑weighted non‑uniform K‑means objective replaces isotropic L₂ distortion. Limited quantization centroids are allocated preferentially to weights with high Fisher information, preserving biophysical feature resolution.

3. **Hardware‑Friendly LUT Module** – Sparse full‑precision outliers and bit‑packed quantized weights are fused inside a single lookup‑table linear layer, enabling direct deployment on laboratory GPUs.

The method is evaluated on a **hybrid CNN‑Mamba‑Transformer** architecture trained to predict *E. coli* ribosome stalling strength from 5′‑UTR‑CDS sequences and seven biophysical features.

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/lth-fighting/SENS-Q.git
cd SENS-Q
```

### 2. Set up a virtual environment (recommended for reproducibility)
We recommend using **Conda** (Python 3.8+) or Python’s built‑in `venv`.

**Option A – Conda**:
```bash
conda create -n sensq python=3.10 -y
conda activate sensq
```

**Option B – venv + pip**:
```bash
python -m venv sensq-env
# Linux / macOS
source sensq-env/bin/activate
# Windows
sensq-env\Scripts\activate
```

### 3. Install PyTorch (must match your CUDA version)
The code was tested with PyTorch 2.4.1 and CUDA 11.8.  
For the same setup:
```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```
For a CPU‑only installation, omit the `--index-url`.

### 4. Install core dependencies
```bash
pip install -r requirements.txt
```
This installs `mamba‑ssm`, `numpy`, `pandas`, `scikit‑learn`, `scipy`, `matplotlib`, `tqdm`, and all necessary sub‑dependencies.

### 5. Prepare the dataset
Download the preprocessed *E. coli* ribosome stalling dataset (228,000 sequences with biophysical annotations) from the sources listed in **Data Availability** of the paper.  
Place the CSV files in the `data/processed_data/` directory:

```
data/processed_data/
├── train_Ecoli_data.csv
├── val_Ecoli_data.csv
└── test_Ecoli_data.csv
```

### 6. [Optional] Web demo dependencies
To run the interactive web dashboard:
```bash
pip install -r webapp/requirements.txt
```

---

## 🚀 Quick Start

### Train the full‑precision model
```bash
python -m utils.training
```
This will produce a checkpoint `best_model.pth` inside `checkpoints/`.

> **Note**: Training from scratch requires the dataset to be correctly placed and the Mamba‑SSM library to be installed.

### Apply SENS‑Q quantization (3‑bit, default τ=0.05)
```bash
python -m quantization.quantize_main \
    --model_path ./checkpoints/best_model.pth \
    --output_dir ./checkpoints/quantized_3bit.pth \
    --data_dir ./data/processed_data \
    --bit 3 \
    --sensitivity 0.05 \
    --num_examples 200
```

### Compare original vs. quantized model
```bash
python -m evaluation.compare_models \
    --original_model ./checkpoints/best_model.pth \
    --quantized_model ./checkpoints/quantized_3bit.pth \
    --data_dir ./data/processed_data
```
The script outputs R², MAE, MSE, model size, and inference time – replicating **Table I** of the paper.

---

## 🧪 Reproducing Paper Experiments

All figures and tables can be regenerated using the scripts in `experiments/`.  
Make sure the trained model (`best_model.pth`) is in the project root (or adjust paths inside the scripts).

| Experiment | Script | Description |
|------------|--------|-------------|
| Performance vs. bit‑width | `exp1_performance.py` | R², MSE, memory for 2/3/4‑bit (Table I) |
| Sensitivity threshold sweep | `exp2_sensitivity_robustness.py` | Fig. 6 – τ vs. R² |
| Feature importance preservation | `exp3_biological_interpretability.py` | Spearman ρ, attention fidelity |
| Gradient visualisation | `visualize_gradients.py` | Violin plots, heatmaps (Fig. 8) |
| Outlier distribution | `visualize_outliers.py` | Outlier ratio per layer |
| Clustering analysis | `visualize_clustering.py` | Center pulling effect |
| Attention heatmaps | `visualize_attention.py` | Combined sample heatmap (Fig. 7B) |
| Uniform quantization baseline | `uniform_quant_baseline.py` | Comparison with uniform PTQ |
| Multi‑run stability | `multiple_quantization.py` | Boxplots of R² across runs |
| Sensitivity curve | `plot_sensitivity_curve.py` | Cubic spline interpolation (Fig. 6) |

Run with `python -m experiments.<script_name>` after placing the required checkpoints.

---

## 🌐 Interactive Web Demo

A self‑contained web application demonstrates the quantized model’s utility. It features:

- Single‑sequence prediction with adjustable biophysical parameters
- Batch CSV file processing
- Real‑time 3D molecular viewer (ribosome‑mRNA channel)
- Model evaluation dashboard with R², MAE, and residual analysis

### Launch the demo
```bash
python webapp/start.py
```
The backend (Flask) will start on `http://localhost:5000` and the dashboard on `http://localhost:8080`.  
By default, the demo uses a simplified predictor; **to deploy the actual quantized model**, replace the prediction logic in `webapp/backend/app.py` with the quantized `CNNMambaTransformer` (instructions are in the code comments).

---

## 📊 Key Results (from Paper)

| Model | R² | MAE | MSE | Stored Elements (M) | Compression |
|-------|-----|------|------|----------------------|--------------|
| FP32 Baseline | 0.848 | 0.268 | 0.157 | 3.515 | 1× |
| SENS‑Q 3‑bit (ours) | **0.843** | 0.283 | 0.156 | 0.415 | **8.5×** |
| Uniform 3‑bit | 0.825 | 0.292 | 0.174 | 0.352 | 10× (collapsed fidelity) |

*SENS‑Q preserves near‑lossless prediction while removing 88% of the stored parameters.*

---

## 📜 Citation

If you use this code or method, please cite our paper:

```bibtex
@article{li2026sensq,
  title={SENS‑Q: Sensitivity‑Guided Non‑Uniform Quantization for Efficient and 
         Biologically Faithful Prediction of Ribosome Stalling with Hybrid Deep Learning Models},
  author={Li, Tianhui and Liu, Huiping and Zeng, Weiliang},
  journal={IEEE/ACM Transactions on Computational Biology and Bioinformatics},
  year={2026},
  publisher={IEEE}
}
```

---

## 📧 Contact

For questions, please contact **Tianhui Li** (3085237492@qq.com) or **Weiliang Zeng** (weiliangzeng@gdut.edu.cn).  
Code repository: [https://github.com/lth-fighting/SENS-Q](https://github.com/lth-fighting/SENS-Q)

---

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

*The E. coli ribosome stalling dataset was originally generated by Cambray et al. (2018) and curated by Nikolados et al. (2022). Please refer to their original publications for data usage terms.*
```
