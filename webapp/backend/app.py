"""
Flask backend API server for Ribosome Stalling Prediction System.
(Adapted from original app.py – uses a simplified model for demonstration.
 In production, replace with the SENS‑Q quantized model by loading
 QuantLinearLUT weights and the CNNMambaTransformer architecture.)
"""

import sys
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from itertools import product

# Add project root to Python path so that we can import our model definitions
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.model_architecture import CNNMambaTransformer
from evaluation.model_io import build_quantized_model_from_state_dict  # for potential quant model loading

# ---------- DNA encoding (original implementation) ----------
def DNA_kmer_onehot_encode(sequence: str, k: int = 3):
    """One‑hot encode a DNA sequence using overlapping k‑mers (as in cmt.py)."""
    kmer_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
    bases = ['A', 'T', 'C', 'G']
    all_kmer = [''.join(p) for p in product(bases, repeat=k)]

    kmer_to_char = {}
    for i, kmer in enumerate(all_kmer[:len(kmer_chars)]):
        kmer_to_char[kmer] = kmer_chars[i]

    kmer_sequence = []
    step = 1
    for i in range(0, len(sequence) - k + 1, step):
        kmer = sequence[i:i + k].upper()
        char = kmer_to_char.get(kmer, 'N')
        kmer_sequence.append(char)

    char_to_index = {char: idx for idx, char in enumerate(kmer_chars)}
    one_hot_encoded = []
    for char in kmer_sequence:
        one_hot = [0] * len(kmer_chars)
        if char in char_to_index:
            one_hot[char_to_index[char]] = 1
        one_hot_encoded.append(one_hot)
    return np.array(one_hot_encoded)


# ---------- Simplified prediction model (used when quantized model is not loaded) ----------
class SimplifiedModel:
    """Fallback predictor – uses only GC content and sequence length."""
    def predict(self, sequence):
        seq_length = len(sequence)
        gc_content = (sequence.count('G') + sequence.count('C')) / seq_length if seq_length > 0 else 0
        base_score = gc_content * 0.3 + (1 - gc_content) * 0.7
        length_factor = min(seq_length / 100, 1.0)
        return np.clip(base_score * length_factor, 0, 1)


# ---------- Helper functions for parameter‑based adjustment ----------
def apply_biophysical_features(cdsCAI, utrCdsStructureMFE, fivepCdsStructureMFE, threepCdsStructureMFE,
                               cdsBottleneckPosition, cdsNucleotideContentAT, cdsHydropathyIndex):
    """Apply biophysical feature adjustments to the base prediction."""
    cai_adj = cdsCAI * 0.1
    structure_adj = (abs(utrCdsStructureMFE) + abs(fivepCdsStructureMFE) + abs(threepCdsStructureMFE)) / 3 * 0.01
    bottleneck_adj = (cdsBottleneckPosition / 100) * 0.05
    at_adj = cdsNucleotideContentAT * 0.05
    hydropathy_adj = (cdsHydropathyIndex + 2) / 4 * 0.05
    return cai_adj + structure_adj + bottleneck_adj + at_adj + hydropathy_adj


def apply_parameters_to_prediction(base_prediction, sequence, parameters):
    """Adjust base prediction using the seven user‑defined biophysical parameters."""
    cdsCAI = parameters.get('cdsCAI', 0.50)
    utrCdsStructureMFE = parameters.get('utrCdsStructureMFE', -25.0)
    fivepCdsStructureMFE = parameters.get('fivepCdsStructureMFE', -15.0)
    threepCdsStructureMFE = parameters.get('threepCdsStructureMFE', -15.0)
    cdsBottleneckPosition = parameters.get('cdsBottleneckPosition', 50)
    cdsNucleotideContentAT = parameters.get('cdsNucleotideContentAT', 0.50)
    cdsHydropathyIndex = parameters.get('cdsHydropathyIndex', 0.0)

    adjustment = apply_biophysical_features(cdsCAI, utrCdsStructureMFE, fivepCdsStructureMFE,
                                            threepCdsStructureMFE, cdsBottleneckPosition,
                                            cdsNucleotideContentAT, cdsHydropathyIndex)
    adjusted = np.clip(base_prediction + adjustment, 0, 1)
    return adjusted


# ---------- Flask app ----------
app = Flask(__name__)
CORS(app)

# Global model object – will be initialized on first request
model = None

def initialize_model():
    """
    Initialize the prediction model.
    Currently loads a SimplifiedModel for demonstration.
    To deploy the quantized SENS‑Q model:
        1. train/quantize using quantization/quantize_main.py
        2. save the state_dict
        3. load the architecture with CNNMambaTransformer and call
           build_quantized_model_from_state_dict()
        4. replace `model` with the quantized instance
    """
    global model
    # Path to the trained checkpoint (if available)
    checkpoint_path = os.path.join(os.path.dirname(__file__), '..', '..', 'best_model.pth')
    try:
        if os.path.exists(checkpoint_path):
            # Example of loading the FP32 model (replace with quantized loading for deployment)
            # model = CNNMambaTransformer(k=3, num_bio_features=7, ...)
            # model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['model_state_dict'])
            # For now, use the simplified predictor
            model = SimplifiedModel()
            print("Simplified predictor loaded (replace with quantized SENS‑Q model for production)")
        else:
            model = SimplifiedModel()
            print("Simplified predictor loaded (checkpoint not found)")
    except Exception as e:
        print(f"Failed to load full model: {e}. Using simplified predictor.")
        model = SimplifiedModel()


@app.before_request
def before_first_request():
    if model is None:
        initialize_model()


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'message': 'Ribosome Stalling Prediction API running'
    })


@app.route('/api/predict', methods=['POST'])
@app.route('/api/predict/single', methods=['POST'])
def single_prediction():
    try:
        data = request.json
        sequence = data.get('sequence', '').strip().upper()
        parameters = data.get('parameters', {})

        if not sequence:
            return jsonify({'status': 'error', 'message': 'Please provide a DNA sequence'}), 400
        if not all(base in 'ATCG' for base in sequence):
            return jsonify({'status': 'error', 'message': 'Sequence must contain only A, T, C, G'}), 400

        # Predict using the currently loaded model (can be simplified or quantized)
        base_pred = model.predict(sequence)
        final_pred = apply_parameters_to_prediction(base_pred, sequence, parameters)

        return jsonify({
            'status': 'success',
            'prediction': float(final_pred),
            'sequence_length': len(sequence),
            'gc_content': (sequence.count('G') + sequence.count('C')) / len(sequence) if sequence else 0,
            'parameters_used': parameters,
            'message': 'Prediction completed (demo mode)'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/model/evaluation', methods=['GET'])
def model_evaluation():
    """
    Provide evaluation metrics (simulated data for demo).
    Replace with real evaluation using the quantized model on the test set.
    """
    test_data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed_data', 'test_Ecoli_data.csv')
    if os.path.exists(test_data_path):
        data = pd.read_csv(test_data_path)
        targets = data['Protein'].values
        if targets.max() > 1e6:
            targets = targets / 1e6
    else:
        targets = np.random.normal(22.5, 8.0, 500)

    # Generate predictions that roughly match the paper's R²
    np.random.seed(42)
    expected_r2 = 0.848
    error_var = np.var(targets) * (1 - expected_r2)
    predictions = targets + np.random.normal(0, np.sqrt(error_var), len(targets))
    r2 = r2_score(targets, predictions)
    residuals = targets - predictions
    errors = np.abs(residuals)

    return jsonify({
        'status': 'success',
        'r2_score': float(r2),
        'targets': targets.tolist(),
        'predictions': predictions.tolist(),
        'residuals': residuals.tolist(),
        'errors': errors.tolist(),
        'sample_count': len(targets),
        'message': 'Simulated evaluation (replace with real quantized model metrics)'
    })


if __name__ == '__main__':
    print("Starting Ribosome Stalling Prediction API (demo with simplified model)")
    print("Access at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
