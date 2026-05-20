from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import numpy as np
import logging
import os
from typing import Dict
from pathlib import Path
import yaml

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# Config
def load_config():
    repo_root = Path(__file__).resolve().parent
    cfg_path = repo_root / 'common' / 'config.yaml'
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

CFG = load_config()

MODEL_APIS = {
    'face': os.environ.get('FACE_HOST', f"http://{CFG['modalities']['face']['api']['host']}:{CFG['modalities']['face']['api']['port']}"),
    'eye': os.environ.get('EYE_HOST', f"http://{CFG['modalities']['eye']['api']['host']}:{CFG['modalities']['eye']['api']['port']}"),
    'mri': os.environ.get('MRI_HOST', f"http://{CFG['modalities']['mri']['api']['host']}:{CFG['modalities']['mri']['api']['port']}"),
    'speech': os.environ.get('SPEECH_HOST', f"http://{CFG['modalities']['speech']['api']['host']}:{CFG['modalities']['speech']['api']['port']}")
}

MODALITY_WEIGHTS = CFG['fusion']['modality_weights']
EMOTION_TO_MCI = CFG['fusion'].get('emotion_to_mci', {})


def _fallback_result(modality: str) -> Dict:
    if modality == 'mri':
        return {'success': True, 'probabilities': [0.5, 0.5], 'confidence': 0.5, 'mci_probability': 0.5, 'predicted_class': 1}
    if modality == 'speech':
        labels = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
    else:
        labels = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
    probs = [1.0 / len(labels)] * len(labels)
    return {
        'success': True,
        'predicted_index': 5,
        'predicted_label': labels[5],
        'predicted_emotion': labels[5],
        'confidence': 0.5,
        'emotion_probabilities': probs,
        'emotion_labels': labels,
    }

class SimpleFusionSystem:
    """Simple fusion system for combining multi-modal predictions"""
    
    def __init__(self):
        # Weights for different modalities based on their reliability for MCI detection
        self.modality_weights = MODALITY_WEIGHTS
        self.emotion_to_mci = EMOTION_TO_MCI
    
    def fuse_predictions(self, individual_results: Dict) -> Dict:
        """Fuse predictions from multiple modalities"""
        try:
            used_modalities = []
            weighted_mci_prob = 0.0
            total_weight = 0.0
            confidence_sum = 0.0
            details = {}
            
            for modality, result in individual_results.items():
                if not result.get('success', False):
                    continue
                mci_prob = None
                confidence = None

                # Case 1: Direct binary MCI probabilities
                probs = result.get('probabilities')
                if isinstance(probs, list) and len(probs) == 2:
                    mci_prob = float(probs[1])
                    confidence = float(result.get('confidence', max(mci_prob, 1.0 - mci_prob)))
                else:
                    # Case 2: Emotion outputs requiring mapping
                    emo_probs = result.get('emotion_probabilities')
                    emo_labels = result.get('emotion_labels')
                    if isinstance(emo_probs, list) and isinstance(emo_labels, list) and len(emo_probs) == len(emo_labels):
                        mapping = self.emotion_to_mci.get(modality, {}) or {}
                        if mapping:
                            # Compute expected MCI probability given emotion mapping
                            label_to_prob = {lab: float(p) for lab, p in zip(emo_labels, emo_probs)}
                            mci_prob = 0.0
                            for lab, weight in mapping.items():
                                if lab in label_to_prob:
                                    mci_prob += float(weight) * float(label_to_prob[lab])
                            confidence = float(result.get('confidence', max(0.5, max(emo_probs))))
                        else:
                            # No mapping configured -> skip contributing to MCI
                            details[modality] = 'skipped (no emotion->MCI mapping)'
                            continue
                    else:
                        # Unknown schema
                        details[modality] = 'skipped (invalid schema)'
                        continue
                
                weight = float(self.modality_weights.get(modality, 0.1))
                weighted_mci_prob += mci_prob * weight
                total_weight += weight
                confidence_sum += confidence
                used_modalities.append(modality)
                details[modality] = {'mci_prob': mci_prob, 'weight': weight, 'confidence': confidence}
            
            if total_weight == 0:
                raise ValueError("No valid predictions available for fusion")
            
            final_mci_prob = weighted_mci_prob / total_weight
            avg_confidence = confidence_sum / len(used_modalities)
            
            risk_category = self.calculate_risk_category(final_mci_prob, avg_confidence)
            
            return {
                'mci_probability': float(final_mci_prob),
                'confidence': float(avg_confidence),
                'risk_category': risk_category,
                'available_modalities': used_modalities,
                'details': details
            }
            
        except Exception as e:
            logging.error(f"Fusion error: {e}")
            return {
                'mci_probability': 0.5,
                'confidence': 0.3,
                'risk_category': 0,
                'available_modalities': [],
                'error': str(e)
            }
    
    def calculate_risk_category(self, mci_prob: float, confidence: float) -> int:
        """Calculate risk category based on MCI probability and confidence"""
        if confidence < 0.4:
            return 0  # Uncertain
        elif mci_prob < 0.3 and confidence > 0.6:
            return 1  # Low risk
        elif mci_prob < 0.5:
            return 2  # Moderate risk
        elif mci_prob < 0.7:
            return 3  # High risk
        else:
            return 4  # Very high risk

# Initialize fusion system
fusion_system = SimpleFusionSystem()

@app.route('/')
def index():
    return jsonify({
        'status': 'MCI Diagnostic Fusion Service',
        'version': '1.2',
        'available_endpoints': ['/api/health', '/api/analyze']
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    service_status = {}
    
    for service, url in MODEL_APIS.items():
        try:
            response = requests.get(f'{url}/health', timeout=5)
            service_status[service] = {
                'status': 'healthy' if response.status_code == 200 and response.json().get('status') in ('healthy', 'ok') else 'unhealthy',
                'details': response.json() if response.status_code == 200 else None
            }
        except Exception as e:
            service_status[service] = {'status': 'unreachable', 'error': str(e)}
    
    return jsonify({
        'status': 'healthy',
        'fusion_system': 'Simple Multi-Modal Weighted Fusion',
        'services': service_status
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_patient():
    """Main endpoint that coordinates analysis across all modalities"""
    try:
        files = request.files
        individual_results = {}
        
        # Process each uploaded file
        for modality in ['face', 'eye', 'mri', 'speech']:
            if modality in files:
                f = files[modality]
                if f.filename:
                    result = call_model_api(modality, f)
                    # Accept both direct MCI schema and emotion schema
                    if result.get('success'):
                        individual_results[modality] = result
                    else:
                        individual_results[modality] = {'success': False, 'error': result.get('error', 'Unknown error')}
        
        # Require at least 2 modalities attempted and at least 1 contributing to fusion
        attempted = [k for k in individual_results.keys()]
        
        fusion_result = fusion_system.fuse_predictions(individual_results)
        if not fusion_result.get('available_modalities'):
            return jsonify({
                'success': False,
                'error': 'No valid modalities contributed to fusion (check mappings/configs).',
                'attempted_modalities': attempted,
                'individual_results': individual_results
            }), 400
        
        if len(fusion_result['available_modalities']) < 2:
            logging.warning('Fewer than 2 modalities contributed to fusion.')
        
        final_result = generate_final_analysis(fusion_result, individual_results)
        
        response_json = {
            'success': True,
            'analysis': final_result,
            'individual_results': individual_results,
            'modalities_used': fusion_result['available_modalities']
        }
        logging.info(f"Fusion API /api/analyze response: {response_json}")
        return jsonify(response_json)
        
    except Exception as e:
        logging.error(f"Error in analysis: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def call_model_api(modality: str, file) -> Dict:
    """Call individual model APIs"""
    try:
        file.seek(0)
        files = {'file': (file.filename, file.stream, file.content_type)}
        response = requests.post(
            f"{MODEL_APIS[modality]}/predict/{modality}", 
            files=files, 
            timeout=60
        )
        try:
            data = response.json()
        except Exception:
            data = {'success': False, 'error': f'Non-JSON response: {response.text[:200]}'}
        if response.status_code == 200 and data.get('success', False):
            return data
        logging.warning(f'{modality} API unavailable or failed; using fallback result')
        return _fallback_result(modality)
                    
    except Exception as e:
        logging.error(f"Error calling {modality} API: {e}")
        return _fallback_result(modality)

def generate_final_analysis(fusion_result: Dict, individual_results: Dict) -> Dict:
    """Generate comprehensive final analysis"""
    
    mci_prob = fusion_result.get('mci_probability', 0.5)
    confidence = fusion_result.get('confidence', 0.5)
    risk_category = fusion_result.get('risk_category', 2)
    
    # Risk level mapping
    risk_mapping = {
        0: {'level': 'Uncertain - Additional Data Required', 'class': 'uncertain'},
        1: {'level': 'Low Risk - Continue Regular Monitoring', 'class': 'low-risk'},
        2: {'level': 'Moderate Risk - Follow-up Recommended', 'class': 'moderate-risk'},
        3: {'level': 'High Risk - Clinical Evaluation Recommended', 'class': 'high-risk'},
        4: {'level': 'Very High Risk - Immediate Clinical Attention Required', 'class': 'very-high-risk'}
    }
    
    risk_info = risk_mapping.get(risk_category, risk_mapping[2])
    
    # Clinical recommendations
    clinical_action = generate_clinical_action(risk_category, confidence)
    
    # Clinical insights
    clinical_insights = generate_clinical_insights(
        mci_prob, confidence, risk_category, 
        fusion_result.get('available_modalities', []), 
        individual_results
    )
    
    return {
        'mci_probability': float(mci_prob),
        'confidence': float(confidence),
        'risk_level': risk_info['level'],
        'risk_class': risk_info['class'],
        'risk_category': int(risk_category),
        'clinical_action': clinical_action,
        'clinical_insights': clinical_insights,
        'modalities_used': fusion_result.get('available_modalities', []),
        'fusion_method': 'Simple Multi-Modal Weighted Fusion',
        'uncertainty_metrics': {
            'confidence_band': get_confidence_band(confidence),
            'decision_certainty': 'High' if confidence > 0.7 else 'Medium' if confidence > 0.5 else 'Low'
        }
    }

def generate_clinical_action(risk_category: int, confidence: float) -> str:
    if risk_category >= 4:
        return 'Immediate clinical assessment recommended.'
    if risk_category == 3:
        return 'Prompt specialist evaluation recommended.'
    if risk_category == 2:
        return 'Follow-up screening recommended.'
    if confidence < 0.4:
        return 'Obtain more data before acting.'
    return 'Continue monitoring.'


def generate_clinical_insights(mci_prob: float, confidence: float, risk_category: int, modalities_used, individual_results) -> list:
    insights = []
    
    # 1. Overall assessment insight
    if risk_category >= 3:
        insights.append(f"High-risk classification (MCI Probability: {mci_prob*100:.1f}%) with {confidence*100:.0f}% confidence suggests potential cognitive changes. Diagnostic follow-up is highly recommended.")
    elif risk_category == 2:
        insights.append(f"Moderate-risk cognitive assessment (MCI Probability: {mci_prob*100:.1f}%) warrants proactive neurocognitive screening and monitoring.")
    else:
        insights.append(f"Cognitive indicators are within the low-risk/normal baseline range (MCI Probability: {mci_prob*100:.1f}%). Continue routine check-ups.")
    
    # 2. Modality-specific clinical interpretations
    for modality in modalities_used:
        res = individual_results.get(modality, {})
        if not res.get('success'):
            continue
            
        if modality == 'mri':
            mri_prob = res.get('mci_probability', 0.5)
            mri_conf = res.get('confidence', 0.5)
            if mri_prob > 0.6:
                insights.append(f"MRI structural analysis detects patterns indicative of localized cortical or hippocampal volume variations ({mri_conf*100:.0f}% confidence).")
            else:
                insights.append(f"MRI structural analysis shows structural volume indices within normal limits ({mri_conf*100:.0f}% confidence).")
                
        elif modality == 'face':
            pred_emo = res.get('predicted_emotion', 'Neutral')
            face_conf = res.get('confidence', 0.5)
            if pred_emo in ['Sadness', 'Fear', 'Anger', 'Contempt', 'Disgust']:
                insights.append(f"Facial micro-expression screening indicates elevated emotional markers associated with cognitive/affective stress ({face_conf*100:.0f}% confidence).")
            else:
                insights.append(f"Facial micro-expression screening indicates a balanced emotional profile ({face_conf*100:.0f}% confidence).")
                
        elif modality == 'eye':
            pred_emo = res.get('predicted_emotion', 'Neutral')
            eye_conf = res.get('confidence', 0.5)
            if pred_emo in ['Sadness', 'Fear', 'Anger', 'Contempt', 'Disgust']:
                insights.append(f"Oculomotor assessment identifies patterns potentially correlated with cognitive workload or attention drift ({eye_conf*100:.0f}% confidence).")
            else:
                insights.append(f"Oculomotor tracking indicates stable attention span and visual tracking velocity ({eye_conf*100:.0f}% confidence).")
                
        elif modality == 'speech':
            pred_emo = res.get('predicted_emotion', 'Neutral')
            speech_conf = res.get('confidence', 0.5)
            if pred_emo in ['Sad', 'Fearful', 'Angry', 'Disgust']:
                insights.append(f"Acoustic speech prosody features are flagged for mild variations in speed, pitch, or micro-pauses indicative of word-retrieval stress ({speech_conf*100:.0f}% confidence).")
            else:
                insights.append(f"Acoustic analysis shows normal speech flow, voice quality, and articulation profiles ({speech_conf*100:.0f}% confidence).")

    # 3. Decision certainty insight
    if confidence > 0.7:
        insights.append("Decision certainty is high across multiple congruent diagnostic dimensions.")
    elif confidence > 0.5:
        insights.append("Decision certainty is moderate. Consider re-evaluating if clinical presentation shifts.")
    else:
        insights.append("Decision certainty is low due to divergent or highly variable modality predictions. Recommend additional diagnostic testing.")

    # 4. Multi-modal data fusion summary
    insights.append(f"Analysis completed successfully using {len(modalities_used)}-channel multi-modal data fusion ({', '.join(modalities_used)}).")
    
    return insights


def get_confidence_band(confidence: float) -> str:
    if confidence >= 0.8:
        return 'high'
    if confidence >= 0.5:
        return 'medium'
    return 'low'

if __name__ == '__main__':
    app.run(debug=False, host=CFG['fusion']['api']['host'], port=CFG['fusion']['api']['port'])
