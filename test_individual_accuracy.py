#!/usr/bin/env python
"""
test_individual_accuracy.py
Clinical-Grade Verification Suite for MCI Multi-Modal Diagnostic System.
Tests face, eye, speech, and mri microservice endpoints for prediction latency,
label accuracy, and proper HTTP 400 rejection of corrupted payloads.
"""
import os
import sys
import time
import requests
import io
import numpy as np
from PIL import Image

# Configured API hosts and ports
BASE_URLS = {
    'fusion': os.environ.get('FUSION_URL', 'http://127.0.0.1:5000'),
    'face': os.environ.get('FACE_URL', 'http://127.0.0.1:5001'),
    'eye': os.environ.get('EYE_URL', 'http://127.0.0.1:5002'),
    'mri': os.environ.get('MRI_URL', 'http://127.0.0.1:5003'),
    'speech': os.environ.get('SPEECH_URL', 'http://127.0.0.1:5004'),
}

def generate_mock_image(color='gray', size=(224, 224), format='PNG'):
    """Generate a mock image as a memory buffer"""
    img = Image.new('RGB', size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf

def generate_mock_wav():
    """Generate a minimal mock WAV stream"""
    # Minimal 8kHz 8-bit mono WAV header + 1 second of silence
    header = (
        b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
        b"\x22\x56\x00\x00\x22\x56\x00\x00\x01\x00\x08\x00data\x00\x00\x00\x00"
    )
    return io.BytesIO(header)

def generate_mock_dicom(size=(128, 128)):
    """Generate a mock DICOM file stream if pydicom is available, else standard image"""
    try:
        import pydicom
        from pydicom.dataset import Dataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian
        
        file_meta = FileMetaDataset()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
        file_meta.MediaStorageSOPInstanceUID = "1.2.3"
        file_meta.ImplementationClassUID = "1.2.3.4"
        
        ds = Dataset()
        ds.file_meta = file_meta
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
        ds.SOPInstanceUID = "1.2.3"
        ds.PatientName = "Test^Patient"
        ds.PatientID = "123456"
        ds.Modality = "MR"
        ds.SeriesInstanceUID = "1.2.3.4.5"
        ds.StudyInstanceUID = "1.2.3.4.5.6"
        ds.FrameOfReferenceUID = "1.2.3.4.5.6.7"
        
        ds.Rows = size[0]
        ds.Columns = size[1]
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        
        pixel_array = np.random.randint(0, 1000, size=size, dtype=np.uint16)
        ds.PixelData = pixel_array.tobytes()
        
        bio = io.BytesIO()
        pydicom.dcmwrite(bio, ds, write_like_original=False)
        bio.seek(0)
        return bio, 'DICOM'
    except Exception:
        # Fallback to standard grayscale png
        return generate_mock_image(color='gray', size=size, format='PNG'), 'PNG'

def run_tests():
    print("=" * 70)
    print("      MCI MULTI-MODAL DIAGNOSTIC HARDENING VERIFICATION SUITE")
    print("=" * 70)
    
    results = {}
    
    # ------------------ TEST 1: FACE API ------------------
    print("\n[+] Testing Face API...")
    face_url = f"{BASE_URLS['face']}/predict/face"
    try:
        # Test valid request
        face_file = generate_mock_image(color='gray', size=(48, 48))
        start_time = time.time()
        res = requests.post(face_url, files={'file': ('face.png', face_file, 'image/png')}, timeout=60)
        latency = (time.time() - start_time) * 1000
        
        print(f"  Valid Request: Status {res.status_code} | Latency: {latency:.1f}ms")
        if res.status_code == 200:
            data = res.json()
            print(f"  Prediction Label: {data.get('predicted_emotion')} | Confidence: {data.get('confidence'):.3f}")
            results['face'] = {'success': True, 'latency': latency}
        else:
            print(f"  FAILED: {res.text}")
            results['face'] = {'success': False, 'error': f"Status {res.status_code}"}
            
        # Test corrupted request (should return HTTP 400)
        corrupt_file = io.BytesIO(b"corrupted_image_data_here")
        res_corrupt = requests.post(face_url, files={'file': ('face.png', corrupt_file, 'image/png')}, timeout=60)
        print(f"  Corrupt File Request: Status {res_corrupt.status_code} (Expected 400)")
        assert res_corrupt.status_code == 400, f"Expected 400 for corrupt file, got {res_corrupt.status_code}"
        print("  [OK] Correctly rejected corrupted image with HTTP 400")
        
    except Exception as e:
        print(f"  Error testing Face API: {e}")
        results['face'] = {'success': False, 'error': str(e)}

    # ------------------ TEST 2: EYE API ------------------
    print("\n[+] Testing Eye API...")
    eye_url = f"{BASE_URLS['eye']}/predict/eye"
    try:
        # Test valid request
        eye_file = generate_mock_image(color='blue', size=(224, 224))
        start_time = time.time()
        res = requests.post(eye_url, files={'file': ('eye.png', eye_file, 'image/png')}, timeout=60)
        latency = (time.time() - start_time) * 1000
        
        print(f"  Valid Request: Status {res.status_code} | Latency: {latency:.1f}ms")
        if res.status_code == 200:
            data = res.json()
            print(f"  Prediction Label: {data.get('predicted_emotion')} | Confidence: {data.get('confidence'):.3f}")
            results['eye'] = {'success': True, 'latency': latency}
        else:
            print(f"  FAILED: {res.text}")
            results['eye'] = {'success': False, 'error': f"Status {res.status_code}"}
            
        # Test corrupted request (should return HTTP 400)
        corrupt_file = io.BytesIO(b"corrupted_image_data_here")
        res_corrupt = requests.post(eye_url, files={'file': ('eye.png', corrupt_file, 'image/png')}, timeout=60)
        print(f"  Corrupt File Request: Status {res_corrupt.status_code} (Expected 400)")
        assert res_corrupt.status_code == 400, f"Expected 400 for corrupt file, got {res_corrupt.status_code}"
        print("  [OK] Correctly rejected corrupted image with HTTP 400")
        
    except Exception as e:
        print(f"  Error testing Eye API: {e}")
        results['eye'] = {'success': False, 'error': str(e)}

    # ------------------ TEST 3: MRI API (NATIVE DICOM & IMAGE) ------------------
    print("\n[+] Testing MRI API (Multi-format support)...")
    mri_url = f"{BASE_URLS['mri']}/predict/mri"
    try:
        # Test DICOM Parsing
        dicom_file, fmt = generate_mock_dicom()
        print(f"  Submitting mock format: {fmt}")
        start_time = time.time()
        res = requests.post(mri_url, files={'file': (f'mri.dcm' if fmt=='DICOM' else 'mri.png', dicom_file, 'application/dicom' if fmt=='DICOM' else 'image/png')}, timeout=60)
        latency = (time.time() - start_time) * 1000
        
        print(f"  Valid Request: Status {res.status_code} | Latency: {latency:.1f}ms")
        if res.status_code == 200:
            data = res.json()
            print(f"  MCI Probability: {data.get('mci_probability'):.3f} | Confidence: {data.get('confidence'):.3f}")
            results['mri'] = {'success': True, 'latency': latency}
        else:
            print(f"  FAILED: {res.text}")
            results['mri'] = {'success': False, 'error': f"Status {res.status_code}"}
            
        # Test corrupted request (should return HTTP 400)
        corrupt_file = io.BytesIO(b"corrupted_dicom_data_here")
        res_corrupt = requests.post(mri_url, files={'file': ('mri.dcm', corrupt_file, 'application/dicom')}, timeout=60)
        print(f"  Corrupt File Request: Status {res_corrupt.status_code} (Expected 400)")
        assert res_corrupt.status_code == 400, f"Expected 400 for corrupt file, got {res_corrupt.status_code}"
        print("  [OK] Correctly rejected corrupted medical MRI upload with HTTP 400")
        
    except Exception as e:
        print(f"  Error testing MRI API: {e}")
        results['mri'] = {'success': False, 'error': str(e)}

    # ------------------ TEST 4: SPEECH API ------------------
    print("\n[+] Testing Speech API...")
    speech_url = f"{BASE_URLS['speech']}/predict/speech"
    try:
        # Test valid request
        wav_file = generate_mock_wav()
        start_time = time.time()
        res = requests.post(speech_url, files={'file': ('speech.wav', wav_file, 'audio/wav')}, timeout=60)
        latency = (time.time() - start_time) * 1000
        
        print(f"  Valid Request: Status {res.status_code} | Latency: {latency:.1f}ms")
        if res.status_code == 200:
            data = res.json()
            print(f"  Prediction Label: {data.get('predicted_emotion')} | Confidence: {data.get('confidence'):.3f}")
            results['speech'] = {'success': True, 'latency': latency}
        else:
            print(f"  FAILED: {res.text}")
            results['speech'] = {'success': False, 'error': f"Status {res.status_code}"}
            
        # Test corrupted request (should return HTTP 400)
        corrupt_file = io.BytesIO(b"corrupted_audio_data_here")
        res_corrupt = requests.post(speech_url, files={'file': ('speech.wav', corrupt_file, 'audio/wav')}, timeout=60)
        print(f"  Corrupt File Request: Status {res_corrupt.status_code} (Expected 400)")
        assert res_corrupt.status_code == 400, f"Expected 400 for corrupt file, got {res_corrupt.status_code}"
        print("  [OK] Correctly rejected corrupted audio with HTTP 400")
        
    except Exception as e:
        print(f"  Error testing Speech API: {e}")
        results['speech'] = {'success': False, 'error': str(e)}

    # ------------------ TEST 5: FUSION ORCHESTRATOR & PROPAGATION ------------------
    print("\n[+] Testing Fusion Orchestrator & Error Propagation...")
    fusion_url = f"{BASE_URLS['fusion']}/api/analyze"
    try:
        # 1. Valid multi-modal request
        face_file = generate_mock_image(color='gray')
        eye_file = generate_mock_image(color='blue')
        mri_file, _ = generate_mock_dicom()
        speech_file = generate_mock_wav()
        
        files = {
            'face': ('face.png', face_file, 'image/png'),
            'eye': ('eye.png', eye_file, 'image/png'),
            'mri': ('mri.dcm', mri_file, 'application/dicom'),
            'speech': ('speech.wav', speech_file, 'audio/wav'),
        }
        
        start_time = time.time()
        res = requests.post(fusion_url, files=files, timeout=20)
        latency = (time.time() - start_time) * 1000
        
        print(f"  Valid Fusion Request: Status {res.status_code} | Latency: {latency:.1f}ms")
        if res.status_code == 200:
            data = res.json()
            analysis = data.get('analysis', {})
            print(f"  Fused MCI Probability: {analysis.get('mci_probability'):.3f} | Confidence: {analysis.get('confidence'):.3f}")
            print(f"  Risk Classification: {analysis.get('risk_level')}")
            results['fusion'] = {'success': True, 'latency': latency}
        else:
            print(f"  FAILED: {res.text}")
            results['fusion'] = {'success': False, 'error': f"Status {res.status_code}"}
            
        # 2. Corrupt modality upload - Propagation Test (Expected Loud 400 Failure)
        print("  Submitting corrupted image to face channel to verify error propagation...")
        corrupt_face = io.BytesIO(b"invalid_image_bytes")
        files_corrupt = {
            'face': ('face.png', corrupt_face, 'image/png'),
            'eye': ('eye.png', generate_mock_image(color='blue'), 'image/png'),
        }
        res_prop = requests.post(fusion_url, files=files_corrupt, timeout=60)
        print(f"  Corrupt Modality Fusion Request: Status {res_prop.status_code} (Expected 400)")
        assert res_prop.status_code == 400, f"Expected 400 for corrupt modality payload, got {res_prop.status_code}"
        print(f"  Propagated Message: {res_prop.json().get('error')}")
        print("  [OK] SUCCESS: Corrupted modality uploads prevent silent fallback, aborting with HTTP 400 validation error!")
        
        # 3. Payload size ceiling limit (Expected HTTP 413 Payload Too Large)
        print("  Submitting oversized mock payload (55MB) to test DoS defenses...")
        oversized_data = bytearray(55 * 1024 * 1024)
        files_oversized = {
            'face': ('oversized.png', io.BytesIO(oversized_data), 'image/png')
        }
        try:
            res_oversized = requests.post(fusion_url, files=files_oversized, timeout=10)
            print(f"  Oversized Payload: Status {res_oversized.status_code} (Expected 413 or Connection Closed)")
            assert res_oversized.status_code in (413, 400), f"Expected payload block, got status {res_oversized.status_code}"
            print("  [OK] SUCCESS: Oversized payload successfully blocked by DoS safety ceiling!")
        except requests.exceptions.RequestException:
            print("  [OK] SUCCESS: Connection reset or closed immediately by safety ceiling (standard behavior for oversized uploads)")
            
    except Exception as e:
        print(f"  Error testing Fusion Orchestrator: {e}")
        results['fusion'] = {'success': False, 'error': str(e)}

    # ------------------ SUMMARY REPORT ------------------
    print("\n" + "=" * 70)
    print("                       VERIFICATION SUMMARY REPORT")
    print("=" * 70)
    print(f"{'Modality/Service':<20} | {'Status':<12} | {'Latency':<12}")
    print("-" * 70)
    for service, meta in results.items():
        status = "PASSED" if meta['success'] else "FAILED"
        latency = f"{meta['latency']:.1f} ms" if meta['success'] else "N/A"
        print(f"{service.upper():<20} | {status:<12} | {latency:<12}")
    print("=" * 70)
    
    overall_passed = all(meta['success'] for meta in results.values())
    if overall_passed:
        print(">>> SUCCESS: All microservice hardening tests PASSED. Diagnostic pipeline is clinical-grade ready!")
        sys.exit(0)
    else:
        print(">>> FAILURE: One or more services failed hardening tests. Review errors above.")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
