Model_Fusion
============

Fusion service that orchestrates the per-modality model APIs and produces final MCI analysis.

Contents to copy:
- Files from `model_final/` including `main.py`, `enhanced_mci_system.py`, `multimodal_alignment.py`.

Deployment notes:
- The fusion service expects modality APIs to be reachable (e.g., Face: `http://face:5000/predict/face`, Eye: `http://eye:5000/predict/eye`, Speech: `http://speech:5000/predict/speech`, MRI: `http://mri:5000/predict/mri`).
- Configure `common/config.yaml` or environment variables with the hostnames/ports for each modality.

Quick start:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m main
```

Docker and orchestration:
- Use Docker Compose to bring up modality services and the fusion service. Example `docker-compose.yml` template is included.
