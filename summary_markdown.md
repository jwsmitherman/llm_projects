# PCIS Model Service – Python 3.12 Migration & Infrastructure Changes
## Business Summary

---

### Overview

This effort modernized the PCIS PCS Modeling Prediction Service to run on Python 3.12, resolved a series of infrastructure and deployment blockers, and improved the reliability of model delivery in both local development and CI/CD environments. The work spanned two repositories: `pcis-model-service` (the API) and `pcs-model-training` (the model package).

---

### What Changed and Why

#### 1. AWS Credential Workflow (Developer Experience)
**What:** The local AWS credential setup process was broken — developers could not authenticate to AWS from their workstations, which blocked all downstream work including model downloads.

**Why it matters:** Without valid AWS credentials, the service cannot download model files from S3, meaning the API cannot start. This was the root blocker for local development and testing.

**Resolution:** Fixed the `~/.aws/credentials` file setup process so that credentials obtained from the USCIS ICAM SSO portal (myaccess.uscis.dhs.gov) are correctly written to disk. Because these credentials rotate every two hours, a repeatable workflow was established for refreshing them.

---

#### 2. Model Files Now Download Automatically at Startup
**What:** Previously, AI/ML model files had to be manually downloaded and placed in specific directories before the service could start. This was a fragile, manual process.

**Why it matters:** In containerized and cloud environments, model files cannot be baked into the application — they must be retrieved at runtime from secure storage (AWS S3). Manual steps are error-prone and do not scale.

**Resolution:** The service now automatically downloads required model files from S3 when it starts up, before the model inference engine initializes. This works in both local development and production container environments.

---

#### 3. Model Directory Alignment
**What:** The application had hardcoded references to `/build_assets` as the location for model files, but the actual download location was different — causing the service to start but fail to find its models.

**Why it matters:** A mismatch between where models are downloaded and where the application looks for them causes startup failures and silent errors in production.

**Resolution:** Updated all configuration and code to dynamically resolve the correct model directory using Python's standard package resource system, ensuring consistency across all environments (local, Docker, and cloud).

---

#### 4. Docker Container Build Fixes
**What:** The Docker image build process had several failures related to the new Python 3.12 environment — including an unavailable package version (`onnxruntime`), missing system tools, and a Python binary naming difference (`python` vs `python3`) in the hardened Alpine base image.

**Why it matters:** A broken Docker build means the service cannot be deployed. These were blocking the entire CI/CD pipeline from producing a deployable artifact.

**Resolution:** Updated the Dockerfile to use compatible package versions, removed unnecessary system package installations (already present in the hardened base image), and corrected Python binary references.

---

#### 5. PySpark Dependency for Model Loading
**What:** The ANM (A-Number Matching) model was serialized using PySpark and could not be loaded in the updated environment due to version incompatibilities.

**Why it matters:** If the model cannot be loaded, the service cannot serve predictions for A-Number matching, which is a core function of the PCIS modeling service.

**Resolution:** Installed the correct version of PySpark (`4.0.1`) matching the version used to train the model. For local development, the service is configured to skip loading the model (`ENV_NAME=dev`) to allow API development and testing without requiring the full model stack.

---

#### 6. Automated Test Suite Fixes (CI/CD Pipeline)
**What:** The automated test pipeline (Harness CI) was failing because tests attempted to load actual AI model files from disk — files that do not exist in the CI environment because they are large binary assets stored in S3.

**Why it matters:** A broken test pipeline blocks all code from being promoted through environments (nonprod → staging → production). Every developer's changes were blocked.

**Resolution:** Updated the test suite to use mock (simulated) versions of the model inference components. This allows tests to verify application logic without requiring the actual multi-hundred-megabyte model files to be present. Two test files were updated: the ANM model test and the Dedup (deduplication) model test.

---

#### 7. Model Package Path Standardization
**What:** The `pcs-models` Python package (which contains the shared model logic used by both the training and serving repositories) had hardcoded paths that pointed to legacy container directories.

**Why it matters:** Hardcoded paths break when the application runs in different environments (developer laptops, Docker containers, cloud infrastructure).

**Resolution:** Updated the package to use Python's built-in package resource resolution, making path references portable and environment-agnostic.

---

### Summary of Repositories Modified

| Repository | Area Changed |
|---|---|
| `pcis-model-service` | Startup model download, settings/configuration, Docker build, API initialization |
| `pcs-model-training` | Model package paths, test suite mocking, CI pipeline compatibility |

---

### Technical Migration Details

#### ANM Model: From Encrypted Pickle (.enc) to Distilled Pickle (.pkl)

**Old approach:** The ANM (A-Number Matching) model was stored as an encrypted pickle file (`anm_model.pkl.enc`). At startup, the service had to:
1. Download the encrypted file from S3
2. Decrypt it using an encryption key (`ENC_KEY` / `ENC_KEY_NONPROD`) passed as an environment variable
3. Load the decrypted PySpark ML pipeline object into memory

This meant the service required Java (for PySpark), the encryption key at runtime, and a full PySpark environment just to serve predictions.

**New approach:** The model was distilled and re-serialized as `anm_distilled_312.pkl` — a plain `joblib`-serialized scikit-learn compatible model. The new loading pattern in `anm.py` is:

```python
from importlib.resources import files

MODEL_PATH = files('pcs_models.models').joinpath('anm_distilled_312.pkl')
model = None

def get_model():
    global model
    if model is None:
        model = joblib.load(MODEL_PATH)
    return model
```

Key changes:
- **No decryption step** — the file is no longer encrypted at rest in this format
- **Lazy loading** — `get_model()` only loads the model on first use, not at import time, preventing startup failures when the file isn't present
- **Path resolution** — uses `importlib.resources.files()` to locate the model within the installed `pcs_models` package regardless of environment, replacing the old hardcoded `/build_assets` path
- **Still requires PySpark** — the distilled model was serialized with PySpark 4.0.1 and still requires it to unpickle, which is why `pyspark==4.0.1` remains in `requirements.txt`

---

#### Dedup Model: From Encrypted Pickle (.enc) to ONNX (.onnx)

**Old approach:** The deduplication model was also stored as an encrypted pickle (`dedup_model.pkl.enc`), with the same decrypt-then-load pattern as ANM.

**New approach:** The model was exported to ONNX (Open Neural Network Exchange) format as `dedup_model.onnx`. ONNX is a portable, open standard for ML models that runs via the `onnxruntime` library — no PySpark, no Java, no decryption needed.

The inference pattern in `dedup.py`:
```python
cls._sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
cls._input_name = cls._sess.get_inputs()[0].name
outputs = cls._sess.run(None, {cls._input_name: X})
```

Key changes:
- **No encryption** — ONNX files are portable binary model artifacts
- **No PySpark dependency** for dedup — `onnxruntime` handles inference natively
- **Thread-safe lazy loading** — `_load_onnx()` uses a class-level lock (`cls._model_lock`) so concurrent requests don't race to load the model
- **Platform constraint** — ONNX models are platform-specific; the Docker image is built for `linux/amd64`, so `onnxruntime` must have a compatible wheel for that platform (this caused the build failure with version 1.22.1, resolved by downgrading)

---

#### Python 3.9 → 3.12 Migration

**Pydantic v1 → v2**

This was the most pervasive code change. Pydantic v2 (required for Python 3.12 compatibility) introduced breaking API changes:

| Area | Old (Pydantic v1) | New (Pydantic v2) |
|---|---|---|
| Import location | `from pydantic import BaseSettings` | `from pydantic_settings import BaseSettings` |
| Model config | `class Config: arbitrary_types_allowed = True` | `model_config = ConfigDict(arbitrary_types_allowed=True)` |
| Field validators | `@validator('field')` | `@field_validator('field', mode="before")` |
| Validator signature | `def val(cls, v)` | `def val(value, field)` (no `cls`) |
| Schema extra | `schema_extra` | `json_schema_extra` |
| Extra fields | `class Config: extra = 'ignore'` | `model_config = ConfigDict(extra="ignore")` |

In `anm.py` specifically, all field validators were updated to use `@field_validator` with `mode="before"`:
```python
# Old (Pydantic v1)
@validator('fullName', pre=True)
def fullName_validator(cls, v):
    ...

# New (Pydantic v2)
@field_validator('fullName', mode="before")
def fullName_validator(value, field):
    ...
```

**`importlib.resources` API change**

Python 3.9 used the older `importlib.resources` API:
```python
# Old (Python 3.9)
import importlib.resources
path = importlib.resources.path('pcs_models.models', 'anm_distilled_312.pkl')
```

Python 3.12 uses the newer `files()` API which returns a `Traversable` object:
```python
# New (Python 3.12)
from importlib.resources import files
MODEL_PATH = files('pcs_models.models').joinpath('anm_distilled_312.pkl')
```

This change was applied in `anm.py`, `s3.py`, `settings.py`, and `manager/__init__.py`.

**`onnxruntime` version**

`onnxruntime==1.22.1` does not have a published `manylinux` wheel for `linux/amd64` (the Docker target platform). Downgraded to `1.21.0` which has full platform support.

**H2O removed**

The old Python 3.9 environment included H2O for certain model training tasks. H2O does not support Python 3.12 and was removed from dependencies entirely. Any models that required H2O were retrained or converted to scikit-learn/ONNX compatible formats.

**PySpark 3.x → 4.0.1**

PySpark 4.x dropped support for Python 3.8/3.9 and added native Python 3.12 support. The upgrade was required but introduced a serialization compatibility issue — pickle files created with PySpark 3.x cannot be loaded by PySpark 4.x due to changes in `pyspark.serializers`. The ANM model was retrained and re-serialized under PySpark 4.0.1 to resolve this.

---

### How to Run Locally (Updated Process)

#### One-Time Setup

**1. Install AWS CLI**
```bash
brew install awscli
mkdir -p ~/.aws
```

**2. Set environment variables** — add these to your `~/.bashrc` or `~/.zshrc` and run `source ~/.bashrc`:
```bash
export ENV_NAME=dev
export AWS_ENV=NONPROD
export ENC_KEY=abc123
export ENC_KEY_NONPROD=abc123
export AUTO_LOAD_MODELS=0
export SKIP_REGISTERING_TOKENS=1
```

> **Note:** `ENC_KEY` values above are placeholders for local/GDS testing only. Real keys are required for full model loading.

---

#### Every ~2 Hours (AWS Credential Refresh)

AWS credentials from the USCIS ICAM portal expire every two hours. When they expire, refresh them:

1. Go to [https://login.uscis.dhs.gov/sso/XUI/#login/](https://login.uscis.dhs.gov/sso/XUI/#login/)
2. Log in and navigate to the ICAM Dashboard → select **AWS** → choose your account/role → select **API KEY**
3. Copy all 3 export lines from the AWS Response page
4. Run `make creds` in the terminal and paste the 3 values when prompted

To verify credentials are working:
```bash
aws sts get-caller-identity
```

---

#### Starting the API

From the root of `pcis-model-service`:
```bash
uvicorn app:awsgi_app
```

The API will start on `http://127.0.0.1:8000`. Swagger documentation is available at the same address.

On first start (or after model files are deleted), the service will automatically download model files from S3 before initializing. You will see log lines like:
```
INFO:pcs_models.utils.s3:Downloading mock_data/anm_distilled_312.pkl ...
INFO:pcs_models.utils.s3:Download complete...
```

To test changes without restarting, use the `--reload` flag:
```bash
uvicorn app:awsgi_app --reload
```

---

#### API Authorization for Local Testing

With `ENV_NAME=dev` set, you can authenticate using any string in the Swagger UI. Click **Authorize** and enter any value to access protected endpoints including the GDS endpoint.

---

#### Running with Docker

```bash
AWS_ENV=NONPROD docker-compose up --build
```

The compose file mounts `~/.aws` into the container automatically, so your local credentials are passed through. The service will be available on port 5000.

---

### Current Status

- Local development environment: **Working** — service starts, downloads models from S3, and serves predictions
- Docker build: **Working** — container builds and starts successfully
- CI/CD test pipeline: **In progress** — ANM tests passing; Dedup test mock under active resolution
- Production deployment: Pending CI/CD green build
