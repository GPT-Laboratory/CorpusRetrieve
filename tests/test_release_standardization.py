"""Release-level checks that are independent of any consumer corpus."""
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from corpusretrieve.config import VERSION, Settings
from docparse import __version__ as DOCPARSE_VERSION

ROOT = Path(__file__).resolve().parents[1]


def test_application_modules_match_release_version():
    assert VERSION == "2.0.0"
    assert DOCPARSE_VERSION == VERSION
    assert (ROOT / "VERSION").read_text().strip() == VERSION
    assert 'version = "2.0.0"' in (ROOT / "pyproject.toml").read_text()
    assert 'VERSION: 2.0.0' in (ROOT / "docker-compose.yml").read_text()
    assert 'ARG VERSION=2.0.0' in (ROOT / "Dockerfile").read_text()


def test_production_requires_protected_persistent_runtime():
    with pytest.raises(ValueError, match="STORAGE_BACKEND=s3"):
        Settings(deployment_mode="production").validate()
    with pytest.raises(ValueError, match="API_KEYS"):
        Settings(deployment_mode="production", storage_backend="s3", job_mode="rq").validate()
    with pytest.raises(ValueError, match="S3_SECRET_KEY"):
        Settings(deployment_mode="production", storage_backend="s3", job_mode="rq",
                 api_keys=("test-credential",)).validate()


def test_production_accepts_valid_configuration():
    Settings(deployment_mode="production", storage_backend="s3", job_mode="rq",
             api_keys=("test-credential",), s3_secret_key="not-the-default-secret",
             credential_encryption_key=Fernet.generate_key().decode(),
             cors_origins=()).validate()


def test_distribution_contains_no_customer_specific_runtime_defaults():
    paths = (list((ROOT / "corpusretrieve").rglob("*.py"))
             + list((ROOT / "docparse").rglob("*.py"))
             + list((ROOT / "service").rglob("*.py"))
             + list((ROOT / "tools").glob("*.py"))
             + [ROOT / "corpusretrieve/web/static/app.jsx"])
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        for token in ("bioai", "sourcecollector", "satakunta", "luonnonvarakeskus"):
            assert token not in text, (path, token)
