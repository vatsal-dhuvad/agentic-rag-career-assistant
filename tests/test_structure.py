from pathlib import Path


def test_default_resume_exists():
    assert Path("Vatsal_Dhuvad_Resume.pdf").exists()


def test_main_app_exists():
    assert Path("app.py").exists()
