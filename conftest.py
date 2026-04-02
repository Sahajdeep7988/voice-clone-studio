from pathlib import Path


def pytest_ignore_collect(collection_path, config):
    p = Path(str(collection_path))
    if p.name == "test_integration.py" and "supabase" in p.parts:
        return True
    return False
