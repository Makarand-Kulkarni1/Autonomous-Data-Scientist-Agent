from src.sandbox import run_in_sandbox


def test_network_is_blocked(tmp_path):
    """Code inside the sandbox must not be able to reach the internet."""
    code = """
import urllib.request
try:
    urllib.request.urlopen("https://example.com", timeout=5)
    print("INTERNET WORKS")
except Exception as e:
    print("Blocked:", type(e).__name__)
"""
    result = run_in_sandbox(code, tmp_path)

    # The dangerous outcome must not happen
    assert "INTERNET WORKS" not in result.stdout
    # And the block must really have happened (guards against a silent crash)
    assert "Blocked" in result.stdout
