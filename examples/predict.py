"""Example script: call the /predict endpoint from Python.

Usage:
    python examples/predict.py --text "This movie was great"
    python examples/predict.py --url http://my-deployed-url.com --text "..."

Defaults to http://localhost:8000.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def predict(base_url: str, text: str) -> dict:
    """POST to /predict and return the parsed response body.

    Args:
        base_url: Root URL of the ml-serve instance (no trailing slash).
        text: Input text to classify.

    Returns:
        Response body as a dict with keys ``label``, ``score``, ``model``.

    Raises:
        SystemExit: If the server returns a non-2xx status or is unreachable.
    """
    payload = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        print(f"HTTP {exc.code}: {body}", file=sys.stderr)
        sys.exit(1)
    except OSError as exc:
        print(f"Connection failed: {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Call the ml-serve /predict endpoint.")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API.")
    parser.add_argument("--text", required=True, help="Text to classify.")
    args = parser.parse_args()

    result = predict(args.url, args.text)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
