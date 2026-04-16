"""PyInstaller runtime hook: point SSL at the bundled certifi CA bundle."""
import os
import sys


def _certifi_ca_bundle():
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
        pem = os.path.join(base, "certifi", "cacert.pem")
        if os.path.isfile(pem):
            return pem
    return None


_pem = _certifi_ca_bundle()
if _pem:
    os.environ.setdefault("SSL_CERT_FILE", _pem)
