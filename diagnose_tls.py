"""Finds out what is actually breaking the connection to Foundry.

    python diagnose_tls.py

Works down the stack one layer at a time: DNS, TCP, the TLS handshake, then real
HTTPS requests. The Anthropic SDK reports every one of these as "Connection
error", which is why this exists. Nothing here needs credentials.
"""

import os
import socket
import ssl
import sys
import traceback
from urllib.parse import urlparse

HOST_DEFAULT = "aif-app-tmp-cus-dev.services.ai.azure.com"


def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(".env.local", override=False)
        load_dotenv(".env", override=False)
    except ImportError:
        pass


def chain(exc):
    """Every exception in the cause chain, outermost first."""
    out, seen = [], set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        out.append(f"{type(exc).__module__}.{type(exc).__name__}: {exc}")
        exc = exc.__cause__ or exc.__context__
    return out


def show(exc, indent="    "):
    for i, line in enumerate(chain(exc)):
        print(f"{indent}{'caused by ' if i else ''}{line[:300]}")


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main():
    load_env()
    base_url = os.environ.get("CLAUDE_BASE_URL", "").strip()
    host = urlparse(base_url).hostname if base_url else HOST_DEFAULT
    path = urlparse(base_url).path or "/anthropic"
    print("host :", host)
    print("path :", path)
    print("python:", sys.version.split()[0])

    rule("1. DNS")
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
        addrs = sorted({i[4][0] for i in infos})
        for a in addrs:
            private = a.startswith(("10.", "192.168.")) or a.startswith("172.")
            print(f"    {a}  {'PRIVATE, needs VNet access' if private else 'public'}")
    except Exception as exc:
        print("    FAILED")
        show(exc)
        return 1

    rule("2. TCP 443")
    try:
        sock = socket.create_connection((host, 443), timeout=15)
        print("    connected from", sock.getsockname()[0])
        sock.close()
    except Exception as exc:
        print("    FAILED")
        show(exc)
        return 1

    rule("3. TLS handshake, verification OFF")
    print("    Shows the certificate actually being served, trusted or not.")
    try:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with socket.create_connection((host, 443), timeout=15) as raw:
            with ctx.wrap_socket(raw, server_hostname=host) as tls:
                print("    protocol :", tls.version())
                print("    cipher   :", tls.cipher()[0])
                der = tls.getpeercert(binary_form=True)
                print("    cert len :", len(der) if der else "none")
                try:
                    from cryptography import x509
                    cert = x509.load_der_x509_certificate(der)
                    print("    subject  :", cert.subject.rfc4514_string()[:120])
                    print("    ISSUER   :", cert.issuer.rfc4514_string()[:120])
                    print()
                    issuer = cert.issuer.rfc4514_string().lower()
                    if any(v in issuer for v in ("microsoft", "digicert", "entrust", "globalsign")):
                        print("    Issued by a public CA. No TLS inspection on this host.")
                    else:
                        print("    NOT a public CA. Something is re-signing this traffic.")
                except ImportError:
                    print("    (pip install cryptography to decode the issuer)")
    except Exception as exc:
        print("    FAILED even with verification off, so this is not about trust.")
        show(exc)

    rule("4. TLS handshake, verification ON")
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, 443), timeout=15) as raw:
            with ctx.wrap_socket(raw, server_hostname=host) as tls:
                print("    OK, protocol", tls.version())
    except Exception as exc:
        print("    FAILED")
        show(exc)

    rule("5. TLS handshake using the OS certificate store")
    try:
        import truststore
        ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        with socket.create_connection((host, 443), timeout=15) as raw:
            with ctx.wrap_socket(raw, server_hostname=host) as tls:
                print("    OK, protocol", tls.version())
    except ImportError:
        print("    truststore not installed")
    except Exception as exc:
        print("    FAILED")
        show(exc)

    rule("6. HTTPS GET on the root path")
    print("    Any status code is a success. Only an exception is a failure.")
    try:
        import httpx
        r = httpx.get(f"https://{host}/", timeout=20)
        print(f"    HTTP {r.status_code}")
    except Exception as exc:
        print("    FAILED")
        show(exc)

    rule("7. HTTPS POST on the real messages endpoint, no auth")
    print("    Expect 401. That proves the route exists and the network is fine.")
    try:
        import httpx
        url = f"https://{host}{path.rstrip('/')}/v1/messages"
        print("   ", url)
        r = httpx.post(url, timeout=20,
                       headers={"content-type": "application/json",
                                "anthropic-version": "2023-06-01"},
                       json={"model": os.environ.get("CLAUDE_DEPLOYMENT_NAME", "x"),
                             "max_tokens": 8,
                             "messages": [{"role": "user", "content": "hi"}]})
        print(f"    HTTP {r.status_code}")
        print("    body:", r.text[:400])
    except Exception as exc:
        print("    FAILED")
        show(exc)

    rule("8. Same POST, verification disabled")
    print("    If only this one works, it is a certificate problem after all.")
    try:
        import httpx
        url = f"https://{host}{path.rstrip('/')}/v1/messages"
        r = httpx.post(url, timeout=20, verify=False,
                       headers={"content-type": "application/json",
                                "anthropic-version": "2023-06-01"},
                       json={"model": os.environ.get("CLAUDE_DEPLOYMENT_NAME", "x"),
                             "max_tokens": 8,
                             "messages": [{"role": "user", "content": "hi"}]})
        print(f"    HTTP {r.status_code}")
        print("    body:", r.text[:400])
    except Exception as exc:
        print("    FAILED")
        show(exc)

    rule("READ IT LIKE THIS")
    print("""
    3 fails                     a middlebox is resetting the handshake. Network,
                                not certificates. Goes back to infrastructure.
    3 works, issuer not public  TLS inspection. Section 5 should then work, and
                                if it does the fix is already in llm.py.
    4 and 5 work, 7 fails       the network is fine and the problem is the route
                                or the deployment, not the connection.
    7 returns 401               everything works. The failure is authentication.
    only 8 works                certificate trust. Send me the issuer from 3.
    """)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
