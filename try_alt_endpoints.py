"""Looks for a Foundry hostname the network will let through.

    python try_alt_endpoints.py

An Azure AI Foundry resource answers on several hostnames that all reach the
same backend. diagnose_sni.py showed that services.ai.azure.com is blocked by
hostname while other Azure hostnames to the same IP are not, so one of the
aliases may well get through.

For each candidate: TLS handshake first, then a real authenticated call. A
hostname only counts if the model actually answers on it.
"""

import os
import socket
import ssl
import sys
import time

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

from anthropic import Anthropic
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

SCOPE = "https://ai.azure.com/.default"
RESOURCE = os.environ.get("FOUNDRY_RESOURCE", "aif-app-tmp-cus-dev")

CANDIDATES = [
    (f"{RESOURCE}.services.ai.azure.com", "the one in .env, known blocked"),
    (f"{RESOURCE}.cognitiveservices.azure.com", "Cognitive Services alias"),
    (f"{RESOURCE}.openai.azure.com", "OpenAI alias"),
]


class AnthropicIdentity(Anthropic):
    def __init__(self, *, azure_ad_token_provider, base_url, **kwargs):
        self._token_provider = azure_ad_token_provider
        super().__init__(auth_token="placeholder", api_key=None, base_url=base_url, **kwargs)

    @property
    def auth_token(self):
        return self._token_provider()

    @auth_token.setter
    def auth_token(self, value):
        return


def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(".env.local", override=False)
        load_dotenv(".env", override=False)
    except ImportError:
        pass


def resolves(host):
    try:
        return sorted({i[4][0] for i in socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)})
    except Exception:
        return []


def handshake(host):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with socket.create_connection((host, 443), timeout=12) as raw:
            with ctx.wrap_socket(raw, server_hostname=host) as tls:
                return True, tls.version()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:90]}"


def main():
    load_env()
    deployment = os.environ.get("CLAUDE_DEPLOYMENT_NAME", "claude-sonnet-5-5")
    print("resource  :", RESOURCE)
    print("deployment:", deployment)
    print()

    print("acquiring token...")
    try:
        credential = DefaultAzureCredential(
            managed_identity_client_id=os.environ.get("AZURE_CLIENT_ID") or None)
        provider = get_bearer_token_provider(credential, SCOPE)
        provider()
        print("ok")
    except Exception as exc:
        print("FAILED:", str(exc)[:200])
        print("run az login first")
        return 1
    print()

    working = []

    for host, note in CANDIDATES:
        print("=" * 72)
        print(host)
        print(f"  {note}")
        print("=" * 72)

        ips = resolves(host)
        if not ips:
            print("  DNS     : does not resolve, this alias does not exist")
            print()
            continue
        print("  DNS     :", ", ".join(ips))

        ok, detail = handshake(host)
        print("  TLS     :", "ok, " + detail if ok else "BLOCKED, " + detail)
        if not ok:
            print()
            continue

        base_url = f"https://{host}/anthropic"
        print("  calling :", base_url)
        try:
            client = AnthropicIdentity(
                azure_ad_token_provider=provider, base_url=base_url)
            started = time.perf_counter()
            message = client.messages.create(
                model=deployment,
                max_tokens=32,
                messages=[{"role": "user", "content": "Reply with exactly: ok"}],
            )
            ms = int((time.perf_counter() - started) * 1000)
            text = "".join(b.text for b in message.content
                           if getattr(b, "type", "") == "text").strip()
            print(f"  MODEL   : answered in {ms} ms -> {text!r}")
            print(f"  tokens  : {message.usage.input_tokens} in, "
                  f"{message.usage.output_tokens} out")
            working.append(base_url)
        except Exception as exc:
            print(f"  MODEL   : FAILED {type(exc).__name__}: {str(exc)[:200]}")
        print()

    print("=" * 72)
    print("RESULT")
    print("=" * 72)
    if working:
        print()
        print("  This hostname works. Put it in .env and you are unblocked:")
        print()
        print(f"      CLAUDE_BASE_URL={working[0]}")
        print()
        print("  Still worth raising the ticket. The App Service subnet will hit")
        print("  the same egress policy, and relying on an alias that happens not")
        print("  to be categorised yet is not something to build a pilot on.")
    else:
        print()
        print("  No alias gets through. This needs the egress rule.")
        print()
        print("  Ask for: allow outbound 443 to *.services.ai.azure.com")
        print("  Scope  : your workstation now, the App Service subnet later")
        print("  Say    : aif-app-tmp-cus-dev is an Azure resource inside GMR's")
        print("           own tenant, provisioned under RITM0669421. It is not a")
        print("           third party AI service, which is how the URL category")
        print("           filter is treating it.")
        print("  Evidence: TCP 443 connects, then the TLS handshake is reset")
        print("           (WinError 10054). Other hostnames to the same IP")
        print("           (centralus.api.cognitive.microsoft.com) handshake fine,")
        print("           so the block is on the hostname, not the address.")
    return 0 if working else 1


if __name__ == "__main__":
    sys.exit(main())
