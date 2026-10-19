"""Proves the Foundry connection works. Run this before touching the app.

    python smoke_test.py

Needs three settings, from a .env file or the environment:

    CLAUDE_BASE_URL         https://<resource>.services.ai.azure.com/anthropic
    CLAUDE_DEPLOYMENT_NAME  the deployment name, not the model id
    AZURE_CLIENT_ID         3a4f14ef-9084-47fb-8ff5-36017fa21f63

Locally it authenticates with your az login session. On the App Service the same
code picks up the managed identity. No keys either way.
"""

import os
import sys
import time

from anthropic import Anthropic
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

SCOPE = "https://ai.azure.com/.default"


class AnthropicIdentity(Anthropic):
    """Anthropic client that fetches a fresh Entra token on every request.

    A one-shot token expires in about an hour, which is fine for a script and
    not fine for a long running App Service. The auth_token property is
    overridden so the SDK asks the provider each time it builds a header.
    """

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


def main():
    load_env()

    base_url = os.environ.get("CLAUDE_BASE_URL", "").strip()
    deployment = os.environ.get("CLAUDE_DEPLOYMENT_NAME", "").strip()
    client_id = os.environ.get("AZURE_CLIENT_ID", "").strip()

    print("CLAUDE_BASE_URL        :", base_url or "NOT SET")
    print("CLAUDE_DEPLOYMENT_NAME :", deployment or "NOT SET")
    print("AZURE_CLIENT_ID        :", client_id or "not set, using az login identity")
    print()

    missing = [n for n, v in
               [("CLAUDE_BASE_URL", base_url), ("CLAUDE_DEPLOYMENT_NAME", deployment)] if not v]
    if missing:
        print("Missing:", ", ".join(missing))
        print("Ask for the resource name and the deployment name.")
        return 1

    if not base_url.rstrip("/").endswith("/anthropic"):
        print("CLAUDE_BASE_URL must end in /anthropic")
        print("Expected: https://<resource>.services.ai.azure.com/anthropic")
        return 1

    print("Acquiring token...")
    started = time.perf_counter()
    try:
        credential = DefaultAzureCredential(managed_identity_client_id=client_id or None)
        token_provider = get_bearer_token_provider(credential, SCOPE)
        token_provider()
    except Exception as exc:
        print("FAILED to get a token:", type(exc).__name__)
        print(" ", str(exc)[:400])
        print()
        print("Locally: run  az login  then try again.")
        print("On the App Service: confirm the managed identity is attached and AZURE_CLIENT_ID is set.")
        return 1
    print(f"token acquired in {int((time.perf_counter() - started) * 1000)} ms")
    print()

    client = AnthropicIdentity(azure_ad_token_provider=token_provider, base_url=base_url)

    print("Calling the model...")
    started = time.perf_counter()
    try:
        message = client.messages.create(
            model=deployment,
            max_tokens=64,
            messages=[{"role": "user",
                       "content": "Reply with exactly: foundry connection ok"}],
        )
    except Exception as exc:
        print("FAILED on the model call:", type(exc).__name__)
        print(" ", str(exc)[:600])
        print()
        print("404 or not found    -> CLAUDE_DEPLOYMENT_NAME is wrong. It is the deployment")
        print("                       name from the Foundry portal, not the model id.")
        print("401 or 403          -> the identity has no role on the Foundry resource.")
        print("Connection error    -> CLAUDE_BASE_URL is wrong, or egress is blocked.")
        return 1

    elapsed = int((time.perf_counter() - started) * 1000)
    text = "".join(block.text for block in message.content if getattr(block, "type", "") == "text")

    print(f"responded in {elapsed} ms")
    print()
    print("model reply   :", text.strip())
    print("model id      :", message.model)
    print("stop reason   :", message.stop_reason)
    print("input tokens  :", message.usage.input_tokens)
    print("output tokens :", message.usage.output_tokens)
    print()
    print("Connection works. No keys used.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
