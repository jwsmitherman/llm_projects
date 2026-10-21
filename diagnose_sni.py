"""Works out WHO is resetting the connection: the network or Azure.

    python diagnose_sni.py

The Foundry hostname resolves to a shared Azure gateway that fronts every
Cognitive Services resource in the region. So we can open TLS to that same IP
while announcing different hostnames in the handshake. If a harmless hostname
gets through and the Foundry one is reset, the decision is being made on the
hostname, which only something on our side of the wire can do.

Certificate errors are expected and ignored here. The only question is whether
the handshake completes at all.
"""

import socket
import ssl
import sys

TARGET = "aif-app-tmp-cus-dev.services.ai.azure.com"
CONTROL = "www.microsoft.com"


def handshake(ip, sni, port=443, timeout=12):
    """Open TLS to ip while claiming to be sni. Returns (ok, detail)."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with socket.create_connection((ip, port), timeout=timeout) as raw:
            with ctx.wrap_socket(raw, server_hostname=sni) as tls:
                return True, f"{tls.version()} / {tls.cipher()[0]}"
    except ConnectionResetError as exc:
        return False, f"RESET  {exc}"
    except socket.timeout:
        return False, "TIMEOUT"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:110]}"


def resolve(host):
    try:
        return sorted({i[4][0] for i in socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)})[0]
    except Exception as exc:
        print(f"    cannot resolve {host}: {exc}")
        return None


def main():
    target_ip = resolve(TARGET)
    control_ip = resolve(CONTROL)
    if not target_ip:
        return 1

    print(f"{TARGET}")
    print(f"  resolves to {target_ip}")
    print(f"{CONTROL}")
    print(f"  resolves to {control_ip}")
    print()

    print("=" * 78)
    print("A. Same IP, different hostnames announced in the handshake")
    print("=" * 78)
    print(f"   all three connect to {target_ip}")
    print()

    names = [
        (TARGET, "the Foundry resource"),
        ("centralus.api.cognitive.microsoft.com", "same gateway, older hostname"),
        ("www.microsoft.com", "harmless hostname, wrong cert expected"),
        ("example.com", "nonsense hostname"),
    ]
    results = {}
    for sni, note in names:
        ok, detail = handshake(target_ip, sni)
        results[sni] = ok
        print(f"   {'OK  ' if ok else 'FAIL'}  {sni}")
        print(f"         {note}")
        print(f"         {detail}")
        print()

    print("=" * 78)
    print("B. Control, an unrelated host on its own IP")
    print("=" * 78)
    if control_ip:
        ok, detail = handshake(control_ip, CONTROL)
        print(f"   {'OK  ' if ok else 'FAIL'}  {CONTROL}   {detail}")
    print()

    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    foundry_ok = results.get(TARGET, False)
    others_ok = any(ok for sni, ok in results.items() if sni != TARGET)

    if foundry_ok:
        print("""
    The handshake worked here but not in diagnose_tls.py. Run both again;
    something is intermittent.""")
    elif others_ok:
        print("""
    The Foundry hostname is reset while other hostnames to the SAME IP are
    allowed. Azure cannot do that, because it only sees the hostname after the
    handshake it is supposedly rejecting. Something between this laptop and
    Azure is reading the hostname out of the handshake and killing the
    connection.

    This is a network egress policy. Many enterprises block the whole
    "generative AI" URL category by default, and services.ai.azure.com sits in
    it. az login keeps working because login.microsoftonline.com and
    management.azure.com are categorised separately.

    ASK FOR: an egress allow rule for *.services.ai.azure.com on 443, for your
    workstation and later for the App Service subnet. Name the resource,
    aif-app-tmp-cus-dev, and note that it is an internal Azure resource in GMR's
    own tenant rather than a third party AI service.""")
    else:
        print("""
    Every hostname to that IP is reset, including harmless ones, so nothing is
    being decided on the hostname. Either that IP is blocked outright, or the
    Azure resource is refusing public traffic.

    CHECK THIS NEXT, it is the deciding question:

        az cognitiveservices account show -n aif-app-tmp-cus-dev -g <rg> \\
          --query "{public:properties.publicNetworkAccess, acls:properties.networkAcls}"

    publicNetworkAccess: Disabled means the resource is private endpoint only
    and no workstation can reach it. That is the answer, and it goes back to
    infrastructure as a new ticket.""")

    if control_ip and not results.get("www.microsoft.com", False):
        print("""
    Note: the control failed too, so general HTTPS from this machine is also
    being interfered with. Worth mentioning when you raise it.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
