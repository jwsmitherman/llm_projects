# Wiring the model into app.py

Three edits. The fixtures stay in place as a fallback and as regression tests.

## 1. Import and a toggle, near the top of `app.py`

```python
USE_MODEL = os.environ.get("USE_MODEL", "false").lower() in ("1", "true", "yes")

_llm = None
if USE_MODEL:
    import llm as _llm
```

## 2. In `evaluate`, before the fixture lookup

Replace the body of `evaluate` between `fixture, match_method = resolve_fixture(trip)`
and the `envelope(...)` call with:

```python
    supplied_question = trip.get("clarifyingQuestion")
    supplied_answer = trip.get("clarifyingAnswer")
    call_type = "reevaluation" if (supplied_question and supplied_answer) else "initial"

    if USE_MODEL:
        clarification = None
        if call_type == "reevaluation":
            clarification = {"question": supplied_question, "answer": supplied_answer}
        try:
            block = _llm.evaluate_trip(trip, clarification)
            fixture = {
                "fixture_name": "model",
                "payload_file": None,
                "initial_file": None,
                "reevaluation_file": None,
                "initial": block,
                "reevaluation": block,
            }
            match_method = "model"
        except Exception as exc:
            print(f"model call failed, falling back to fixture: {exc}")
            fixture, match_method = resolve_fixture(trip)
    else:
        fixture, match_method = resolve_fixture(trip)

    body = envelope(trip, fixture, call_type, match_method, received_at, started, delay)
    status = 200 if match_method != "none" else 422
    return JSONResponse(status_code=status, content=body)
```

Do the same in `reevaluate`, passing the clarification from the request body.

## 3. Turn off the artificial delay when the model is live

```python
DEFAULT_DELAY_MS = 0 if USE_MODEL else int(os.environ.get("SIMULATED_LATENCY_MS", "350"))
```

The delay existed to make a fixture response feel like a real call. A real call
has its own latency and should not have padding added to it.

## Running it

```powershell
$env:USE_MODEL = "true"
$env:CLAUDE_BASE_URL = "https://<resource>.services.ai.azure.com/anthropic"
$env:CLAUDE_DEPLOYMENT_NAME = "<deployment-name>"
$env:AZURE_CLIENT_ID = "3a4f14ef-9084-47fb-8ff5-36017fa21f63"
python -m uvicorn app:app --reload --port 8000
```

Leave `USE_MODEL` unset and the service behaves exactly as it does today, on
fixtures. That is the switch to flip back if a demo goes sideways.

## Checking the result

`meta.fixtureMatchedOn` reads `model` when the model answered and `olosRequestId`
when a fixture did. The notebook prints that field already, so you can see at a
glance which path produced a response.

`detail.usage` carries input and output tokens per call, which is what you need
for the cost projection.
