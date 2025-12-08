# --- connectivity test using AzureChatOpenAI directly ---

try:
    from langchain_openai import AzureChatOpenAI
except Exception:
    from langchain.chat_models import AzureChatOpenAI

def test_azure_llm():
    # use the SAME values you use in your real code
    llm = AzureChatOpenAI(
        azure_endpoint   = "https://joshu-meub0vlp-swedencentral.cognitiveservices.azure.com/",
        api_key          = "7Tt8CuQXFHSbKgTHdVL7THXsXqn90Dt4SbTIe4EGAwXLfsVnFqU1JQQ9J9BHAcfHmk5X3W3AAAAACOGAOeD",
        api_version      = "2024-12-01-preview",
        azure_deployment = "gpt-5-mini_ng",
        temperature      = 0.0,
        timeout          = 30,
    )

    try:
        print("Testing AzureChatOpenAI call...")
        resp = llm.invoke("Return exactly the word PING.")
        print("SUCCESS Azure responded:")
        print(resp.content)
    except Exception as e:
        print("ERROR could NOT complete Azure call.")
        print(repr(e))

# run the test
test_azure_llm()
