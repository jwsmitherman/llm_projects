# ----- Azure OpenAI (hard-coded) -----
# !! Replace these with your real values !!
AZURE_OPENAI = {
    "API_KEY": "REPLACE_WITH_YOUR_KEY",
    "ENDPOINT": "https://YOUR-RESOURCE-NAME.openai.azure.com/",
    "API_VERSION": "2024-02-15-preview",
    "CHAT_DEPLOYMENT": "gpt-4o-mini",   # your chat deployment name
}

# Toggle LLM usage (True = use AzureChatOpenAI; False = rules+fuzzy only)
LLM_ENABLED = True

# Optional: cap preview row count in the UI table (for speed)
PREVIEW_ROWS = 1000
