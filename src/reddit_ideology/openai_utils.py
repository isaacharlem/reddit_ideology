import os
from openai import OpenAI

def init_openai(api_key: str = None):
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key not set")
    client = OpenAI(api_key=key)
    return client

def generate_topic_label(client: OpenAI, top_terms: list[str], model: str = "gpt-3.5-turbo") -> str:
    """
    Ask OpenAI to produce a concise label for a topic given its top terms.
    """
    prompt = (
        "I have a topic characterized by these top terms:\n\n"
        f"{', '.join(top_terms)}\n\n"
        "Please give me a one-to-three word concise label that best summarizes this topic:"
    )
    res = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=0.2,
        max_tokens=10,
    )
    return res.choices[0].message.content.strip()