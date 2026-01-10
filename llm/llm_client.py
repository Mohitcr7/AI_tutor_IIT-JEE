from transformers import pipeline

# --------------------------------------------------
# Load HuggingFace model ONCE (important)
# --------------------------------------------------

generator = pipeline(
    task="text2text-generation",
    model="google/flan-t5-large",
    max_new_tokens=800,
    temperature=0.3,
    truncation=True
)

def generate_response(prompt: str) -> str:
    """
    Generate tutor response using HuggingFace model
    """
    output = generator(prompt)
    return output[0]["generated_text"].strip()