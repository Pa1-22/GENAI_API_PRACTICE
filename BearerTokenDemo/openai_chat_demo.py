import os
from openai import OpenAI

client = OpenAI()  # automatically reads OPENAI_API_KEY

def ask_chatbot(user_message: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful API tutor."},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    print(ask_chatbot("Explain REST API in 3 bullet points."))
