import os
import dotenv
import openai
import streamlit as st

dotenv.load_dotenv()

deployment = "gpt-4"
avatar = "https://super.abril.com.br/wp-content/uploads/2021/01/musical-ratatouille_site.jpg?quality=90&strip=info&w=720&h=440&crop=1"

client = openai.AzureOpenAI(
    azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key = os.environ["OPENAI_API_KEY"],
    api_version = "2024-02-01"
)

system_message = {
    "role": "system",
    "content": "você é um chef mineiro sem educação chamado Remy que tem 5 estrelas michelin. você fala gírias de Minas em toda mensagem."
}

st.title("Remy AI")

if "messages" not in st.session_state:
    st.session_state["messages"] = [system_message, {"role": "assistant", "content": "Vamo cozinhar o que hoje, uai?"}]
    
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    st.chat_message(msg["role"], avatar=avatar if msg["role"] == "assistant" else None).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = client.chat.completions.create(
        model=deployment, 
        messages=st.session_state.messages,
        temperature=0
        )
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant", avatar=avatar).write(msg)