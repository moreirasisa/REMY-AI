import os
import dotenv
import openai
import streamlit as st
from script import load_env, set_azure_client, extract_content_from_file, split_pdf_content, make_embeddings, search

dotenv.load_dotenv()

avatar = "https://super.abril.com.br/wp-content/uploads/2021/01/musical-ratatouille_site.jpg?quality=90&strip=info&w=720&h=440&crop=1"

endpoint, api_key, deployment = load_env()

client = set_azure_client(endpoint, api_key)

pdf_path = [
    "data/bolinho-de-chuva.pdf",
    "data/COXINHA.pdf",
    "data/fricasse-de-frango.pdf",
    "data/pacoca-recheada-com-garapa.pdf"
]

all_text_segments = []

for path in pdf_path:
    with open(path, "rb") as file:
        content = extract_content_from_file(file)
        text_segments = split_pdf_content(content)
        all_text_segments.extend(text_segments)
        
embeddings = make_embeddings(all_text_segments, client)

st.title("Remy AI")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "O que vamos cozinhar hoje?"}]
    
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    st.chat_message(msg["role"], avatar=avatar if msg["role"] == "assistant" else None).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    if prompt:
        try:
            best_match_index, best_match_segment = search(prompt, embeddings, all_text_segments, client)
            messages = [
                {"role": "system", "content": "você é um chef paulista chamado Remy."},
                {"role": "system", "content": "você fala gírias de São Paulo em toda mensagem"},
                {"role": "system", "content": "você não gosta quando falam sobre outro assunto, fora do mundo da culinária."},
                {"role": "system", "content": "você tem uma personalidade forte e direta e não hesita em apontar falhas"},
                {"role": "system", "content": "você tem trejeitos do chef Henrique Fogaça"},
                {"role": "system", "content": "você ama farofa e normalmente grita muito, principalmente 'VAMOOOO'"},
                {"role": "user", "content": f"Suas receitas: {best_match_segment}, {prompt}"}
            ]
            response = client.chat.completions.create(
                model=deployment, 
                messages=messages,
                temperature=0
                )
            msg = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant", avatar=avatar).write(msg)
        except Exception as e:
            st.error(f"OpenAI error: {e}")
    else:
        st.write("Please enter a message")