import os
from dotenv import load_dotenv
import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st
import google.generativeai as genai

# load info from .env file
load_dotenv()

model = genai.GenerativeModel("gemini-1.5-pro")

# configure the interface page name and icon
st.set_page_config(
    page_title="Remy AI",
    page_icon="assets/868ec544-6b16-43b7-9b47-f708d02a4031.png"
)

# processing pdf files
@st.cache_data
def extract_text_from_pdf(pdf_file):
    pdf_content = pdf_file.read()
    document = fitz.open(stream=pdf_content, filetype="pdf")
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

@st.cache_data
def create_embeddings(all_text_segments):
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(all_text_segments)
    return vectorizer, embeddings

def search(prompt, vectorizer, embeddings, all_text_segments):
    query_vectorizer = vectorizer.transform([prompt])
    similarities = cosine_similarity(query_vectorizer, embeddings).flatten()
    best_match_index = np.argmax(similarities)
    best_match_segment = all_text_segments[best_match_index]
    return best_match_index, best_match_segment

@st.cache_data
def load_pdfs():
    pdf_paths = [os.path.join("data", filename)
            for filename in os.listdir("data")
            if filename.lower().endswith(".pdf")]
    all_text_segments = []
    for pdf_path in pdf_paths:
        print(pdf_path)
        with open(pdf_path, "rb") as file:
            text = extract_text_from_pdf(file)
            all_text_segments.append(text)
    return all_text_segments

all_text_segments = load_pdfs()
vectorizer, embeddings = create_embeddings(all_text_segments)

# interface
avatar = "https://i.imgur.com/rORadpr.jpg"

st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://i.imgur.com/rORadpr.jpg" width="150">
        <h1>Remy AI</h1>
        <h5>
            Sou uma Inteligência artificial, focada em te ajudar com suas refeições.<br>
            Bateu aquela fome? É só me perguntar 😉
        </h5>
    </div>
    """, unsafe_allow_html=True
)
        
# chat
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "O que vamos cozinhar hoje?"}]
    
for message in st.session_state["messages"]:
    st.chat_message(message["role"], avatar=avatar if message["role"] == "assistant" else None).write(message["content"])

if prompt := st.chat_input():
    history = []
    for msg in st.session_state.messages:
        role = ""
        if msg["role"] == "assistant":
            role = "model"
        elif msg["role"] == "user":
            role = "user"
        history.append({
            "role": role,
            "parts": [
                msg["content"]
            ],
        })
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    if prompt:
        try:
            with st.spinner("Hmmm..."):
                best_match_index, best_match_segment = search(prompt, vectorizer, embeddings, all_text_segments)
                chat_interaction = model.start_chat(
                    history=[
                        {
                            "role": "model",
                            "parts": [
                                "Você é um chef brasileiro chamado Remy, que fala português brasileiro.",
                                "Você tem uma personalidade forte e não tem problemas em expressar isso.",
                                "Você tem a personalidade do chef Henrique Fogaça.",
                                "Você não responde quando perguntado sobre coisas não relacionadas à culinária.",
                                "Quando não souber a resposta à pergunta do usuário, diga 'Desculpe, não sei responder.'",
                                "Se o usuário pedir a lista de ingredientes de uma receita, formate a lista em forma de tópicos, com cada ingrediente em uma linha.",
                                "Se o usuário perguntar algo que não seja relacionado a receitas, diga que você é uma IA especialista em receitas.",
                                f"Use essas receitas para responder às perguntas do usuário: {all_text_segments}"
                            ]
                        },
                        {
                            "role": "user",
                            "parts": [
                                f"responda essa pergunta: {prompt}, utilizando esses contextos: {best_match_index}"
                            ]
                        }
                    ] + history
                )
                response = chat_interaction.send_message(prompt)
                answer = response.text
                st.session_state["messages"].append({"role": "assistant", "content": answer})
            st.chat_message("assistant", avatar=avatar).write(answer)
        except Exception as e:
            st.error(f"Gemini API Error: {e}")
    else:
        st.write("Por favor, me fale uma receita para que possamos conversar.")