# ===============================
# 📦 IMPORT LIBRARY
# ===============================
import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DataFrameLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


# ===============================
# ⚙️ KONFIGURASI HALAMAN
# ===============================
st.set_page_config(page_title="Big Data Proyek Wisata", layout="wide")
st.title("🌴 Proyek Analisis Wisata Pantai dengan LLM")
st.sidebar.title("Navigasi")
menu = st.sidebar.selectbox("Pilih Menu", ["Home", "Lihat Data Statistik", "Chatbot LLM Ekstraksi Sentimen"])


# ===============================
# 📂 LOAD DAN CACHE DATA
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv(r"df_comment1.csv")

@st.cache_resource
def init_chain_with_memory(df):
    
    loader = DataFrameLoader(df, page_content_column="full_text")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=250)
    chunks = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key="AIzaSyCjgx8T7CAPmQeL2tobFIY7VHJcE0BOZnM"
    )

    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 120})

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        google_api_key="AIzaSyCjgx8T7CAPmQeL2tobFIY7VHJcE0BOZnM"
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False
    )



# ===============================
# 🧠 INISIALISASI DATA & CHAIN
# ===============================
df_comment = load_data()

if "conv_chain" not in st.session_state:
    st.session_state.conv_chain = init_chain_with_memory(df_comment)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Saya siap membantu menganalisis komentar pengunjung pantai. Silakan ajukan pertanyaan."}
    ]


# ===============================
# 🏠 HALAMAN HOME
# ===============================
if menu == "Home":
    st.header("📌 Tentang Proyek")
    
    st.subheader("👨‍💻 Anggota Kelompok")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image("Amando.jpg", width=200)
        st.markdown("**Nama:** Amando Yuviano\n\n**NPM:** 2215061098")
    with col2:
        st.image("Malvin.jpg", width=200)
        st.markdown("**Nama:** Mohammad Malvin Rafi\n\n**NPM:** 2215061074")
    with col3:
        st.image("Rama.jpg", width=200)
        st.markdown("**Nama:** Naswan Fachri Ramadhan Zain\n\n**NPM:** 2255061011")
    with col4:
        st.image("Ghebi.jpg", width=200)
        st.markdown("**Nama:** Ghebi Armando\n\n**NPM:** 2215061094")

    st.subheader("📄 Ringkasan Proyek")
    st.markdown("""
    Proyek ini bertujuan untuk menganalisis komentar pengunjung terhadap berbagai pantai di wilayah Lampung.
    Kami menggunakan teknologi **Big Data dan LLM (Large Language Model)** untuk:
    
    - Menyediakan visualisasi statistik popularitas dan rating pantai.
    - Mengekstrak topik sentimen dari komentar secara otomatis.
    - Menyediakan chatbot yang bisa menjawab pertanyaan seputar tempat wisata berdasarkan data yang dikumpulkan.

    Anda dapat menjelajahi dua menu utama lainnya:
    - **Data Statistik** – melihat data visualisasi singkat pantai-pantai terpopular.
    - **Chatbot LLM Ekstraksi Sentimen** – berinteraksi dengan chatbot berbasis data komentar.
    """)


# ===============================
# 📊 HALAMAN DATA STATISTIK
# ===============================
elif menu == "Lihat Data Statistik":
    st.header("📊 Data Statistik Tempat Wisata")
    image_list = ["chart1.jpg", "chart2.jpg", "chart3.jpg","chart4.jpg", "chart5.jpg", "chart6.jpg",  "chart7.jpg"]  

    for img in image_list:
        st.image(img, use_column_width=True, caption=f"Gambar: {img}")


# ===============================
# 💬 HALAMAN CHATBOT
# ===============================
elif menu == "Chatbot LLM Ekstraksi Sentimen":
    st.header("💬 Chatbot Ekstraksi Sentimen Tempat Wisata")
    st.caption("Tanyakan apa pun tentang pantai yang ada di dataset kami, seperti: `Apa topik sentimen poisitif tentang Pantai Mutun?`")

    # 🔁 Tombol reset untuk menghapus chat dan memori LLM
    if st.button("🔄 Reset Percakapan"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Halo! Saya siap membantu menganalisis komentar pengunjung pantai. Silakan ajukan pertanyaan."}
        ]
        if "conv_chain" in st.session_state:
            del st.session_state["conv_chain"] 
        st.rerun()

    # 🔁 Tampilkan riwayat obrolan
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ➕ Input pertanyaan user
    if query := st.chat_input("Tanyakan sesuatu..."):
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            with st.spinner("Sedang memproses jawaban..."):
                try:
                    # Buat ulang chain jika ter-reset
                    if "conv_chain" not in st.session_state:
                        st.session_state.conv_chain = init_chain_with_memory(df_comment)

                    response = st.session_state.conv_chain.run(query)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"❌ Gagal menjawab: {e}")


