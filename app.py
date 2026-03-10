import streamlit as st
from OpenAI import OpenAI
import os
import glob
import streamlit.components.v1 as components
import io
import zipfile

# CONFIGURACIÓN DE PÁGINA - DEBE SER LO PRIMERO
st.set_page_config(
    page_title="IA Prometeo",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'Get Help': None, 'Report a bug': None, 'About': "IA Prometeo - Asistente Inteligente"}
)

# IMPORTACIONES PARA LANGCHAIN
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    st.error("Faltan librerías. Instala con: pip install langchain-community langchain-text-splitters faiss-cpu pypdf sentence-transformers")
    st.stop()

try:
    from streamlit_mic_recorder import mic_recorder
    MIC_RECORDER_AVAILABLE = True
except ImportError:
    MIC_RECORDER_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_FOLDER = os.path.join(BASE_DIR, "documentos")

@st.cache_resource
def load_knowledge_base():
    if not os.path.exists(DOCS_FOLDER):
        try:
            os.makedirs(DOCS_FOLDER)
        except OSError as e:
            st.error(f"Error al crear la carpeta 'documentos': {e}")
            return None, []

    pdf_files = glob.glob(os.path.join(DOCS_FOLDER, "*.pdf"))
    
    if not pdf_files: 
        return None, []

    all_docs = []
    valid_files = []
    error_files = []

    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            if not docs:
                continue
            
            filename = os.path.basename(pdf_path)
            for doc in docs: 
                doc.metadata["source"] = filename
            
            all_docs.extend(docs)
            valid_files.append(filename)
        except Exception as e:
            error_files.append((os.path.basename(pdf_path), str(e)))

    if error_files:
        st.warning(f"No se pudieron leer {len(error_files)} archivos.")

    if not all_docs: 
        return None, []

    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = text_splitter.split_documents(all_docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore, valid_files
    except Exception as e:
        st.error(f"Error al procesar embeddings: {e}")
        return None, []

# CSS
css_neutral = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stDecoration"] {display: none;}
    [data-testid="stToolbar"] {display: none;}
    :root {
        --primary-color: #4F46E5;
        --secondary-color: #6366F1;
        --bg-color: #F3F4F6;
        --sidebar-bg: #FFFFFF;
        --text-color: #1F2937;
    }
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-color);
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        text-align: center;
        padding: 2rem 1rem 1rem 1rem;
    }
    .main-title {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: clamp(2rem, 6vw, 3rem);
        color: var(--primary-color);
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #6B7280;
        font-size: 1rem;
        font-weight: 400;
    }
    .content-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border: 1px solid #E5E7EB;
    }
    .fixed-chat-wrapper {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    [data-testid="stChatMessage"] {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    [data-testid="stSidebar"] {
        background: var(--sidebar-bg) !important;
        border-right: 1px solid #E5E7EB;
    }
    .stButton button {
        background-color: var(--primary-color) !important;
        color: white !important;
        border-radius: 10px !important;
    }
    .info-card {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        color: #1E40AF;
    }
</style>
"""
st.markdown(css_neutral, unsafe_allow_html=True)

header_html = """
<div class="main-header">
    <h1 class="main-title">IA PROMETEO 🔥</h1>
    <p class="subtitle">Tu asistente inteligente para documentos y planeación</p>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# API KEY
api_key = None
try:
    if "groq" in st.secrets and "api_key" in st.secrets["groq"]:
        api_key = st.secrets["groq"]["api_key"]
except:
    pass

# SIDEBAR
with st.sidebar:
    st.markdown("<h2>⚙️ Panel de Control</h2>", unsafe_allow_html=True)
    st.markdown("#### 🔑 Configuración")
    if not api_key:
        api_key_input = st.text_input("API Key de Groq", type="password", key="api_key_input_groq")
        if api_key_input:
            api_key = api_key_input
        else:
            st.warning("Necesitas API Key de Groq.")
            st.info("Obtén una en: console.groq.com")
    else:
        st.success("API Key configurada ✅")
    
    voice_enabled = st.checkbox("Activar respuestas de voz", value=True)
    st.markdown("---")
    st.markdown("#### 📦 Base de Conocimiento")
    st.caption("Carpeta: `documentos/`")
    
    uploaded_zip = st.file_uploader("Sube un ZIP con PDFs", type="zip", key="zip_uploader")

    if uploaded_zip:
        if "processed_zip_name" not in st.session_state or st.session_state.processed_zip_name != uploaded_zip.name:
            try:
                with zipfile.ZipFile(uploaded_zip, 'r') as z:
                    z.extractall(DOCS_FOLDER)
                st.session_state.processed_zip_name = uploaded_zip.name
                st.toast("✅ Archivos extraídos. Recargando...")
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Error al descomprimir: {e}")

    st.markdown("---")
    st.markdown("#### 📚 Estado")
    
    if st.button("🔄 Recargar Base de Datos", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    if st.session_state.get("loaded_files"):
        st.success(f"🟢 {len(st.session_state.loaded_files)} Documentos Activos")
        with st.expander("Ver lista"):
            for f in st.session_state.loaded_files:
                st.write(f"📄 {f}")
    else:
        st.info("🔴 Repositorio Vacío. Añade PDFs.")

    st.markdown("---")
    st.markdown("#### 💡 Tips")
    st.markdown('<div class="info-card">Puedes preguntarme sobre los documentos cargados o pedir ayuda para planear clases.</div>', unsafe_allow_html=True)
    st.markdown("<br><p style='text-align:center; font-size:0.8rem; color:#9CA3AF;'>Desarrollado por IA Prometeo</p>", unsafe_allow_html=True)

if not api_key:
    st.stop()

try:
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key
    )
except Exception as e:
    st.error(f"Error al conectar con Groq: {e}")
    st.stop()

# PROMPTS
SYSTEM_PROMPT_BASE = """
Eres **IA Prometeo**, un asistente inteligente, conciso y profesional.
Tu objetivo es ayudar al usuario a analizar documentos y realizar tareas de planeación o consulta.
Tono: Profesional, cercano y eficiente.
"""

loaded_files_list_str = "No hay archivos cargados."
if st.session_state.get("loaded_files"):
    loaded_files_list_str = "\n".join([f"{i+1}. {fname}" for i, fname in enumerate(st.session_state.loaded_files)])

SYSTEM_PROMPT_PLANNING = f"""
Eres **IA Prometeo - Experto en Planeación**.

**ARCHIVOS DISPONIBLES:**
{loaded_files_list_str}
-----------------------------------------

**REGLAS:**
1. Usa solo información del contexto.
2. Busca patrones como "Unidad", "Módulo", "Bloque".
3. Si no tienes información, indícalo.

**FLUJO:**
**PASO 1: ACTIVACIÓN**
Si el usuario dice "vamos a planear":
1. Muestra la lista de archivos.
2. Pregunta: "¿Cuál es el **número** del programa a utilizar?"

**PASO 2: LECTURA**
Cuando el usuario elija un número:
1. Identifica el archivo.
2. Lista las unidades encontradas numeradas.
3. Pregunta: "¿Qué **número** de unidad(es) vamos a planear?"

**PASO 3: SESIONES**
"¿Cuántas **sesiones** en total necesita?"

**PASO 4: DIAS**
"¿Qué **días de la semana** se imparten las clases?"

**PASO 5: CRITERIOS**
"¿Cuáles son los **criterios de evaluación**?"

**PASO 6: FECHAS**
"Indica **fecha de inicio** y **fecha de término**."

**PASO 7: BORRADOR**
Genera ejemplos.

**PASO 8: FINAL**
Genera planeación completa.
"""

# INICIALIZACIÓN
if "messages" not in st.session_state: 
    st.session_state.messages = []

if "planning_mode" not in st.session_state:
    st.session_state.planning_mode = False

if "vectorstore" not in st.session_state:
    vectorstore, loaded_files = load_knowledge_base()
    st.session_state.vectorstore = vectorstore
    st.session_state.loaded_files = loaded_files

# FUNCIONES
def get_audio_button_html(text, key):
    text_clean = text.replace("'", "").replace('"', '').replace("\n", " ")
    return f"""
    <div style="margin-top: 10px; text-align: right;">
        <button onclick="
            var u = new SpeechSynthesisUtterance('{text_clean}');
            u.lang = 'es-MX';
            u.rate = 0.95;
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(u);
        " style="
            background-color: #4F46E5;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 500;
            cursor: pointer;
            font-family: 'Inter', sans-serif;
            font-size: 0.85rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        ">🔊 Escuchar</button>
    </div>
    """

def get_context_for_planning(user_input, vectorstore, loaded_files):
    selected_file_index = None
    if loaded_files:
        try:
            potential_index = int(user_input.strip()) - 1
            if 0 <= potential_index < len(loaded_files):
                selected_file_index = potential_index
        except ValueError:
            pass

    if selected_file_index is not None:
        target_filename = loaded_files[selected_file_index]
        structure_queries = ["Unidades de aprendizaje", "Contenido temático desglosado", "Bloques Módulos Temas", "Índice de contenidos estructura"]
        all_docs = []
        seen_content = set()
        
        try:
            for query in structure_queries:
                docs = vectorstore.similarity_search(query=query, k=4, filter={"source": target_filename})
                for doc in docs:
                    if doc.page_content not in seen_content:
                        all_docs.append(doc)
                        seen_content.add(doc.page_content)
            
            final_docs = all_docs[:12]
            if not final_docs:
                return f"No se encontró información estructural en {target_filename}.", target_filename
            
            context_text = "\n\n---\n\n".join([f"Fragmento:\n{doc.page_content}" for doc in final_docs])
            return context_text, target_filename
            
        except Exception as e:
            return f"Error al leer {target_filename}: {e}", target_filename
    else:
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            docs = retriever.invoke(user_input)
            return "\n\n---\n\n".join([f"Fuente: {doc.metadata.get('source', 'Desconocido')}\n{doc.page_content}" for doc in docs]), None
        except Exception as e:
            return "", None

# CHAT INTERFACE
audio_data = None

if MIC_RECORDER_AVAILABLE:
    try:
        audio_data = mic_recorder(
            start_prompt="🎤 Iniciar Grabación",
            stop_prompt="🛑 Detener Grabación",
            just_once=False,
            key="mic_main_btn"
        )
    except Exception as e:
        pass

# PROCESAR AUDIO
if audio_data:
    try:
        audio_bytes = audio_data['bytes']
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = f"audio.{audio_data['format']}"

        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3-turbo",
            language="es"
        )
        if transcription.text:
            st.toast(f"🎤 Escuché: {transcription.text}")
            prompt = transcription.text
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            if "vamos a planear" in prompt.lower():
                st.session_state.planning_mode = True
                st.toast("📑 Modo Planeación Activado")

            current_prompt = SYSTEM_PROMPT_PLANNING if st.session_state.planning_mode else SYSTEM_PROMPT_BASE
            
            context_text = ""
            if st.session_state.get("vectorstore"):
                context_text, _ = get_context_for_planning(prompt, st.session_state.vectorstore, st.session_state.loaded_files)

            full_prompt = current_prompt
            if context_text:
                full_prompt += f"\n\nContexto de documentos:\n{context_text}"

            formatted_messages = [{"role": "system", "content": full_prompt}] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=formatted_messages
            )

            ai_response = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": ai_response})

    except Exception as e:
        st.error(f"Error de audio: {e}")

# INPUT DE CHAT
if prompt := st.chat_input("Escribe tu mensaje..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    if "vamos a planear" in prompt.lower():
        st.session_state.planning_mode = True
        st.toast("📑 Modo Planeación Activado")

    current_prompt = SYSTEM_PROMPT_PLANNING if st.session_state.planning_mode else SYSTEM_PROMPT_BASE

    context_text = ""
    if st.session_state.get("vectorstore"):
        context_text, _ = get_context_for_planning(prompt, st.session_state.vectorstore, st.session_state.loaded_files)

    full_prompt = current_prompt
    if context_text:
        full_prompt += f"\n\nContexto de documentos:\n{context_text}"

    formatted_messages = [{"role": "system", "content": full_prompt}] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=formatted_messages
        )
        ai_response = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
    except Exception as e:
        st.error(f"Error: {e}")

# CONTENEDOR DE CHAT
st.markdown("<div class='fixed-chat-wrapper'>", unsafe_allow_html=True)
chat_container = st.container(height=450, key="chat_container")

with chat_container:
    for i, message in enumerate(st.session_state.messages):
        if message["role"] != "system":
            avatar = "🔥" if message["role"] == "assistant" else "👤"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
                if message["role"] == "assistant" and voice_enabled:
                    components.html(
                        get_audio_button_html(message["content"], f"audio_{i}"),
                        height=50,
                    )

st.markdown("</div>", unsafe_allow_html=True)
