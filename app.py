import streamlit as st
import os
import glob
import zipfile
import io

# CONFIGURACIÓN DE PÁGINA - DEBE SER LO PRIMERO
st.set_page_config(
    page_title="IA Prometeo",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "IA Prometeo - Asistente Inteligente"
    }
)

# ============================================================
# IMPORTACIONES CON MANEJO DE ERRORES ROBUSTO
# ============================================================

def check_and_import_dependencies():
    """Verifica e importa todas las dependencias necesarias."""
    missing_deps = []
    
    # Verificar openai
    try:
        from openai import OpenAI
    except ImportError:
        missing_deps.append("openai")
    
    # Verificar langchain
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        missing_deps.append("langchain-community langchain-text-splitters")
    
    # Verificar faiss
    try:
        import faiss
    except ImportError:
        missing_deps.append("faiss-cpu")
    
    # Verificar sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        missing_deps.append("sentence-transformers")
    
    # Verificar pypdf
    try:
        import pypdf
    except ImportError:
        missing_deps.append("pypdf")
    
    return missing_deps

# Verificar dependencias
missing = check_and_import_dependencies()
if missing:
    st.error(f"❌ Faltan dependencias. Instala con: `pip install {' '.join(missing)}`")
    st.code(f"pip install {' '.join(missing)}", language="bash")
    st.stop()

# Importar después de verificar
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Importar mic_recorder opcionalmente
MIC_RECORDER_AVAILABLE = False
try:
    from streamlit_mic_recorder import mic_recorder
    MIC_RECORDER_AVAILABLE = True
except ImportError:
    pass  # No es crítico, la app puede funcionar sin esto

# ============================================================
# CONFIGURACIÓN DE CARPETA
# ============================================================

# En Streamlit Cloud, usar un directorio temporal o relativo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_FOLDER = os.path.join(BASE_DIR, "documentos")

# Crear carpeta si no existe
if not os.path.exists(DOCS_FOLDER):
    try:
        os.makedirs(DOCS_FOLDER)
    except OSError as e:
        st.warning(f"No se pudo crear la carpeta 'documentos': {e}")
        # Usar directorio temporal como alternativa
        import tempfile
        DOCS_FOLDER = tempfile.mkdtemp()

# ============================================================
# FUNCIONES DE CARGA DE DOCUMENTOS
# ============================================================

@st.cache_resource(show_spinner="Cargando base de conocimiento...")
def load_knowledge_base():
    """Carga y procesa los documentos PDF de la carpeta documentos."""
    
    pdf_files = glob.glob(os.path.join(DOCS_FOLDER, "*.pdf"))
    
    if not pdf_files:
        return None, []
    
    all_docs = []
    valid_files = []
    error_files = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, pdf_path in enumerate(pdf_files):
        try:
            status_text.text(f"Procesando: {os.path.basename(pdf_path)}")
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
        
        progress_bar.progress((i + 1) / len(pdf_files))
    
    progress_bar.empty()
    status_text.empty()
    
    if error_files:
        st.warning(f"⚠️ No se pudieron leer {len(error_files)} archivo(s).")
        with st.expander("Ver errores"):
            for fname, err in error_files:
                st.write(f"📄 **{fname}**: {err}")
    
    if not all_docs:
        return None, []
    
    try:
        # Dividir texto
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(all_docs)
        
        # Crear embeddings
        with st.spinner("Generando embeddings..."):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        
        # Crear vector store
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        return vectorstore, valid_files
        
    except Exception as e:
        st.error(f"❌ Error al procesar embeddings: {e}")
        return None, []

# ============================================================
# ESTILOS CSS
# ============================================================

css_styles = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Ocultar elementos de Streamlit */
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

    section[data-testid="stMain"] {
        position: relative;
        z-index: 1 !important;
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

    .chat-wrapper {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    .st-key-chat_container > div > div {
        border: none !important;
        background: transparent !important;
    }

    [data-testid="stChatMessage"] {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    
    [data-testid="stChatMessageContent"] {
        color: var(--text-color) !important;
    }

    [data-testid="stChatInput"] {
        border: 1px solid #D1D5DB !important;
        border-radius: 16px !important;
        background-color: #FFFFFF !important;
    }
    
    [data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        color: var(--text-color) !important;
    }

    [data-testid="stSidebar"] {
        background: var(--sidebar-bg) !important;
        border-right: 1px solid #E5E7EB;
    }
    
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: var(--primary-color) !important;
        font-family: 'Inter', sans-serif !important;
    }

    .stButton button {
        background-color: var(--primary-color) !important;
        color: white !important;
        border-radius: 10px !important;
    }
    
    .stButton button:hover {
        background-color: var(--secondary-color) !important;
    }

    .stAlert {
        background: #FFFFFF !important;
        border-left: 4px solid var(--primary-color) !important;
        border-radius: 8px !important;
    }

    .info-card {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        color: #1E40AF;
    }
    
    .audio-btn {
        background-color: #4F46E5;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 500;
        cursor: pointer;
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
    }
    
    .audio-btn:hover {
        background-color: #6366F1;
    }
</style>
"""

st.markdown(css_styles, unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

header_html = """
<div class="main-header">
    <h1 class="main-title">🔥 IA PROMETEO</h1>
    <p class="subtitle">Tu asistente inteligente para documentos y planeación</p>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# ============================================================
# CONFIGURACIÓN DE API KEY
# ============================================================

api_key = None

# Intentar obtener API key de secrets (Streamlit Cloud)
try:
    if "groq" in st.secrets and "api_key" in st.secrets["groq"]:
        api_key = st.secrets["groq"]["api_key"]
except Exception:
    pass

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("<h2>⚙️ Panel de Control</h2>", unsafe_allow_html=True)
    
    # Configuración de API Key
    st.markdown("#### 🔑 Configuración")
    
    if not api_key:
        api_key_input = st.text_input(
            "API Key de Groq", 
            type="password", 
            key="api_key_input_groq",
            help="Obtén tu API Key en console.groq.com"
        )
        if api_key_input:
            api_key = api_key_input
        else:
            st.warning("⚠️ Necesitas una API Key de Groq")
            st.markdown("[Obtén una aquí](https://console.groq.com)")
    else:
        st.success("✅ API Key configurada")
    
    # Opción de voz
    voice_enabled = st.checkbox("Activar respuestas de voz", value=False)
    st.markdown("---")
    
    # Base de conocimiento
    st.markdown("#### 📦 Base de Conocimiento")
    st.caption(f"Carpeta: `{DOCS_FOLDER}`")
    
    # Subir ZIP
    uploaded_zip = st.file_uploader(
        "Sube un ZIP con PDFs", 
        type="zip", 
        key="zip_uploader"
    )
    
    if uploaded_zip:
        if "processed_zip_name" not in st.session_state or \
           st.session_state.processed_zip_name != uploaded_zip.name:
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
    
    # Estado de documentos
    st.markdown("#### 📚 Estado")
    
    if st.button("🔄 Recargar Base de Datos", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()
    
    # Mostrar archivos cargados
    if st.session_state.get("loaded_files"):
        st.success(f"🟢 {len(st.session_state.loaded_files)} Documento(s) Activo(s)")
        with st.expander("Ver lista"):
            for f in st.session_state.loaded_files:
                st.write(f"📄 {f}")
    else:
        st.info("🔴 Repositorio Vacío. Añade PDFs.")
    
    st.markdown("---")
    
    # Tips
    st.markdown("#### 💡 Tips")
    st.markdown(
        '<div class="info-card">'
        'Puedes preguntarme sobre los documentos cargados o pedir ayuda para planear clases.'
        '</div>',
        unsafe_allow_html=True
    )
    
    st.markdown(
        "<br><p style='text-align:center; font-size:0.8rem; color:#9CA3AF;'>"
        "Desarrollado por IA Prometeo</p>",
        unsafe_allow_html=True
    )

# Detener si no hay API key
if not api_key:
    st.info("👈 Por favor, configura tu API Key de Groq en el panel lateral.")
    st.stop()

# ============================================================
# CONEXIÓN CON GROQ
# ============================================================

try:
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key
    )
except Exception as e:
    st.error(f"❌ Error al conectar con Groq: {e}")
    st.stop()

# ============================================================
# PROMPTS DEL SISTEMA
# ============================================================

SYSTEM_PROMPT_BASE = """
Eres **IA Prometeo**, un asistente inteligente, conciso y profesional.
Tu objetivo es ayudar al usuario a analizar documentos y realizar tareas de planeación o consulta.

**Características:**
- Profesional pero cercano
- Respuestas concisas y útiles
- Orientado a resultados

Si el usuario pregunta sobre documentos, usa el contexto proporcionado.
"""

SYSTEM_PROMPT_PLANNING = """
Eres **IA Prometeo - Experto en Planeación**.

**FLUJO DE PLANEACIÓN:**

**PASO 1: ACTIVACIÓN**
Si el usuario dice "vamos a planear":
1. Muestra la lista de archivos disponibles.
2. Pregunta: "¿Cuál es el **número** del programa a utilizar?"

**PASO 2: LECTURA**
Cuando el usuario elija un número:
1. Identifica el archivo correspondiente.
2. Lista las unidades encontradas (numeradas).
3. Pregunta: "¿Qué **número** de unidad(es) vamos a planear?"

**PASO 3: SESIONES**
Pregunta: "¿Cuántas **sesiones** en total necesitas?"

**PASO 4: DÍAS**
Pregunta: "¿Qué **días de la semana** se imparten las clases?"

**PASO 5: CRITERIOS**
Pregunta: "¿Cuáles son los **criterios de evaluación**?"

**PASO 6: FECHAS**
Pregunta: "Indica **fecha de inicio** y **fecha de término**."

**PASO 7: BORRADOR**
Genera ejemplos de planeación.

**PASO 8: FINAL**
Genera la planeación completa.

**REGLAS:**
1. Usa solo información del contexto.
2. Busca patrones como "Unidad", "Módulo", "Bloque".
3. Si no tienes información, indícalo claramente.
"""

# ============================================================
# INICIALIZACIÓN DE ESTADO DE SESIÓN
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "planning_mode" not in st.session_state:
    st.session_state.planning_mode = False

if "vectorstore" not in st.session_state:
    vectorstore, loaded_files = load_knowledge_base()
    st.session_state.vectorstore = vectorstore
    st.session_state.loaded_files = loaded_files

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def get_audio_button_html(text, key):
    """Genera HTML para botón de audio (text-to-speech)."""
    text_clean = text.replace("'", "").replace('"', '').replace("\n", " ")
    return f"""
    <div style="margin-top: 10px; text-align: right;">
        <button onclick="
            var u = new SpeechSynthesisUtterance('{text_clean}');
            u.lang = 'es-MX';
            u.rate = 0.95;
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(u);
        " class="audio-btn">🔊 Escuchar</button>
    </div>
    """

def get_context_for_query(user_input, vectorstore, loaded_files):
    """Obtiene contexto relevante de los documentos para una consulta."""
    
    if not vectorstore:
        return "", None
    
    # Intentar detectar si es una selección de archivo por número
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
        
        # Consultas para encontrar estructura del documento
        structure_queries = [
            "Unidades de aprendizaje",
            "Contenido temático desglosado",
            "Bloques Módulos Temas",
            "Índice de contenidos estructura"
        ]
        
        all_docs = []
        seen_content = set()
        
        try:
            for query in structure_queries:
                docs = vectorstore.similarity_search(
                    query=query,
                    k=4,
                    filter={"source": target_filename}
                )
                for doc in docs:
                    if doc.page_content not in seen_content:
                        all_docs.append(doc)
                        seen_content.add(doc.page_content)
            
            if all_docs:
                context_text = "\n\n---\n\n".join([
                    f"Fragmento:\n{doc.page_content}" 
                    for doc in all_docs[:12]
                ])
                return context_text, target_filename
                
        except Exception as e:
            return f"Error al leer {target_filename}: {e}", target_filename
        
        return f"No se encontró información estructural en {target_filename}.", target_filename
    
    else:
        # Búsqueda general
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            docs = retriever.invoke(user_input)
            
            context_text = "\n\n---\n\n".join([
                f"Fuente: {doc.metadata.get('source', 'Desconocido')}\n{doc.page_content}"
                for doc in docs
            ])
            return context_text, None
            
        except Exception as e:
            return "", None

def get_ai_response(user_message, is_planning_mode=False):
    """Obtiene respuesta del modelo de IA."""
    
    current_prompt = SYSTEM_PROMPT_PLANNING if is_planning_mode else SYSTEM_PROMPT_BASE
    
    # Añadir lista de archivos si está disponible
    if is_planning_mode and st.session_state.get("loaded_files"):
        files_list = "\n".join([
            f"{i+1}. {fname}" 
            for i, fname in enumerate(st.session_state.loaded_files)
        ])
        current_prompt += f"\n\n**ARCHIVOS DISPONIBLES:**\n{files_list}"
    
    # Obtener contexto de documentos
    context_text = ""
    if st.session_state.get("vectorstore"):
        context_text, _ = get_context_for_query(
            user_message,
            st.session_state.vectorstore,
            st.session_state.loaded_files
        )
    
    # Construir mensaje completo
    full_prompt = current_prompt
    if context_text:
        full_prompt += f"\n\n**Contexto de documentos:**\n{context_text}"
    
    # Construir historial de mensajes
    formatted_messages = [
        {"role": "system", "content": full_prompt}
    ] + [
        {"role": m["role"], "content": m["content"]} 
        for m in st.session_state.messages
    ]
    
    # Llamar a la API
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=formatted_messages,
            temperature=0.7,
            max_tokens=2048
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error al obtener respuesta: {e}"

# ============================================================
# INTERFAZ DE CHAT
# ============================================================

# Grabación de audio (si está disponible)
audio_data = None
if MIC_RECORDER_AVAILABLE:
    try:
        audio_data = mic_recorder(
            start_prompt="🎤 Iniciar Grabación",
            stop_prompt="🛑 Detener Grabación",
            just_once=False,
            key="mic_main_btn"
        )
    except Exception:
        pass  # Ignorar errores de micrófono

# Procesar audio si se grabó
if audio_data:
    try:
        audio_bytes = audio_data['bytes']
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = f"audio.{audio_data['format']}"
        
        # Transcribir audio con Whisper
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3-turbo",
            language="es"
        )
        
        if transcription.text:
            st.toast(f"🎤 Escuché: {transcription.text}")
            prompt = transcription.text
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Detectar modo planeación
            if "vamos a planear" in prompt.lower():
                st.session_state.planning_mode = True
                st.toast("📑 Modo Planeación Activado")
            
            # Obtener respuesta
            with st.spinner("Pensando..."):
                ai_response = get_ai_response(
                    prompt, 
                    st.session_state.planning_mode
                )
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": ai_response
            })
            
    except Exception as e:
        st.error(f"Error de audio: {e}")

# Input de chat
if prompt := st.chat_input("Escribe tu mensaje..."):
    # Añadir mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Detectar modo planeación
    if "vamos a planear" in prompt.lower():
        st.session_state.planning_mode = True
        st.toast("📑 Modo Planeación Activado")
    
    # Obtener respuesta
    with st.spinner("Pensando..."):
        ai_response = get_ai_response(prompt, st.session_state.planning_mode)
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_response
    })

# ============================================================
# MOSTRAR HISTORIAL DE CHAT
# ============================================================

st.markdown("<div class='chat-wrapper'>", unsafe_allow_html=True)

chat_container = st.container(height=450, key="chat_container")

with chat_container:
    for i, message in enumerate(st.session_state.messages):
        if message["role"] != "system":
            avatar = "🔥" if message["role"] == "assistant" else "👤"
            
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
                
                # Botón de audio para respuestas del asistente
                if message["role"] == "assistant" and voice_enabled:
                    import streamlit.components.v1 as components
                    components.html(
                        get_audio_button_html(message["content"], f"audio_{i}"),
                        height=50,
                    )

st.markdown("</div>", unsafe_allow_html=True)
