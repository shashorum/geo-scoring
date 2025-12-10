import streamlit as st
import pandas as pd
import re
from urllib.parse import urlparse
from collections import Counter

# Lazy imports - solo se cargan cuando se necesitan
@st.cache_resource
def load_plotly():
    import plotly.express as px
    import plotly.graph_objects as go
    return px, go

# ==============================================================================
# 🎯 CONSTANTES Y PATRONES
# ==============================================================================

# Patrones de queries conversacionales (Manteniendo la lógica GEO)
PATTERNS = {
    'qué es': {'regex': r'\bqu[ée]\s+es\b', 'tipo': 'Informacional - Definición', 'funnel': 'TOFU', 'score_base': {'intencion': 0, 'recomendable': 1}},
    'qué son': {'regex': r'\bqu[ée]\s+son\b', 'tipo': 'Informacional - Definición', 'funnel': 'TOFU', 'score_base': {'intencion': 0, 'recomendable': 1}},
    'qué estudia': {'regex': r'\bqu[ée]\s+estudia\b', 'tipo': 'Informacional - Definición', 'funnel': 'TOFU', 'score_base': {'intencion': 1, 'recomendable': 1}},
    'cómo': {'regex': r'\bc[oó]mo\s+(?:hacer|se\s+hace|elaborar|crear|funciona|ser)\b', 'tipo': 'Informacional - Proceso', 'funnel': 'TOFU', 'score_base': {'intencion': 1, 'recomendable': 2}},
    'para qué sirve': {'regex': r'\bpara\s+qu[ée]\s+sirve\b', 'tipo': 'Informacional - Utilidad', 'funnel': 'TOFU', 'score_base': {'intencion': 1, 'recomendable': 1}},
    'requisitos': {'regex': r'\brequisitos\s+para\b', 'tipo': 'Investigación Comercial', 'funnel': 'MOFU', 'score_base': {'intencion': 2, 'recomendable': 2}},
    'qué necesito': {'regex': r'\bqu[ée]\s+(?:necesito|se\s+necesita)\b', 'tipo': 'Investigación Comercial', 'funnel': 'MOFU', 'score_base': {'intencion': 2, 'recomendable': 2}},
    'quiero ser': {'regex': r'\bquiero\s+ser\b', 'tipo': 'Investigación Comercial', 'funnel': 'MOFU', 'score_base': {'intencion': 2, 'recomendable': 2}},
    'mejor': {'regex': r'\bmejor(?:es)?\s+(?:carreras?|cursos?|pa[ií]ses?|universidades?)\b', 'tipo': 'Investigación Comercial', 'funnel': 'MOFU', 'score_base': {'intencion': 2, 'recomendable': 2}},
    'diferencia entre': {'regex': r'\bdiferencias?\s+entre\b', 'tipo': 'Investigación Comercial', 'funnel': 'MOFU', 'score_base': {'intencion': 1, 'recomendable': 1}},
    'carreras': {'regex': r'\bcarreras?\s+(?:mejor|m[aá]s|cortas?|universitarias?)\b', 'tipo': 'Investigación Comercial', 'funnel': 'MOFU', 'score_base': {'intencion': 2, 'recomendable': 2}},
    'qué estudiar': {'regex': r'\bqu[ée]\s+estudiar\s+para\b', 'tipo': 'Investigación Comercial', 'funnel': 'MOFU', 'score_base': {'intencion': 2, 'recomendable': 2}},
    'salidas profesionales': {'regex': r'\bsalidas?\s+profesionales?\b', 'tipo': 'Investigación Comercial', 'funnel': 'MOFU', 'score_base': {'intencion': 2, 'recomendable': 2}},
    'cuánto gana': {'regex': r'\bcu[aá]nto\s+gana\b', 'tipo': 'Investigación Comercial', 'funnel': 'MOFU', 'score_base': {'intencion': 2, 'recomendable': 2}},
    'cuánto dura': {'regex': r'\bcu[aá]nto\s+(?:dura|tiempo)\b', 'tipo': 'Investigación Comercial', 'funnel': 'MOFU', 'score_base': {'intencion': 2, 'recomendable': 1}},
    'curso': {'regex': r'\bcursos?\s+(?:de|online|gratis|homologados?)\b', 'tipo': 'Transaccional - Curso', 'funnel': 'BOFU', 'score_base': {'intencion': 2, 'recomendable': 2}},
    'máster': {'regex': r'\bm[aá]ster(?:es)?\s+(?:de|en|online)\b', 'tipo': 'Transaccional - Curso', 'funnel': 'BOFU', 'score_base': {'intencion': 2, 'recomendable': 2}},
    'formación': {'regex': r'\bformaci[oó]n\s+(?:en|de|online|profesional)\b', 'tipo': 'Transaccional - Curso', 'funnel': 'BOFU', 'score_base': {'intencion': 2, 'recomendable': 2}},
    'dónde estudiar': {'regex': r'\bd[oó]nde\s+(?:estudiar|hacer|sacar|puedo)\b', 'tipo': 'Transaccional - Ubicación', 'funnel': 'BOFU', 'score_base': {'intencion': 2, 'recomendable': 2}},
    'precio': {'regex': r'\bprecio(?:s)?\b|\bcu[áa]nto\s+cuesta\b', 'tipo': 'Transaccional - Precio', 'funnel': 'BOFU', 'score_base': {'intencion': 2, 'recomendable': 2}},
    'ejemplos de': {'regex': r'\bejemplos?\s+de\b', 'tipo': 'Informacional - Ejemplos', 'funnel': 'TOFU', 'score_base': {'intencion': 0, 'recomendable': 1}},
    'tipos de': {'regex': r'\btipos?\s+de\b', 'tipo': 'Informacional - Clasificación', 'funnel': 'TOFU', 'score_base': {'intencion': 0, 'recomendable': 1}},
}

DEFAULT_WEIGHTS = {
    'intencion': 1.0, 'contenido': 1.0, 'competencia': 1.0, 'recomendable': 1.0, 'autoridad': 1.0
}

PRIORITY_THRESHOLDS = {
    'CRÍTICA': 8, 'ALTA': 6, 'MEDIA': 4, 'BAJA': 0
}

# ==============================================================================
# 💡 FUNCIONES HELPER (Lógica de Mapeo y Scoring)
# ==============================================================================

def extract_terms(text: str) -> list:
    """Extrae términos significativos de un texto, eliminando stopwords."""
    stopwords = {
        'de', 'la', 'el', 'en', 'y', 'a', 'los', 'las', 'un', 'una',
        'que', 'es', 'por', 'con', 'para', 'del', 'al', 'como', 'se',
        'su', 'mas', 'pero', 'sus', 'le', 'ya', 'o', 'este', 'si',
        'porque', 'esta', 'cuando', 'muy', 'sin', 'sobre', 'ser', 'tiene',
        'tambien', 'fue', 'siendo', 'son', 'entre', 'todo', 'hacer',
        'articulos', 'blog', 'www', 'com', 'https', 'http', 'qué', 'cúal',
        'cuánto', 'cómo'
    }
    terms = re.findall(r'\b[a-záéíóúñü]+\b', str(text).lower())
    terms = [t for t in terms if t not in stopwords and len(t) > 2]
    return terms

def extract_slug(url: str) -> str:
    """Extrae el slug de una URL."""
    try:
        parsed = urlparse(str(url))
        path = parsed.path.rstrip('/')
        slug = path.split('/')[-1] if '/' in path else path
        slug = re.sub(r'\.[a-z]+$', '', slug)
        slug = re.sub(r'\?.*$', '', slug)
        slug = re.sub(r'[-_]', ' ', slug)
        return slug.lower()
    except:
        return ''

def find_best_url_match(query: str, url_terms: dict, url_list: list) -> str:
    """Encuentra la URL más relevante usando matching por términos (rápido)."""
    if not url_list: 
        return None
    query_terms = set(extract_terms(query))
    if not query_terms: 
        return None

    best_url = None
    max_matches = 1  # Mínimo 2 términos coincidentes
    
    for url, slug_terms in url_terms.items():
        # Contar términos coincidentes
        matches = len(query_terms & slug_terms)
        
        if matches > max_matches:
            max_matches = matches
            best_url = url

    return best_url

def match_urls_to_queries(df: pd.DataFrame) -> pd.DataFrame:
    """Mapea queries a URLs usando matching por términos (optimizado para velocidad)."""
    df_result = df.copy()
    
    unique_urls = df_result['page'].dropna().unique().tolist()
    
    # Pre-calcular términos de cada URL como SET (búsqueda O(1))
    url_terms = {url: set(extract_terms(extract_slug(url))) for url in unique_urls}
    
    # Mapear cada query
    df_result['URL Mapeada'] = df_result['query'].apply(
        lambda q: find_best_url_match(q, url_terms, unique_urls)
    )
    
    df_result['Score Contenido'] = df_result['URL Mapeada'].apply(lambda x: 2 if pd.notna(x) else 0)
    
    return df_result


def suggest_url_for_gap(query: str, existing_urls: list) -> dict:
    """Sugiere una estructura de URL para una query sin contenido."""
    terms = extract_terms(query)
    slug = '-'.join(terms[:5])
    
    base_path = '/blog/'
    if existing_urls:
        paths = [urlparse(str(u)).path for u in existing_urls if pd.notna(u)]
        if paths:
            path_counts = {}
            for p in paths:
                parts = p.strip('/').split('/')
                if parts:
                    base = '/' + parts[0] + '/'
                    path_counts[base] = path_counts.get(base, 0) + 1
            if path_counts:
                base_path = max(path_counts, key=path_counts.get)
                
    suggested_url = f"{base_path.strip('/')}/{slug}"
    return {'slug': slug, 'full': suggested_url}


# --- Funciones de Pattern Detector ---
def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Detecta patrones conversacionales en las queries y asigna scores base."""
    df = df.copy()
    df['Query Lower'] = df['query'].astype(str).str.lower()
    df['Patrón'] = 'Otros'
    df['Tipo Intención'] = 'Indefinido'
    df['Fase Funnel'] = 'Indefinido'
    df['Score Intención'] = 0
    df['Score Recomendable'] = 0

    for pattern_name, pattern_config in PATTERNS.items():
        regex = pattern_config['regex']
        update_mask = (df['Patrón'] == 'Otros') & df['Query Lower'].str.contains(regex, regex=True, na=False)
        
        if update_mask.any():
            df.loc[update_mask, 'Patrón'] = pattern_name
            df.loc[update_mask, 'Tipo Intención'] = pattern_config['tipo']
            df.loc[update_mask, 'Fase Funnel'] = pattern_config['funnel']
            df.loc[update_mask, 'Score Intención'] = pattern_config['score_base'].get('intencion', 0)
            df.loc[update_mask, 'Score Recomendable'] = pattern_config['score_base'].get('recomendable', 0)
            
    df = df.drop(columns=['Query Lower'])
    return df

def get_pattern_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Genera estadísticas agregadas por patrón."""
    stats = df.groupby('Patrón').agg({
        'query': 'count', 'clicks': 'sum', 'impresiones': 'sum',
        'Score': 'mean', 'ctr': 'mean', 'posicion': 'mean'
    }).round(2)
    stats.columns = ['Queries', 'Total Clicks', 'Total Impresiones', 'Score Medio', 'CTR Medio', 'Posición Media']
    stats = stats.sort_values('Total Clicks', ascending=False)
    return stats


# --- Funciones de Scoring ---

def get_priority(score: float) -> str:
    """Asigna una etiqueta de prioridad basada en el score."""
    if score >= PRIORITY_THRESHOLDS['CRÍTICA']: return 'CRÍTICA'
    if score >= PRIORITY_THRESHOLDS['ALTA']: return 'ALTA'
    if score >= PRIORITY_THRESHOLDS['MEDIA']: return 'MEDIA'
    return 'BAJA'

def calculate_competition_score(position: pd.Series) -> pd.Series:
    """Calcula score de competencia priorizando el 'Striking Distance' (4-20)."""
    def position_to_score(pos):
        if pd.isna(pos) or pos == 0:
            return 1 
        elif pos <= 3:
            return 1 
        elif pos <= 20:
            return 2 
        elif pos <= 50:
            return 1 
        else:
            return 0 
    return position.apply(position_to_score)

def calculate_authority_score(df: pd.DataFrame) -> pd.Series:
    """Calcula score de autoridad temática basado en CTR y clicks."""
    if 'ctr' not in df.columns or 'clicks' not in df.columns:
        return pd.Series([1] * len(df))

    ctr_median = df['ctr'].median() if not df['ctr'].empty else 0
    clicks_median = df['clicks'].median() if not df['clicks'].empty else 0
    ctr_25 = df['ctr'].quantile(0.25) if not df['ctr'].empty else 0
    clicks_25 = df['clicks'].quantile(0.25) if not df['clicks'].empty else 0

    def authority_to_score_median(row):
        ctr_val = row.get('ctr', 0) or 0
        clicks_val = row.get('clicks', 0) or 0
        if ctr_val > ctr_median and clicks_val > clicks_median:
            return 2 
        elif ctr_val > ctr_25 or clicks_val > clicks_25:
            return 1 
        return 0 

    return df.apply(authority_to_score_median, axis=1)

def calculate_score(df: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    """Calcula el score de probabilidad de mención en LLM para cada query."""
    if weights is None: weights = DEFAULT_WEIGHTS
    df = df.copy()
    
    df['Score Competencia'] = calculate_competition_score(df['posicion'])
    df['Score Autoridad'] = calculate_authority_score(df)
    
    df['Score Raw'] = (
        df['Score Intención'] * weights.get('intencion', 1.0) +
        df['Score Contenido'] * weights.get('contenido', 1.0) +
        df['Score Competencia'] * weights.get('competencia', 1.0) +
        df['Score Recomendable'] * weights.get('recomendable', 1.0) +
        df['Score Autoridad'] * weights.get('autoridad', 1.0)
    )
    
    max_possible = 2 * sum(weights.values())
    df['Score'] = (df['Score Raw'] / max_possible * 10).round(1)
    df['Score'] = df['Score'].clip(0, 10)
    
    return df

# --- Nueva Función para N-Grams (Análisis) ---
def get_ngrams(text_series, n=2, top_k=15):
    """Genera los n-grams más comunes ignorando stopwords básicas."""
    ngrams_list = []
    for text in text_series.astype(str):
        words = extract_terms(text) 

        if len(words) >= n:
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                ngrams_list.append(ngram)
                
    return pd.DataFrame(Counter(ngrams_list).most_common(top_k), columns=['Frase', 'Frecuencia'])


# ==============================================================================
# 🚀 STREAMLIT APP
# ==============================================================================

# Configuración de página
st.set_page_config(
    page_title="GEO Scoring App",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1e293b; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #64748b; margin-bottom: 2rem; }
    .priority-critica { background-color: #fee2e2; color: #991b1b; padding: 0.25rem 0.75rem; border-radius: 9999px; font-weight: 600; }
    .priority-alta { background-color: #ffedd5; color: #9a3412; padding: 0.25rem 0.75rem; border-radius: 9999px; font-weight: 600; }
    .priority-media { background-color: #fef9c3; color: #854d0e; padding: 0.25rem 0.75rem; border-radius: 9999px; font-weight: 600; }
    .priority-baja { background-color: #dcfce7; color: #166534; padding: 0.25rem 0.75rem; border-radius: 9999px; font-weight: 600; }
    .gap-badge { background-color: #fecaca; color: #991b1b; padding: 0.25rem 0.5rem; border-radius: 0.375rem; font-size: 0.75rem; font-weight: 600; }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# Estado de la sesión
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None

# Header
st.markdown('<p class="main-header">🎯 GEO Scoring App</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analiza queries combinadas de GSC y prioriza contenido para Generative Engine Optimization</p>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("📁 Cargar datos")
    
    combined_file = st.file_uploader("CSV de GSC (Consultas + Páginas)", type=['csv'], key="combined")
    
    st.divider()
    
    st.header("⚙️ Configurar Scoring")
    
    with st.expander("Ajustar pesos", expanded=False):
        peso_intencion = st.slider("Intención comercial", 0.0, 2.0, 1.0, 0.1)
        peso_competencia = st.slider("Competencia (posición)", 0.0, 2.0, 1.0, 0.1)
        peso_recomendable = st.slider("Tema recomendable LLM", 0.0, 2.0, 1.0, 0.1)
        peso_autoridad = st.slider("Autoridad temática", 0.0, 2.0, 1.0, 0.1)
    
    st.divider()
    
    st.markdown("### 🧹 Limpieza")
    marca_exclude = st.text_input("Excluir marca (separar por comas)", placeholder="euroinnova, euro innova")
    
    st.divider()
    
    st.markdown("### ℹ️ Formato del CSV")
    st.info("""
    Exporta desde GSC con dimensiones **Query** y **Page**.
    
    Columnas esperadas:
    - `page` o `Top pages`
    - `query` o `Top queries`
    - `clicks` o `Clics`
    - `impressions` o `Impresiones`
    - `ctr` o `CTR`
    - `position` o `Posición`
    """)

# --- Funciones de Carga y Procesamiento ---

def clean_ctr_column(series):
    """Limpia la columna CTR que puede venir como '2,5%' o 0.025"""
    if series.dtype == 'object':
        # Es string, limpiar
        cleaned = series.astype(str).str.replace('%', '', regex=False)
        cleaned = cleaned.str.replace(',', '.', regex=False)
        cleaned = cleaned.str.strip()
        cleaned = pd.to_numeric(cleaned, errors='coerce').fillna(0)
        # Si los valores son pequeños (< 1), es decimal, multiplicar por 100
        if cleaned.max() < 1:
            cleaned = cleaned * 100
        return cleaned
    else:
        # Es numérico
        if series.max() < 1:
            return series * 100
        return series

def clean_url_fragments(url):
    """Elimina fragmentos (#...) de las URLs para agrupar por página real."""
    if pd.isna(url):
        return url
    url_str = str(url)
    # Eliminar todo después de #
    if '#' in url_str:
        return url_str.split('#')[0]
    return url_str

def normalize_columns(df):
    """Normaliza nombres de columnas para manejar diferentes formatos de exportación GSC."""
    df = df.copy()
    
    # Limpiar nombres de columnas
    df.columns = df.columns.str.strip().str.lower()
    
    # Mapeo exhaustivo de posibles nombres
    column_mapping = {
        # Query
        'top queries': 'query',
        'consultas principales': 'query',
        'queries': 'query',
        'consulta': 'query',
        'search query': 'query',
        # Page
        'top pages': 'page',
        'páginas principales': 'page',
        'paginas principales': 'page',
        'página': 'page',
        'pagina': 'page',
        'url': 'page',
        'landing page': 'page',
        # Clicks
        'clics': 'clicks',
        'click': 'clicks',
        # Impressions
        'impressions': 'impresiones',
        'impresions': 'impresiones',
        # Position
        'position': 'posicion',
        'posición': 'posicion',
        'avg position': 'posicion',
        'average position': 'posicion',
        'posicion media': 'posicion',
        'posición media': 'posicion',
    }
    
    df = df.rename(columns=column_mapping)
    
    return df

@st.cache_data(show_spinner="Cargando CSV...")
def load_and_process_data(file_content, file_name):
    """Carga el CSV combinado, normaliza columnas y agrupa. Optimizado para datasets grandes."""
    if file_content is None: 
        return None, "No se proporcionó archivo"
    
    try:
        # Leer CSV con tipos optimizados
        try:
            df = pd.read_csv(
                file_content, 
                encoding='utf-8',
                dtype={'query': 'str', 'page': 'str'},  # Evitar inferencia costosa
                low_memory=True
            )
        except:
            file_content.seek(0)
            df = pd.read_csv(
                file_content, 
                encoding='latin-1',
                dtype={'query': 'str', 'page': 'str'},
                low_memory=True
            )
        
        # Normalizar columnas
        df = normalize_columns(df)
        
        # Verificar columnas requeridas
        required = ['page', 'query', 'clicks', 'impresiones', 'ctr', 'posicion']
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            available = list(df.columns)
            return None, f"Faltan columnas: {missing}. Columnas disponibles: {available}"
        
        # Limpiar CTR
        df['ctr'] = clean_ctr_column(df['ctr'])
        
        # Convertir tipos
        df['clicks'] = pd.to_numeric(df['clicks'], errors='coerce').fillna(0).astype(int)
        df['impresiones'] = pd.to_numeric(df['impresiones'], errors='coerce').fillna(0).astype(int)
        df['posicion'] = pd.to_numeric(df['posicion'], errors='coerce').fillna(0)
        
        # Limpiar fragmentos (#) de las URLs antes de agrupar
        df['page'] = df['page'].apply(clean_url_fragments)
        
        # Agrupar por Page y Query (ahora las URLs con # están unificadas)
        df_grouped = df.groupby(['page', 'query']).agg({
            'clicks': 'sum',
            'impresiones': 'sum',
            'ctr': 'mean',
            'posicion': 'mean'
        }).reset_index()
        
        return df_grouped, None
        
    except Exception as e:
        return None, f"Error al procesar: {str(e)}"

def full_processing_pipeline(df_raw, weights):
    """Ejecuta toda la pipeline de procesamiento y scoring. Optimizado para datasets grandes."""
    
    total_rows = len(df_raw)
    
    # Para datasets muy grandes, limitar el matching de URLs a las top URLs
    if total_rows > 100000:
        st.info(f"📊 Procesando {total_rows:,} filas (modo optimizado para datasets grandes)")
    
    # 1. Detección de patrones (vectorizado, rápido)
    df = detect_patterns(df_raw)
    
    # 2. Mapeo de URLs - Solo usar top URLs por clicks para matching
    if total_rows > 50000:
        # Para datasets grandes, solo considerar top 5000 URLs únicas por clicks
        top_urls = df.groupby('page')['clicks'].sum().nlargest(5000).index.tolist()
        df_for_matching = df[df['page'].isin(top_urls)].copy()
        df = match_urls_to_queries_fast(df, top_urls)
    else:
        df = match_urls_to_queries(df)
    
    # 3. Cálculo de Scoring (vectorizado, rápido)
    df = calculate_score(df, weights)
    
    # 4. Asignar prioridad
    df['Prioridad'] = df['Score'].apply(get_priority)
    
    # 5. Marcar GAPs
    df['Es GAP'] = df['Score Contenido'] == 0
    
    return df


def match_urls_to_queries_fast(df: pd.DataFrame, top_urls: list) -> pd.DataFrame:
    """Versión optimizada de matching para datasets grandes."""
    df_result = df.copy()
    
    # Pre-calcular términos solo de top URLs
    url_terms = {url: set(extract_terms(extract_slug(url))) for url in top_urls}
    
    # Matching vectorizado por chunks
    def fast_match(query):
        query_terms = set(extract_terms(query))
        if not query_terms or len(query_terms) < 2:
            return None
        
        best_url = None
        max_matches = 1
        
        for url, slug_terms in url_terms.items():
            matches = len(query_terms & slug_terms)
            if matches > max_matches:
                max_matches = matches
                best_url = url
        
        return best_url
    
    df_result['URL Mapeada'] = df_result['query'].apply(fast_match)
    df_result['Score Contenido'] = df_result['URL Mapeada'].apply(lambda x: 2 if pd.notna(x) else 0)
    
    return df_result

# --- Bloque principal de carga y procesamiento ---

if combined_file is not None:
    with st.spinner("Procesando datos..."):
        # Cargar datos
        df_raw, error = load_and_process_data(combined_file, combined_file.name)
        
        if error:
            st.error(error)
            st.stop()
        
        if df_raw is not None:
            # Configurar pesos
            weights = {
                'intencion': peso_intencion,
                'competencia': peso_competencia,
                'recomendable': peso_recomendable,
                'autoridad': peso_autoridad
            }
            
            # Procesar
            df_processed = full_processing_pipeline(df_raw, weights)
            
            # Filtrar marca si se especifica
            if marca_exclude:
                terms = [t.strip().lower() for t in marca_exclude.split(',') if t.strip()]
                if terms:
                    mask = df_processed['query'].apply(lambda x: any(term in str(x).lower() for term in terms))
                    original_count = len(df_processed)
                    df_processed = df_processed[~mask].copy()
                    st.sidebar.success(f"Filtradas {original_count - len(df_processed)} queries de marca")
            
            st.session_state.df_processed = df_processed

# --- Visualización de resultados ---
if st.session_state.df_processed is not None:
    df = st.session_state.df_processed
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "🎯 Scoring", "🔍 GAPs", "📈 Análisis", "💾 Exportar"])
    
    # TAB 1: Dashboard
    with tab1:
        px, go = load_plotly()  # Cargar Plotly solo cuando se necesita
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1: st.metric("Total Queries", f"{len(df):,}")
        with col2: st.metric("🔴 Críticas", len(df[df['Prioridad'] == 'CRÍTICA']))
        with col3: st.metric("🟠 Altas", len(df[df['Prioridad'] == 'ALTA']))
        with col4: st.metric("⚠️ GAPs", len(df[df['Es GAP'] == True]))
        with col5: st.metric("Total Clicks", f"{df['clicks'].sum():,}") 
        
        st.divider()
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("### Distribución por Prioridad")
            priority_counts = df['Prioridad'].value_counts()
            colors = {'CRÍTICA': '#ef4444', 'ALTA': '#f97316', 'MEDIA': '#eab308', 'BAJA': '#22c55e'}
            
            fig_priority = px.pie(
                values=priority_counts.values, names=priority_counts.index, color=priority_counts.index,
                color_discrete_map=colors, hole=0.4
            )
            fig_priority.update_layout(margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_priority, use_container_width=True)
        
        with col_chart2:
            st.markdown("### Queries por Patrón")
            pattern_counts = df['Patrón'].value_counts().head(10)
            fig_patterns = px.bar(
                x=pattern_counts.values, y=pattern_counts.index, orientation='h', 
                labels={'x':'Cantidad', 'y':'Patrón'}
            )
            fig_patterns.update_layout(margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_patterns, use_container_width=True)
        
        st.markdown("### 🏆 Top 10 Queries por Score")
        top_queries = df.nlargest(10, 'Score')[['query', 'page', 'Patrón', 'Score', 'Prioridad', 'clicks', 'impresiones', 'posicion']]
        st.dataframe(
            top_queries, use_container_width=True, hide_index=True,
            column_config={
                "query": "Query",
                "page": st.column_config.LinkColumn("Página", display_text="Ver URL"),
                "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=10, format="%.1f"),
                "posicion": st.column_config.NumberColumn("Posición", format="%.1f"),
                "clicks": "Clicks",
                "impresiones": "Impresiones"
            }
        )

    # TAB 2: Scoring
    with tab2:
        st.markdown("### 🎯 Tabla de Scoring Detallada")
        
        col_f1, col_f2, col_f3, col_f4 = st.columns([1,1,1,2])
        
        with col_f1:
            priority_filter = st.multiselect("Prioridad", ['CRÍTICA', 'ALTA', 'MEDIA', 'BAJA'], default=['CRÍTICA', 'ALTA'])
        with col_f2:
            pattern_filter = st.selectbox("Patrón", ['Todos'] + sorted(df['Patrón'].unique().tolist()))
        with col_f3:
            funnel_filter = st.selectbox("Funnel", ['Todos'] + sorted(df['Fase Funnel'].unique().tolist()))
        with col_f4:
            search_query = st.text_input("🔍 Buscar query", placeholder="Escribe para filtrar...")
        
        df_filtered = df.copy()
        if priority_filter:
            df_filtered = df_filtered[df_filtered['Prioridad'].isin(priority_filter)]
        if pattern_filter != 'Todos':
            df_filtered = df_filtered[df_filtered['Patrón'] == pattern_filter]
        if funnel_filter != 'Todos':
            df_filtered = df_filtered[df_filtered['Fase Funnel'] == funnel_filter]
        if search_query:
            df_filtered = df_filtered[df_filtered['query'].str.contains(search_query, case=False, na=False)]
        
        st.caption(f"Mostrando {len(df_filtered):,} de {len(df):,} queries")
        
        df_display = df_filtered.sort_values('Score', ascending=False)[
            ['query', 'Patrón', 'Fase Funnel', 'Score', 'Prioridad', 'clicks', 'impresiones', 'posicion', 'page', 'URL Mapeada', 'Es GAP']
        ]

        st.dataframe(
            df_display, use_container_width=True, height=500, hide_index=True,
            column_config={
                "query": "Query",
                "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=10, format="%.1f"),
                "posicion": st.column_config.NumberColumn("Posición", format="%.1f"),
                "Es GAP": st.column_config.CheckboxColumn("GAP"),
                "page": st.column_config.LinkColumn("Página Actual", display_text="Ver"),
                "URL Mapeada": st.column_config.LinkColumn("URL Sugerida", display_text="Ver")
            }
        )

    # TAB 3: GAPs
    with tab3:
        st.markdown("### ⚠️ GAPs de Contenido")
        st.markdown("Queries donde no se encontró una URL relevante que responda adecuadamente.")
        
        gaps_df = df[df['Es GAP'] == True].sort_values('Score', ascending=False)
        
        if len(gaps_df) > 0:
            col_g1, col_g2, col_g3, col_g4 = st.columns(4)
            with col_g1: st.metric("Total GAPs", len(gaps_df))
            with col_g2: st.metric("GAPs Críticos", len(gaps_df[gaps_df['Prioridad'] == 'CRÍTICA']))
            with col_g3: st.metric("Clicks en GAPs", f"{gaps_df['clicks'].sum():,}")
            with col_g4: st.metric("Impresiones en GAPs", f"{gaps_df['impresiones'].sum():,}")
            
            st.divider()

            st.markdown("### 🔴 GAPs Prioritarios (Score ≥ 6)")
            gaps_prioritarios = gaps_df[gaps_df['Score'] >= 6]
            
            if len(gaps_prioritarios) > 0:
                url_list = df['page'].dropna().unique().tolist()
                
                for _, row in gaps_prioritarios.head(15).iterrows():
                    suggestion = suggest_url_for_gap(row['query'], url_list)
                    with st.expander(f"**{row['query']}** - Score: {row['Score']} | {row['clicks']} clicks"):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f"""
                            - **Patrón:** `{row.get('Patrón', 'N/A')}`
                            - **Funnel:** `{row.get('Fase Funnel', 'N/A')}`
                            - **Prioridad:** **{row['Prioridad']}**
                            """)
                        with col_b:
                            st.markdown(f"""
                            - **Clicks:** {row['clicks']:,}
                            - **Impresiones:** {row['impresiones']:,}
                            - **Posición media:** {row['posicion']:.1f}
                            """)
                        st.markdown(f"**URL sugerida:** `{suggestion['full']}`")
            else:
                st.info("No hay GAPs con score ≥ 6")
                
            st.divider()
            st.markdown("### 📋 Todos los GAPs")
            st.dataframe(
                gaps_df[['query', 'Patrón', 'Score', 'Prioridad', 'clicks', 'impresiones', 'posicion']].head(100),
                use_container_width=True, hide_index=True
            )
        else:
            st.success("🎉 No se encontraron GAPs de contenido")
            
    # TAB 4: Análisis
    with tab4:
        px, go = load_plotly()  # Cargar Plotly solo cuando se necesita
        
        st.markdown("### 📈 Análisis Detallado")
        
        # Scatter plot Score vs Clicks
        col_a1, col_a2 = st.columns(2)
        
        with col_a1:
            st.markdown("#### Score vs Clicks")
            fig_scatter = px.scatter(
                df.head(500), x='clicks', y='Score', color='Prioridad',
                color_discrete_map={'CRÍTICA': '#ef4444', 'ALTA': '#f97316', 'MEDIA': '#eab308', 'BAJA': '#22c55e'},
                hover_data=['query'], opacity=0.6
            )
            fig_scatter.update_layout(margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col_a2:
            st.markdown("#### Distribución de Scores")
            fig_hist = px.histogram(df, x='Score', nbins=20, color='Prioridad',
                color_discrete_map={'CRÍTICA': '#ef4444', 'ALTA': '#f97316', 'MEDIA': '#eab308', 'BAJA': '#22c55e'})
            fig_hist.update_layout(margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_hist, use_container_width=True)
        
        st.divider()
        
        # N-GRAMS
        st.markdown("### 🧠 Detección de Temas (N-Grams)")
        df_high_priority = df[df['Prioridad'].isin(['CRÍTICA', 'ALTA'])]
        
        if len(df_high_priority) > 20:
            col_n1, col_n2 = st.columns(2)
            
            with col_n1:
                st.markdown("**Bigramas (2 palabras) - Queries CRÍTICA/ALTA**")
                bigrams = get_ngrams(df_high_priority['query'], n=2, top_k=15)
                st.dataframe(bigrams, use_container_width=True, hide_index=True)
                
            with col_n2:
                st.markdown("**Trigramas (3 palabras) - Queries CRÍTICA/ALTA**")
                trigrams = get_ngrams(df_high_priority['query'], n=3, top_k=15)
                st.dataframe(trigrams, use_container_width=True, hide_index=True)
        else:
            st.warning(f"Se necesitan más datos (mínimo 20 queries en CRÍTICA/ALTA). Actualmente hay {len(df_high_priority)}.")

        st.divider()
        
        # Estadísticas por patrón
        st.markdown("### Estadísticas por Patrón de Intención")
        patron_stats = get_pattern_stats(df)
        st.dataframe(patron_stats, use_container_width=True)
        
        # Por fase de funnel
        st.markdown("### Estadísticas por Fase del Funnel")
        funnel_stats = df.groupby('Fase Funnel').agg({
            'query': 'count', 'clicks': 'sum', 'impresiones': 'sum', 'Score': 'mean'
        }).round(2)
        funnel_stats.columns = ['Queries', 'Total Clicks', 'Total Impresiones', 'Score Medio']
        funnel_stats = funnel_stats.sort_values('Score Medio', ascending=False)
        st.dataframe(funnel_stats, use_container_width=True)

    # TAB 5: Exportar
    with tab5:
        st.markdown("### 💾 Exportar Datos")
        
        col_e1, col_e2, col_e3 = st.columns(3)
        
        with col_e1:
            st.markdown("#### CSV Completo")
            csv_full = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar Todo", data=csv_full, 
                file_name="geo_scoring_completo.csv", mime="text/csv"
            )
            st.caption(f"{len(df):,} registros")
        
        with col_e2:
            st.markdown("#### Solo GAPs")
            gaps_export = df[df['Es GAP'] == True]
            csv_gaps = gaps_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar GAPs", data=csv_gaps, 
                file_name="geo_scoring_gaps.csv", mime="text/csv"
            )
            st.caption(f"{len(gaps_export):,} registros")
        
        with col_e3:
            st.markdown("#### Prioritarios (Score ≥ 6)")
            priority_export = df[df['Score'] >= 6]
            csv_priority = priority_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar Prioritarios", data=csv_priority, 
                file_name="geo_scoring_prioritarios.csv", mime="text/csv"
            )
            st.caption(f"{len(priority_export):,} registros")
        
        st.divider()
        
        st.markdown("### 📋 Resumen para documentación")
        resumen = f"""## Resumen GEO Scoring

**Fecha análisis:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

### Métricas Generales
- Total queries analizadas: {len(df):,}
- Queries CRÍTICA: {len(df[df['Prioridad'] == 'CRÍTICA'])}
- Queries ALTA: {len(df[df['Prioridad'] == 'ALTA'])}
- GAPs detectados: {len(df[df['Es GAP'] == True])}
- Total clicks: {df['clicks'].sum():,}
- Total impresiones: {df['impresiones'].sum():,}

### Top 5 Oportunidades (por Score)
"""
        for i, row in df.nlargest(5, 'Score').iterrows():
            resumen += f"\n- **{row['query']}** (Score: {row['Score']}, Clicks: {row['clicks']:,})"
        
        st.code(resumen, language="markdown")
        
        st.download_button(
            label="📥 Descargar Resumen", data=resumen,
            file_name="geo_scoring_resumen.md", mime="text/markdown"
        )

else:
    # Estado inicial
    st.markdown("---")
    st.markdown("""
    ### 👈 Sube tu CSV de Google Search Console
    
    **Pasos:**
    1. Ve a [Google Search Console](https://search.google.com/search-console)
    2. Selecciona tu propiedad
    3. Ve a **Rendimiento** → **Resultados de búsqueda**
    4. Añade dimensiones: **Query** + **Page**
    5. Exporta como CSV
    6. Sube el archivo aquí
    
    ---
    
    ### 🎯 ¿Qué hace esta app?
    
    1. **Detecta patrones** de intención en tus queries (informacional, comercial, transaccional)
    2. **Calcula un Score** de probabilidad de mención en LLMs
    3. **Identifica GAPs** de contenido (queries sin URL relevante)
    4. **Prioriza** qué optimizar primero para GEO
    """)
    
    # Datos de ejemplo
    with st.expander("Ver datos de ejemplo"):
        example_data = pd.DataFrame({
            'page': ['https://ejemplo.com/blog/que-es-seo', 'https://ejemplo.com/curso-marketing'],
            'query': ['qué es el seo', 'curso marketing digital'],
            'clicks': [150, 80],
            'impresiones': [5000, 2000],
            'ctr': [3.0, 4.0],
            'posicion': [8.5, 12.3]
        })
        st.dataframe(example_data)
        st.caption("Tu CSV debe tener un formato similar a este")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #94a3b8; font-size: 0.875rem;'>🎯 GEO Scoring App v1.1 | Optimiza tu contenido para LLMs</p>",
    unsafe_allow_html=True
)
