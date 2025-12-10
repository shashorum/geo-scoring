import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from io import StringIO
import numpy as np
from difflib import SequenceMatcher
from urllib.parse import urlparse
from collections import Counter

# ==============================================================================
# 🎯 CONSTANTES Y PATRONES
# ==============================================================================

# Patrones de queries conversacionales (Manteniendo la lógica GEO)
PATTERNS = {
    'qué es': {'regex': r'\bqu[ée]\s+es\b', 'tipo': 'Informacional - Definición', 'funnel': 'TOFU', 'score_base': {'intencion': 0, 'recomendable': 1}},
    'qué son': {'regex': r'\bqu[ée]\s+son\b', 'tipo': 'Informacional - Definición', 'funnel': 'TOFU', 'score_base': {'intencion': 0, 'recomendable': 1}},
    'requisitos': {'regex': r'\brequisitos\s+para\b', 'tipo': 'Investigación Comercial', 'funnel': 'MOFU', 'score_base': {'intencion': 2, 'recomendable': 2}},
    'comparativa': {'regex': r'\bvs\b|\bcomparativa\b|\bmejor\s+opci[oó]n\b', 'tipo': 'Investigación Comercial', 'funnel': 'MOFU', 'score_base': {'intencion': 1, 'recomendable': 2}},
    'curso': {'regex': r'\bcursos?\s+(?:de|online|gratis|homologados?)\b', 'tipo': 'Transaccional - Curso', 'funnel': 'BOFU', 'score_base': {'intencion': 2, 'recomendable': 2}},
    'precio': {'regex': r'\bprecio(?:s)?\b|\bcu[áa]nto\s+cuesta\b', 'tipo': 'Transaccional - Precio', 'funnel': 'BOFU', 'score_base': {'intencion': 2, 'recomendable': 2}},
    'dónde estudiar': {'regex': r'\bd[oó]nde\s+estudiar\b', 'tipo': 'Transaccional - Dónde', 'funnel': 'BOFU', 'score_base': {'intencion': 2, 'recomendable': 2}}
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
    terms = re.findall(r'\b[a-záéíóúñü]+\b', text.lower())
    terms = [t for t in terms if t not in stopwords and len(t) > 2]
    return terms

def extract_slug(url: str) -> str:
    """Extrae el slug de una URL."""
    try:
        parsed = urlparse(url)
        path = parsed.path.rstrip('/')
        slug = path.split('/')[-1] if '/' in path else path
        slug = re.sub(r'\.[a-z]+$', '', slug)
        slug = re.sub(r'\?.*$', '', slug)
        slug = re.sub(r'[-_]', ' ', slug)
        return slug.lower()
    except:
        return ''

def find_best_url_match(query: str, url_terms: dict, url_list: list) -> str:
    """
    Función de matching simplificada para el nuevo formato. 
    Busca la URL más relevante entre las páginas ya asociadas a otras queries.
    """
    if not url_list: return None
    query_terms = extract_terms(query)
    if not query_terms: return None

    best_url = None
    max_score = 0.5 # Umbral mínimo
    
    query_clean = ' '.join(query_terms)
    
    for url, slug in url_terms.items():
        # Score de similitud de texto (Fuzzy)
        score = SequenceMatcher(None, query_clean, slug).ratio()
        
        # Ponderación extra por términos exactos
        exact_matches = sum(1 for term in query_terms if term in slug)
        score += exact_matches * 0.1
        
        if score > max_score:
            max_score = score
            best_url = url

    return best_url

def match_urls_to_queries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajuste: Mapea queries a otras URLs del mismo dataset si hay mejor coincidencia.
    Esto permite detectar GAPs incluso en el CSV combinado.
    """
    df_result = df.copy()
    
    # 1. Crear un diccionario de slugs de URLs únicas
    unique_urls = df_result['Page'].dropna().unique().tolist()
    url_terms = {url: extract_slug(url) for url in unique_urls}
    
    # 2. Mapear cada query a la mejor URL (incluso si no es la URL original de la fila)
    df_result['URL Mapeada'] = df_result['Query'].apply(lambda q: find_best_url_match(q, url_terms, unique_urls))
    
    # 3. Marcar GAP o Contenido Existente
    # Si la URL Mapeada no es nula, significa que hay *alguna* URL en el sitio que responde
    df_result['Score Contenido'] = df_result['URL Mapeada'].apply(lambda x: 2 if pd.notna(x) else 0)
    
    return df_result


def suggest_url_for_gap(query: str, existing_urls: list) -> dict:
    """Sugiere una estructura de URL para una query sin contenido (usando la lista de URLs existentes)."""
    # Mantenemos la función de sugerencia original para mantener la coherencia
    terms = extract_terms(query)
    slug = '-'.join(terms[:5])
    
    base_path = '/blog/'
    if existing_urls:
        paths = [urlparse(u).path for u in existing_urls if pd.notna(u)]
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
    df['Query Lower'] = df['Query'].str.lower()
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
    # ... (Se mantiene la lógica original)
    stats = df.groupby('Patrón').agg({
        'Query': 'count', 'Clicks': 'sum', 'Impresiones': 'sum',
        'Score': 'mean', 'CTR': 'mean', 'Posición': 'mean'
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
    """
    Mejora: Calcula score de competencia priorizando el 'Striking Distance' (4-20).
    """
    def position_to_score(pos):
        if pd.isna(pos) or pos == 0:
            return 1 
        elif pos <= 3:
            return 1 # Defensa
        elif pos <= 20:
            return 2 # ZONA DE ORO (Striking Distance)
        elif pos <= 50:
            return 1 # Oportunidad lejana
        else:
            return 0 # Irrelevante
    return position.apply(position_to_score)

def calculate_authority_score(df: pd.DataFrame) -> pd.Series:
    """Calcula score de autoridad temática basado en CTR y clicks."""
    if 'CTR' not in df.columns or 'Clicks' not in df.columns:
        return pd.Series([1] * len(df))

    # Usar la mediana para medir el rendimiento medio
    ctr_median = df['CTR'].median() if not df['CTR'].empty else 0
    clicks_median = df['Clicks'].median() if not df['Clicks'].empty else 0
    ctr_25 = df['CTR'].quantile(0.25) if not df['CTR'].empty else 0
    clicks_25 = df['Clicks'].quantile(0.25) if not df['Clicks'].empty else 0

    def authority_to_score_median(row):
        if row['CTR'] > ctr_median and row['Clicks'] > clicks_median:
            return 2 # Alto rendimiento (Máxima autoridad)
        elif row['CTR'] > ctr_25 or row['Clicks'] > clicks_25:
            return 1 # Rendimiento medio
        return 0 # Bajo rendimiento

    return df.apply(authority_to_score_median, axis=1)

def calculate_score(df: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    """Calcula el score de probabilidad de mención en LLM para cada query."""
    if weights is None: weights = DEFAULT_WEIGHTS
    df = df.copy()
    
    # Calcular scores dinámicos
    df['Score Competencia'] = calculate_competition_score(df['Posición'])
    df['Score Autoridad'] = calculate_authority_score(df)
    
    # Calcular score total ponderado
    df['Score Raw'] = (
        df['Score Intención'] * weights.get('intencion', 1.0) +
        df['Score Contenido'] * weights.get('contenido', 1.0) +
        df['Score Competencia'] * weights.get('competencia', 1.0) +
        df['Score Recomendable'] * weights.get('recomendable', 1.0) +
        df['Score Autoridad'] * weights.get('autoridad', 1.0)
    )
    
    # Normalizar a escala 0-10
    max_possible = 2 * sum(weights.values())
    df['Score'] = (df['Score Raw'] / max_possible * 10).round(1)
    df['Score'] = df['Score'].clip(0, 10)
    
    return df

# --- Nueva Función para N-Grams (Análisis) ---
def get_ngrams(text_series, n=2, top_k=15):
    """Genera los n-grams más comunes ignorando stopwords básicas."""
    stopwords = set([
        'de', 'la', 'el', 'en', 'y', 'a', 'los', 'las', 'un', 'una',
        'que', 'es', 'por', 'con', 'para', 'del', 'al', 'como', 'se',
        'su', 'mas', 'pero', 'sus', 'le', 'ya', 'o', 'este', 'si',
        'porque', 'esta', 'cuando', 'muy', 'sin', 'sobre', 'ser', 'tiene',
        'tambien', 'fue', 'siendo', 'son', 'entre', 'todo', 'hacer',
        'articulos', 'blog', 'www', 'com', 'https', 'http', 'qué', 'cúal',
        'cuánto', 'cómo'
    ])

    ngrams_list = []
    for text in text_series.astype(str):
        words = extract_terms(text) 
        words = [w for w in words if w not in stopwords]

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

# CSS personalizado (Se mantiene el estilo para prioridades)
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
    
    # --- CAMBIO CLAVE: UN SOLO UPLOADER ---
    combined_file = st.file_uploader("CSV de GSC (Consultas + Páginas)", type=['csv'], key="combined")
    
    st.divider()
    
    # Configuración de scoring
    st.header("⚙️ Configurar Scoring")
    
    with st.expander("Ajustar pesos", expanded=False):
        peso_intencion = st.slider("Intención comercial (0-2)", 0.0, 2.0, 1.0, 0.1)
        peso_competencia = st.slider("Competencia baja (0-2)", 0.0, 2.0, 1.0, 0.1)
        peso_recomendable = st.slider("Tema recomendable LLM (0-2)", 0.0, 2.0, 1.0, 0.1)
        peso_autoridad = st.slider("Autoridad temática (0-2)", 0.0, 2.0, 1.0, 0.1)
    
    st.divider()
    
    st.markdown("### 🧹 Limpieza")
    # Filtro de marca (Mejora 3)
    marca_exclude = st.text_input("Excluir marca (separar por comas)", placeholder="miempresa, mi marca")
    
    st.divider()
    
    st.markdown("### ℹ️ Cómo usar")
    st.info("Exporta tus datos de GSC seleccionando las dimensiones **Query** y **Page** para obtener un CSV que contenga: `Page, Query, Clicks, Impressions, CTR, Position`.")

# --- Funciones de Carga y Procesamiento ---

def load_data_combined(file_uploader):
    """Carga el CSV combinado, normaliza columnas y agrupa por Page/Query."""
    if file_uploader is None: return None
    try:
        # Intento de lectura con utf-8 y luego latin-1
        try:
            df = pd.read_csv(file_uploader, encoding='utf-8')
        except:
            file_uploader.seek(0)
            df = pd.read_csv(file_uploader, encoding='latin-1')
        
        # Normalizar nombres de columnas esperadas
        column_mapping = {
            'Top queries': 'Query', 'Consultas principales': 'Query', 
            'Page': 'Page', 'Páginas principales': 'Page', 'URL': 'Page',
            'Clicks': 'Clicks', 'Impresiones': 'Impresiones', 
            'Position': 'Posición', 'Posición': 'Posición'
        }
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Filtrar para asegurar que tenemos las columnas mínimas
        required_cols = ['Page', 'Query', 'Clicks', 'Impresiones', 'CTR', 'Posición']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            st.error(f"Faltan columnas requeridas en el CSV: {', '.join(missing)}. Asegúrate de incluir Page y Query.")
            return None

        # Limpiar CTR
        if df['CTR'].dtype == 'object':
            df['CTR'] = df['CTR'].astype(str).str.replace('%', '').str.replace(',', '.').astype(float)
            
        # Agrupar por Page y Query (GSC puede tener filas duplicadas en la exportación combinada)
        df_grouped = df.groupby(['Page', 'Query']).agg({
            'Clicks': 'sum',
            'Impresiones': 'sum',
            'CTR': 'mean',
            'Posición': 'mean'
        }).reset_index()

        return df_grouped.fillna({'Clicks': 0, 'Impresiones': 0, 'Posición': 0, 'CTR': 0})
        
    except Exception as e:
        st.error(f"Error al cargar el CSV combinado: {str(e)}")
        return None

def full_processing_pipeline(df_raw, weights):
    # 1. Detección de patrones
    df = detect_patterns(df_raw)
    
    # 2. Mapeo de URLs (Se auto-mapea contra las URLs de la columna 'Page')
    df = match_urls_to_queries(df)
    
    # 3. Cálculo de Scoring (Incluye la mejora de Striking Distance)
    df = calculate_score(df, weights)
    
    # 4. Asignar prioridad
    df['Prioridad'] = df['Score'].apply(get_priority)
    
    # 5. Marcar GAPs
    # Un GAP es una Query para la que NO encontramos una URL mapeada, 
    # incluso si esa query ya está en una página del CSV.
    df['Es GAP'] = df['Score Contenido'] == 0
    
    return df

# Cargar datos
df_combined_raw = load_data_combined(combined_file)

# Procesar datos si hay queries cargadas
if df_combined_raw is not None:
    weights = {
        'intencion': peso_intencion,
        'competencia': peso_competencia,
        'recomendable': peso_recomendable,
        'autoridad': peso_autoridad
    }
    
    df_raw_processed = full_processing_pipeline(df_combined_raw, weights)

    # LÓGICA DE FILTRADO DE MARCA
    if marca_exclude:
        terms = [t.strip().lower() for t in marca_exclude.split(',') if t.strip()]
        if terms:
            mask = df_raw_processed['Query'].apply(lambda x: any(term in str(x).lower() for term in terms))
            df_processed = df_raw_processed[~mask].copy()
            st.sidebar.success(f"Se filtraron {mask.sum()} queries de marca.")
        else:
            df_processed = df_raw_processed.copy()
    else:
        df_processed = df_raw_processed.copy()

    st.session_state.df_processed = df_processed

# --- Visualización de resultados ---
if st.session_state.df_processed is not None:
    df = st.session_state.df_processed
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "🎯 Scoring", "🔍 GAPs", "📈 Análisis", "💾 Exportar"])
    
    # TAB 1: Dashboard
    with tab1:
        # Métricas principales
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1: st.metric("Total Queries (Filtradas)", f"{len(df):,}")
        with col2: st.metric("🔴 Críticas", len(df[df['Prioridad'] == 'CRÍTICA']))
        with col3: st.metric("🟠 Altas", len(df[df['Prioridad'] == 'ALTA']))
        with col4: st.metric("⚠️ GAPs", len(df[df['Es GAP'] == True]))
        with col5: st.metric("Total Clicks", f"{df['Clicks'].sum():,}")
        
        st.divider()
        
        # Gráficos
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("### Distribución por Prioridad")
            priority_counts = df['Prioridad'].value_counts()
            colors = {'CRÍTICA': '#ef4444', 'ALTA': '#f97316', 'MEDIA': '#eab308', 'BAJA': '#22c55e'}
            
            fig_priority = px.pie(
                values=priority_counts.values, names=priority_counts.index, color=priority_counts.index,
                color_discrete_map=colors, hole=0.4
            )
            st.plotly_chart(fig_priority, use_container_width=True)
        
        with col_chart2:
            st.markdown("### Queries por Patrón")
            if 'Patrón' in df.columns:
                pattern_counts = df['Patrón'].value_counts().head(10)
                fig_patterns = px.bar(
                    x=pattern_counts.values, y=pattern_counts.index, orientation='h', 
                    labels={'x':'Cantidad', 'y':'Patrón'}
                )
                st.plotly_chart(fig_patterns, use_container_width=True)
        
        st.markdown("### 🏆 Top 10 Queries por Score")
        top_queries = df.nlargest(10, 'Score')[['Query', 'Page', 'Patrón', 'Score', 'Prioridad', 'Clicks', 'Impresiones', 'Posición']]
        st.dataframe(
            top_queries, use_container_width=True, hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=10, format="%d"),
                "Posición": st.column_config.NumberColumn("Posición", format="%.1f"),
                "Page": st.column_config.LinkColumn("Página Asociada", display_text="Ver URL")
            }
        )

    # TAB 2: Scoring
    with tab2:
        st.markdown("### 🎯 Tabla de Scoring Detallada")
        st.markdown("Filtra y exporta tu lista de prioridades de contenido.")
        
        # Filtros
        col_f1, col_f2, col_f3 = st.columns([1,1,2])
        priority_filter = col_f1.multiselect("Filtrar por Prioridad", ['CRÍTICA', 'ALTA', 'MEDIA', 'BAJA'], default=['CRÍTICA', 'ALTA'])
        pattern_filter = col_f2.selectbox("Filtrar por Patrón", ['Todos'] + df['Patrón'].unique().tolist())
        
        df_filtered = df[df['Prioridad'].isin(priority_filter)]
        if pattern_filter != 'Todos':
            df_filtered = df_filtered[df_filtered['Patrón'] == pattern_filter]
            
        df_display = df_filtered.sort_values('Score', ascending=False)[
            ['Query', 'Patrón', 'Score', 'Prioridad', 'Clicks', 'Impresiones', 'Posición', 'Page', 'URL Mapeada', 'Es GAP']
        ]

        st.dataframe(
            df_display, use_container_width=True, height=500, hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=10, format="%d"),
                "Posición": st.column_config.NumberColumn("Posición", format="%.1f"),
                "Es GAP": st.column_config.CheckboxColumn("GAP"),
                "Page": st.column_config.LinkColumn("Página Asociada", display_text="Ver URL"),
                "URL Mapeada": st.column_config.LinkColumn("Mejor URL Sugerida", display_text="Ver URL")
            }
        )

    # TAB 3: GAPs
    with tab3:
        st.markdown("### ⚠️ GAPs de Contenido")
        st.markdown("Queries con alto potencial para las que **no se ha encontrado una URL relevante** en tu sitio (incluso entre tus páginas con rendimiento).")
        gaps_df = df[df['Es GAP'] == True].sort_values('Score', ascending=False)
        
        if len(gaps_df) > 0:
            col_g1, col_g2, col_g3 = st.columns(3)
            with col_g1: st.metric("GAPs Críticos", len(gaps_df[gaps_df['Prioridad'] == 'CRÍTICA']))
            with col_g2: st.metric("Clicks en GAPs", f"{gaps_df['Clicks'].sum():,}")
            with col_g3: st.metric("Impresiones en GAPs", f"{gaps_df['Impresiones'].sum():,}")
            st.divider()

            st.markdown("### 🔴 GAPs Prioritarios (Score ≥ 6)")
            gaps_prioritarios = gaps_df[gaps_df['Score'] >= 6]
            
            if len(gaps_prioritarios) > 0:
                url_list = df['Page'].dropna().unique().tolist()
                
                for _, row in gaps_prioritarios.head(10).iterrows():
                    suggestion = suggest_url_for_gap(row['Query'], url_list)
                    with st.expander(f"**{row['Query']}** - 🔴 Score: {row['Score']}", expanded=False):
                        st.markdown(f"""
                        - **Patrón detectado:** `{row.get('Patrón', 'N/A')}`
                        - **Prioridad:** **{row['Prioridad']}**
                        - **Clicks potenciales:** {row['Clicks']:,}
                        - **Posición media:** {row['Posición']:.1f}
                        
                        **Acción sugerida:** Crear un nuevo contenido con la URL sugerida:
                        `{suggestion['full']}` (Slug: `{suggestion['slug']}`)
                        """)
            else:
                st.info("No hay GAPs con score crítico/alto.")
        else:
            st.success("¡Excelente! No se encontraron GAPs de contenido, concéntrate en la optimización de tus URLs existentes.")
            
    # TAB 4: Análisis
    with tab4:
        st.markdown("### 🔍 Análisis Detallado")
        
        # N-GRAMS (Mejora 2)
        st.divider()
        st.markdown("### 🧠 Detección de Temas (N-Grams)")
        st.markdown("Frases más repetidas en las queries críticas (`CRÍTICA` o `ALTA`) que indican temas centrales.")
        
        df_high_priority = df[df['Prioridad'].isin(['CRÍTICA', 'ALTA'])]
        
        if len(df_high_priority) > 50:
            col_n1, col_n2 = st.columns(2)
            
            with col_n1:
                st.markdown("**Bigramas (2 palabras)**")
                bigrams = get_ngrams(df_high_priority['Query'], n=2, top_k=15)
                st.dataframe(bigrams, use_container_width=True, hide_index=True)
                
            with col_n2:
                st.markdown("**Trigramas (3 palabras)**")
                trigrams = get_ngrams(df_high_priority['Query'], n=3, top_k=15)
                st.dataframe(trigrams, use_container_width=True, hide_index=True)
        else:
            st.warning(f"Se necesitan más datos (mínimo 50 queries). Solo hay {len(df_high_priority)} en prioridad CRÍTICA/ALTA para este análisis.")

        # Estadística por patrón
        st.divider()
        st.markdown("### Estadísticas por Patrón de Intención")
        patron_stats = get_pattern_stats(df)
        patron_stats = patron_stats.sort_values('Score Medio', ascending=False)
        st.dataframe(patron_stats, use_container_width=True)

    # TAB 5: Exportar
    with tab5:
        st.markdown("### 💾 Exportar Datos")
        col_e1, col_e2 = st.columns(2)
        
        with col_e1:
            st.markdown("#### CSV Completo")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar CSV Completo", data=csv, file_name="geo_scoring_completo.csv", mime="text/csv"
            )
        
        with col_e2:
            st.markdown("#### Solo GAPs (Oportunidades de Contenido Nuevo)")
            gaps_csv = df[df['Es GAP'] == True].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar GAPs", data=gaps_csv, file_name="geo_scoring_gaps.csv", mime="text/csv"
            )

else:
    st.info("Por favor, sube el CSV combinado de Google Search Console para empezar el análisis.")
