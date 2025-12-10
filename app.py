import streamlit as st
import re
from io import BytesIO
from urllib.parse import urlparse

# ==============================================================================
# 🎯 CONSTANTES Y PATRONES (ligeras, seguras en import)
# ==============================================================================

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

STANDARD_COLS = {
    'page', 'query', 'clicks', 'impresiones', 'ctr', 'posición'
}

# ==============================================================================
# 💡 FUNCIONES HELPER (importan librerías solo cuando se usan)
# ==============================================================================

def extract_terms(text: str) -> list:
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

def find_best_url_match(query: str, url_terms_tokens: dict, url_list: list) -> str:
    query_terms = extract_terms(query)
    if not query_terms:
        return None
    qset = set(query_terms)
    best_url = None
    best_score = 0.0
    for url, slug_tokens in url_terms_tokens.items():
        if not slug_tokens:
            continue
        inter = len(qset & slug_tokens)
        if inter == 0:
            continue
        union = len(qset | slug_tokens)
        score = inter / union if union > 0 else 0.0
        if score > best_score:
            best_score = score
            best_url = url
    return best_url if best_score > 0.0 else None

def match_urls_to_queries(df):
    # import pesado localmente
    import pandas as pd
    df_result = df.copy()
    unique_urls = df_result['page'].dropna().unique().tolist()
    url_terms = {url: extract_slug(url) for url in unique_urls}
    url_terms_tokens = {url: set(extract_terms(slug)) for url, slug in url_terms.items()}
    df_result['URL Mapeada'] = df_result['query'].apply(lambda q: find_best_url_match(q, url_terms_tokens, unique_urls))
    df_result['Score Contenido'] = df_result['URL Mapeada'].apply(lambda x: 2 if pd.notna(x) else 0)
    return df_result

def suggest_url_for_gap(query: str, existing_urls: list) -> dict:
    terms = extract_terms(query)
    slug = '-'.join(terms[:5]) if terms else 'nuevo-contenido'
    base_path = '/blog/'
    if existing_urls:
        paths = [urlparse(u).path for u in existing_urls if u]
        if paths:
            path_counts = {}
            for p in paths:
                parts = p.strip('/').split('/')
                if parts:
                    base = '/' + parts[0] + '/'
                    path_counts[base] = path_counts.get(base, 0) + 1
            if path_counts:
                base_path = max(path_counts, key=path_counts.get)
    suggested_url = f"{base_path.rstrip('/')}/{slug}"
    return {'slug': slug, 'full': suggested_url}

def detect_patterns(df):
    import pandas as pd
    df = df.copy()
    df['Query Lower'] = df['query'].str.lower()
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

def get_pattern_stats(df):
    import pandas as pd
    stats = df.groupby('Patrón').agg({
        'query': 'count', 'clicks': 'sum', 'impresiones': 'sum',
        'Score': 'mean', 'ctr': 'mean', 'posición': 'mean'
    }).round(2)
    stats.columns = ['Queries', 'Total Clicks', 'Total Impresiones', 'Score Medio', 'CTR Medio', 'Posición Media']
    stats = stats.sort_values('Total Clicks', ascending=False)
    return stats

def get_priority(score: float) -> str:
    if score >= PRIORITY_THRESHOLDS['CRÍTICA']: return 'CRÍTICA'
    if score >= PRIORITY_THRESHOLDS['ALTA']: return 'ALTA'
    if score >= PRIORITY_THRESHOLDS['MEDIA']: return 'MEDIA'
    return 'BAJA'

def calculate_competition_score(position):
    import numpy as np
    pos = position.fillna(0).astype(float)
    conds = [
        (pos == 0) | (pos.isna()),
        (pos > 0) & (pos <= 3),
        (pos >= 4) & (pos <= 20),
        (pos > 20) & (pos <= 50)
    ]
    choices = [1, 1, 2, 1]
    scores = np.select(conds, choices, default=0)
    # devuelvo la misma index que entrada
    import pandas as pd
    return pd.Series(scores, index=position.index)

def calculate_authority_score(df):
    import pandas as pd
    if 'ctr' not in df.columns or 'clicks' not in df.columns:
        return pd.Series([1] * len(df), index=df.index)
    ctr = df['ctr'].fillna(0).astype(float)
    clicks = df['clicks'].fillna(0).astype(float)
    ctr_median = ctr.median() if not ctr.empty else 0
    clicks_median = clicks.median() if not clicks.empty else 0
    ctr_25 = ctr.quantile(0.25) if not ctr.empty else 0
    clicks_25 = clicks.quantile(0.25) if not clicks.empty else 0
    score = pd.Series(0, index=df.index)
    mask_top = (ctr > ctr_median) & (clicks > clicks_median)
    mask_mid = (((ctr > ctr_25) | (clicks > clicks_25)) & (~mask_top))
    score.loc[mask_mid] = 1
    score.loc[mask_top] = 2
    return score

def calculate_score(df, weights=None):
    if weights is None: weights = DEFAULT_WEIGHTS
    df = df.copy()
    df['Score Competencia'] = calculate_competition_score(df['posición'])
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

def get_ngrams(text_series, n=2, top_k=15):
    from collections import Counter
    ngrams_list = []
    for text in text_series.astype(str):
        words = extract_terms(text)
        if len(words) >= n:
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                ngrams_list.append(ngram)
    import pandas as pd
    return pd.DataFrame(Counter(ngrams_list).most_common(top_k), columns=['Frase', 'Frecuencia'])

# ==============================================================================
# 🚀 STREAMLIT APP (UI - imports ligeros por defecto)
# ==============================================================================

st.set_page_config(
    page_title="GEO Scoring App",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1e293b; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #64748b; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None

st.markdown('<p class="main-header">🎯 GEO Scoring App</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analiza queries combinadas de GSC y prioriza contenido para Generative Engine Optimization</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("📁 Cargar datos")
    combined_file = st.file_uploader("CSV de GSC (Consultas + Páginas)", type=['csv'], key="combined")
    st.divider()
    st.header("⚙️ Configurar Scoring")
    with st.expander("Ajustar pesos", expanded=False):
        peso_intencion = st.slider("Intención comercial (0-2)", 0.0, 2.0, 1.0, 0.1)
        peso_competencia = st.slider("Competencia baja (0-2)", 0.0, 2.0, 1.0, 0.1)
        peso_recomendable = st.slider("Tema recomendable LLM (0-2)", 0.0, 2.0, 1.0, 0.1)
        peso_autoridad = st.slider("Autoridad temática (0-2)", 0.0, 2.0, 1.0, 0.1)
    st.divider()
    st.markdown("### 🧹 Limpieza")
    marca_exclude = st.text_input("Excluir marca (separar por comas)", placeholder="miempresa, mi marca")
    st.divider()
    st.markdown("### ℹ️ Cómo usar")
    st.info("Exporta tus datos de GSC seleccionando las dimensiones **Query** y **Page** para obtener un CSV que contenga: `page, query, clicks, impressions, ctr, postition`.")

# --- Funciones de Carga y Procesamiento (cacheadas donde corresponde) ---

@st.cache_data(show_spinner=False)
def load_data_combined(file_bytes: bytes, filename: str = ""):
    if not file_bytes:
        return None
    import pandas as pd
    buf = BytesIO(file_bytes)
    try:
        df = pd.read_csv(buf, encoding='utf-8')
    except Exception:
        buf.seek(0)
        df = pd.read_csv(buf, encoding='latin-1')
    df.columns = (df.columns
                  .str.lower().str.strip()
                  .str.replace(' ', '', regex=False)
                  .str.replace('á', 'a', regex=False).str.replace('ó', 'o', regex=False)
                  .str.replace('í', 'i', regex=False).str.replace('é', 'e', regex=False)
                  .str.replace('ú', 'u', regex=False))
    column_mapping_to_standard = {
        'consultasprincipales': 'query', 'topqueries': 'query',
        'paginasprincipales': 'page', 'toppages': 'page', 'url': 'page',
        'clics': 'clicks', 'impressions': 'impresiones', 'impresion': 'impresiones',
        'postition': 'posición', 'position': 'posición', 'posicion': 'posición'
    }
    df = df.rename(columns={k: v for k, v in column_mapping_to_standard.items() if k in df.columns})
    required_cols = list(STANDARD_COLS)
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Faltan columnas requeridas en el CSV: {', '.join(missing)}.")
    if df['ctr'].dtype == 'object':
        df['ctr'] = df['ctr'].astype(str).str.replace('%', '', regex=False).str.replace(',', '.').astype(float)
    df_grouped = df.groupby(['page', 'query']).agg({
        'clicks': 'sum',
        'impresiones': 'sum',
        'ctr': 'mean',
        'posición': 'mean'
    }).reset_index()
    return df_grouped.fillna({'clicks': 0, 'impresiones': 0, 'posición': 0, 'ctr': 0})

def full_processing_pipeline(df_raw, weights):
    df = detect_patterns(df_raw)
    df = match_urls_to_queries(df)
    df = calculate_score(df, weights)
    df['Prioridad'] = df['Score'].apply(get_priority)
    df['Es GAP'] = df['Score Contenido'] == 0
    return df

# --- Bloque principal de carga y procesamiento ---

progress_placeholder = st.empty()
status_text = st.empty()
progress_bar = progress_placeholder.progress(0)

df_combined_raw = None
try:
    if combined_file is not None:
        progress_bar.progress(5)
        status_text.text("1/3: Leyendo y normalizando CSV...")
        file_bytes = combined_file.read()
        df_combined_raw = load_data_combined(file_bytes, combined_file.name)
        progress_bar.progress(40)
        status_text.text("2/3: Agrupando y preparando datos...")
        progress_bar.progress(50)
    else:
        progress_bar.progress(0)
        status_text.text("Esperando la carga del archivo CSV...")
except Exception as e:
    progress_bar.progress(100)
    status_text.text("Error al cargar el CSV.")
    st.error(f"Error al cargar el CSV combinado: {str(e)}")
    df_combined_raw = None

if df_combined_raw is not None:
    weights = {
        'intencion': peso_intencion,
        'competencia': peso_competencia,
        'recomendable': peso_recomendable,
        'autoridad': peso_autoridad
    }
    progress_bar.progress(55)
    status_text.text("Procesando: detectando patrones y calculando scores...")
    try:
        df_raw_processed = full_processing_pipeline(df_combined_raw, weights)
        progress_bar.progress(95)
        status_text.text("Finalizando análisis...")
    except Exception as e:
        progress_bar.progress(100)
        status_text.text("Error en el procesamiento.")
        st.error(f"Error en la pipeline de procesamiento: {str(e)}")
        df_raw_processed = None
    if df_raw_processed is not None:
        if marca_exclude:
            terms = [t.strip().lower() for t in marca_exclude.split(',') if t.strip()]
            if terms:
                mask = df_raw_processed['query'].apply(lambda x: any(term in str(x).lower() for term in terms))
                df_processed = df_raw_processed[~mask].copy()
                st.sidebar.success(f"Se filtraron {mask.sum()} queries de marca.")
            else:
                df_processed = df_raw_processed.copy()
        else:
            df_processed = df_raw_processed.copy()
        st.session_state.df_processed = df_processed
        progress_bar.progress(100)
        status_text.text("Análisis completo.")
        progress_placeholder.empty()
        status_text.empty()

# --- Visualización (importamos plotly/pandas solo cuando vamos a renderizar) ---
if st.session_state.df_processed is not None:
    import pandas as pd
    import plotly.express as px

    df = st.session_state.df_processed
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "🎯 Scoring", "🔍 GAPs", "📈 Análisis", "💾 Exportar"])

    with tab1:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: st.metric("Total Queries (Filtradas)", f"{len(df):,}")
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
            fig_priority = px.pie(values=priority_counts.values, names=priority_counts.index, color=priority_counts.index, color_discrete_map=colors, hole=0.4)
            st.plotly_chart(fig_priority, use_container_width=True)
        with col_chart2:
            st.markdown("### Queries por Patrón")
            if 'Patrón' in df.columns:
                pattern_counts = df['Patrón'].value_counts().head(10)
                fig_patterns = px.bar(x=pattern_counts.values, y=pattern_counts.index, orientation='h', labels={'x':'Cantidad', 'y':'Patrón'})
                st.plotly_chart(fig_patterns, use_container_width=True)

        st.markdown("### 🏆 Top 10 Queries por Score")
        top_queries = df.nlargest(10, 'Score')[['query', 'page', 'Patrón', 'Score', 'Prioridad', 'clicks', 'impresiones', 'posición']]
        top_queries.columns = ['Query', 'Página', 'Patrón', 'Score', 'Prioridad', 'Clicks', 'Impresiones', 'Posición']
        st.dataframe(top_queries, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("### 🎯 Tabla de Scoring Detallada")
        col_f1, col_f2, col_f3 = st.columns([1,1,2])
        priority_filter = col_f1.multiselect("Filtrar por Prioridad", ['CRÍTICA', 'ALTA', 'MEDIA', 'BAJA'], default=['CRÍTICA', 'ALTA'])
        pattern_filter = col_f2.selectbox("Filtrar por Patrón", ['Todos'] + df['Patrón'].unique().tolist())
        df_filtered = df[df['Prioridad'].isin(priority_filter)]
        if pattern_filter != 'Todos':
            df_filtered = df_filtered[df_filtered['Patrón'] == pattern_filter]
        df_display = df_filtered.sort_values('Score', ascending=False)[
            ['query', 'Patrón', 'Score', 'Prioridad', 'clicks', 'impresiones', 'posición', 'page', 'URL Mapeada', 'Es GAP']
        ]
        df_display.columns = ['Query', 'Patrón', 'Score', 'Prioridad', 'Clicks', 'Impresiones', 'Posición', 'Página Asociada', 'Mejor URL Sugerida', 'GAP']
        st.dataframe(df_display, use_container_width=True, height=500, hide_index=True)

    with tab3:
        st.markdown("### ⚠️ GAPs de Contenido")
        gaps_df = df[df['Es GAP'] == True].sort_values('Score', ascending=False)
        if len(gaps_df) > 0:
            col_g1, col_g2, col_g3 = st.columns(3)
            with col_g1: st.metric("GAPs Críticos", len(gaps_df[gaps_df['Prioridad'] == 'CRÍTICA']))
            with col_g2: st.metric("Clicks en GAPs", f"{gaps_df['clicks'].sum():,}")
            with col_g3: st.metric("Impresiones en GAPs", f"{gaps_df['impresiones'].sum():,}")
            st.divider()
            st.markdown("### 🔴 GAPs Prioritarios (Score ≥ 6)")
            gaps_prioritarios = gaps_df[gaps_df['Score'] >= 6]
            if len(gaps_prioritarios) > 0:
                url_list = df['page'].dropna().unique().tolist()
                for _, row in gaps_prioritarios.head(10).iterrows():
                    suggestion = suggest_url_for_gap(row['query'], url_list)
                    with st.expander(f"**{row['query']}** - 🔴 Score: {row['Score']}", expanded=False):
                        st.markdown(f"- **Patrón detectado:** `{row.get('Patrón', 'N/A')}`\n- **Prioridad:** **{row['Prioridad']}**\n- **Clicks potenciales:** {row['clicks']:,}\n- **Posición media:** {row['posición']:.1f}\n\n**Acción sugerida:** `{suggestion['full']}` (Slug: `{suggestion['slug']}`)")
            else:
                st.info("No hay GAPs con score crítico/alto.")
        else:
            st.success("¡Excelente! No se encontraron GAPs de contenido.")

    with tab4:
        st.markdown("### 🔍 Análisis Detallado")
        st.divider()
        st.markdown("### 🧠 Detección de Temas (N-Grams)")
        df_high_priority = df[df['Prioridad'].isin(['CRÍTICA', 'ALTA'])]
        if len(df_high_priority) > 50:
            col_n1, col_n2 = st.columns(2)
            with col_n1:
                st.markdown("**Bigramas (2 palabras)**")
                bigrams = get_ngrams(df_high_priority['query'], n=2, top_k=15)
                st.dataframe(bigrams, use_container_width=True, hide_index=True)
            with col_n2:
                st.markdown("**Trigramas (3 palabras)**")
                trigrams = get_ngrams(df_high_priority['query'], n=3, top_k=15)
                st.dataframe(trigrams, use_container_width=True, hide_index=True)
        else:
            st.warning(f"Se necesitan más datos (mínimo 50 queries). Solo hay {len(df_high_priority)} en prioridad CRÍTICA/ALTA para este análisis.")
        st.divider()
        st.markdown("### Estadísticas por Patrón de Intención")
        patron_stats = get_pattern_stats(df)
        patron_stats = patron_stats.sort_values('Score Medio', ascending=False)
        st.dataframe(patron_stats, use_container_width=True)

    with tab5:
        st.markdown("### 💾 Exportar Datos")
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.markdown("#### CSV Completo")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Descargar CSV Completo", data=csv, file_name="geo_scoring_completo.csv", mime="text/csv")
        with col_e2:
            st.markdown("#### Solo GAPs (Oportunidades de Contenido Nuevo)")
            gaps_csv = df[df['Es GAP'] == True].to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Descargar GAPs", data=gaps_csv, file_name="geo_scoring_gaps.csv", mime="text/csv")

else:
    st.info("Por favor, sube el CSV combinado de Google Search Console para empezar el análisis.")
