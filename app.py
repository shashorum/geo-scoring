import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.pattern_detector import detect_patterns, PATTERNS
from utils.scoring import calculate_score, get_priority
from utils.url_matcher import match_urls_to_queries
import re
from io import StringIO

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
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
    }
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
if 'df_queries' not in st.session_state:
    st.session_state.df_queries = None
if 'df_urls' not in st.session_state:
    st.session_state.df_urls = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None

# Header
st.markdown('<p class="main-header">🎯 GEO Scoring App</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analiza queries de GSC y prioriza contenido para Generative Engine Optimization</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Google_Search_Console_Logo.svg/1200px-Google_Search_Console_Logo.svg.png", width=50)
    st.header("📁 Cargar datos")
    
    st.markdown("### Queries GSC")
    queries_file = st.file_uploader(
        "CSV de consultas",
        type=['csv'],
        help="Exporta desde GSC: Rendimiento → Consultas → Exportar",
        key="queries"
    )
    
    st.markdown("### URLs GSC")
    urls_file = st.file_uploader(
        "CSV de páginas",
        type=['csv'],
        help="Exporta desde GSC: Rendimiento → Páginas → Exportar",
        key="urls"
    )
    
    st.divider()
    
    # Configuración de scoring
    st.header("⚙️ Configurar Scoring")
    
    with st.expander("Ajustar pesos", expanded=False):
        peso_intencion = st.slider("Intención comercial", 0.0, 2.0, 1.0, 0.1)
        peso_competencia = st.slider("Competencia baja", 0.0, 2.0, 1.0, 0.1)
        peso_recomendable = st.slider("Tema recomendable LLM", 0.0, 2.0, 1.0, 0.1)
        peso_autoridad = st.slider("Autoridad temática", 0.0, 2.0, 1.0, 0.1)
    
    st.divider()
    
    st.markdown("### 📊 Propiedades GSC")
    propiedad = st.text_input("Dominio", placeholder="euroinnova.com")

# Función para procesar datos
def process_data(df_queries, df_urls=None, weights=None):
    if weights is None:
        weights = {
            'intencion': 1.0,
            'competencia': 1.0,
            'recomendable': 1.0,
            'autoridad': 1.0
        }
    
    # Detectar patrones
    df = detect_patterns(df_queries)
    
    # Calcular scoring
    df = calculate_score(df, weights)
    
    # Asignar prioridad
    df['Prioridad'] = df['Score'].apply(get_priority)
    
    # Mapear URLs si están disponibles
    if df_urls is not None:
        df = match_urls_to_queries(df, df_urls)
        df['Es GAP'] = df['URL Mapeada'].isna() | (df['URL Mapeada'] == '')
    else:
        df['URL Mapeada'] = None
        df['Es GAP'] = True
    
    return df

# Cargar y procesar datos
if queries_file is not None:
    try:
        # Leer CSV con diferentes encodings
        try:
            df_queries = pd.read_csv(queries_file, encoding='utf-8')
        except:
            queries_file.seek(0)
            df_queries = pd.read_csv(queries_file, encoding='latin-1')
        
        # Normalizar nombres de columnas
        column_mapping = {
            'Top queries': 'Query',
            'Consultas principales': 'Query',
            'Query': 'Query',
            'Clicks': 'Clicks',
            'Clics': 'Clicks',
            'Impressions': 'Impresiones',
            'Impresiones': 'Impresiones',
            'CTR': 'CTR',
            'Position': 'Posición',
            'Posición': 'Posición'
        }
        df_queries = df_queries.rename(columns={k: v for k, v in column_mapping.items() if k in df_queries.columns})
        
        # Limpiar CTR si viene como porcentaje string
        if 'CTR' in df_queries.columns and df_queries['CTR'].dtype == 'object':
            df_queries['CTR'] = df_queries['CTR'].str.replace('%', '').str.replace(',', '.').astype(float)
        
        st.session_state.df_queries = df_queries
        
    except Exception as e:
        st.error(f"Error al cargar queries: {str(e)}")

if urls_file is not None:
    try:
        try:
            df_urls = pd.read_csv(urls_file, encoding='utf-8')
        except:
            urls_file.seek(0)
            df_urls = pd.read_csv(urls_file, encoding='latin-1')
        
        # Normalizar columnas de URLs
        url_column_mapping = {
            'Top pages': 'URL',
            'Páginas principales': 'URL',
            'Page': 'URL',
            'URL': 'URL'
        }
        df_urls = df_urls.rename(columns={k: v for k, v in url_column_mapping.items() if k in df_urls.columns})
        
        st.session_state.df_urls = df_urls
        
    except Exception as e:
        st.error(f"Error al cargar URLs: {str(e)}")

# Procesar datos si hay queries cargadas
if st.session_state.df_queries is not None:
    weights = {
        'intencion': peso_intencion if 'peso_intencion' in dir() else 1.0,
        'competencia': peso_competencia if 'peso_competencia' in dir() else 1.0,
        'recomendable': peso_recomendable if 'peso_recomendable' in dir() else 1.0,
        'autoridad': peso_autoridad if 'peso_autoridad' in dir() else 1.0
    }
    
    st.session_state.df_processed = process_data(
        st.session_state.df_queries,
        st.session_state.df_urls,
        weights
    )

# Tabs principales
if st.session_state.df_processed is not None:
    df = st.session_state.df_processed
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "🎯 Scoring", "🔍 GAPs", "📈 Análisis", "💾 Exportar"])
    
    # TAB 1: Dashboard
    with tab1:
        # Métricas principales
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Queries",
                f"{len(df):,}",
                help="Número total de queries analizadas"
            )
        
        with col2:
            criticas = len(df[df['Prioridad'] == 'CRÍTICA'])
            st.metric(
                "🔴 Críticas",
                criticas,
                help="Queries con score 8-10"
            )
        
        with col3:
            altas = len(df[df['Prioridad'] == 'ALTA'])
            st.metric(
                "🟠 Altas",
                altas,
                help="Queries con score 6-7"
            )
        
        with col4:
            gaps = len(df[df['Es GAP'] == True])
            st.metric(
                "⚠️ GAPs",
                gaps,
                help="Queries sin URL mapeada"
            )
        
        with col5:
            total_clicks = df['Clicks'].sum()
            st.metric(
                "Total Clicks",
                f"{total_clicks:,}",
                help="Suma de clicks de todas las queries"
            )
        
        st.divider()
        
        # Gráficos
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("### Distribución por Prioridad")
            priority_counts = df['Prioridad'].value_counts()
            colors = {'CRÍTICA': '#ef4444', 'ALTA': '#f97316', 'MEDIA': '#eab308', 'BAJA': '#22c55e'}
            
            fig_priority = px.pie(
                values=priority_counts.values,
                names=priority_counts.index,
                color=priority_counts.index,
                color_discrete_map=colors,
                hole=0.4
            )
            fig_priority.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_priority, use_container_width=True)
        
        with col_chart2:
            st.markdown("### Queries por Patrón")
            if 'Patrón' in df.columns:
                pattern_counts = df['Patrón'].value_counts().head(10)
                
                fig_patterns = px.bar(
                    x=pattern_counts.values,
                    y=pattern_counts.index,
                    orientation='h',
                    color=pattern_counts.values,
                    color_continuous_scale='Blues'
                )
                fig_patterns.update_layout(
                    showlegend=False,
                    xaxis_title="Cantidad",
                    yaxis_title="",
                    margin=dict(t=20, b=20, l=20, r=20),
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_patterns, use_container_width=True)
        
        # Top queries por score
        st.markdown("### 🏆 Top 10 Queries por Score")
        top_queries = df.nlargest(10, 'Score')[['Query', 'Patrón', 'Score', 'Prioridad', 'Clicks', 'Impresiones', 'CTR']]
        
        st.dataframe(
            top_queries,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score",
                    min_value=0,
                    max_value=10,
                    format="%d"
                ),
                "CTR": st.column_config.NumberColumn(
                    "CTR",
                    format="%.2f%%"
                ),
                "Clicks": st.column_config.NumberColumn(
                    "Clicks",
                    format="%d"
                ),
                "Impresiones": st.column_config.NumberColumn(
                    "Impresiones",
                    format="%d"
                )
            },
            hide_index=True,
            use_container_width=True
        )
    
    # TAB 2: Scoring
    with tab2:
        st.markdown("### 🎯 Tabla de Scoring Completa")
        
        # Filtros
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        
        with col_f1:
            filter_priority = st.multiselect(
                "Prioridad",
                options=['CRÍTICA', 'ALTA', 'MEDIA', 'BAJA'],
                default=['CRÍTICA', 'ALTA']
            )
        
        with col_f2:
            if 'Patrón' in df.columns:
                patrones_disponibles = df['Patrón'].dropna().unique().tolist()
                filter_pattern = st.multiselect(
                    "Patrón",
                    options=patrones_disponibles
                )
            else:
                filter_pattern = []
        
        with col_f3:
            if 'Tipo Intención' in df.columns:
                tipos_disponibles = df['Tipo Intención'].dropna().unique().tolist()
                filter_tipo = st.multiselect(
                    "Tipo Intención",
                    options=tipos_disponibles
                )
            else:
                filter_tipo = []
        
        with col_f4:
            filter_gap = st.selectbox(
                "GAPs",
                options=['Todos', 'Solo GAPs', 'Sin GAPs']
            )
        
        # Aplicar filtros
        df_filtered = df.copy()
        
        if filter_priority:
            df_filtered = df_filtered[df_filtered['Prioridad'].isin(filter_priority)]
        
        if filter_pattern:
            df_filtered = df_filtered[df_filtered['Patrón'].isin(filter_pattern)]
        
        if filter_tipo:
            df_filtered = df_filtered[df_filtered['Tipo Intención'].isin(filter_tipo)]
        
        if filter_gap == 'Solo GAPs':
            df_filtered = df_filtered[df_filtered['Es GAP'] == True]
        elif filter_gap == 'Sin GAPs':
            df_filtered = df_filtered[df_filtered['Es GAP'] == False]
        
        # Búsqueda
        search = st.text_input("🔍 Buscar query", placeholder="Escribe para filtrar...")
        if search:
            df_filtered = df_filtered[df_filtered['Query'].str.contains(search, case=False, na=False)]
        
        st.markdown(f"**{len(df_filtered):,}** queries encontradas")
        
        # Tabla
        columns_to_show = ['Query', 'Patrón', 'Tipo Intención', 'Score', 'Prioridad', 'Clicks', 'Impresiones', 'CTR', 'Posición', 'Es GAP']
        columns_available = [c for c in columns_to_show if c in df_filtered.columns]
        
        st.dataframe(
            df_filtered[columns_available].sort_values('Score', ascending=False),
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score",
                    min_value=0,
                    max_value=10,
                    format="%d"
                ),
                "CTR": st.column_config.NumberColumn("CTR", format="%.2f%%"),
                "Posición": st.column_config.NumberColumn("Posición", format="%.1f"),
                "Es GAP": st.column_config.CheckboxColumn("GAP")
            },
            hide_index=True,
            use_container_width=True,
            height=500
        )
    
    # TAB 3: GAPs
    with tab3:
        st.markdown("### ⚠️ GAPs de Contenido")
        st.markdown("Queries con alto potencial que **no tienen URL mapeada** en tu sitio.")
        
        gaps_df = df[df['Es GAP'] == True].sort_values('Score', ascending=False)
        
        if len(gaps_df) > 0:
            # Métricas de GAPs
            col_g1, col_g2, col_g3 = st.columns(3)
            
            with col_g1:
                gaps_criticos = len(gaps_df[gaps_df['Prioridad'] == 'CRÍTICA'])
                st.metric("GAPs Críticos", gaps_criticos)
            
            with col_g2:
                clicks_perdidos = gaps_df['Clicks'].sum()
                st.metric("Clicks en GAPs", f"{clicks_perdidos:,}")
            
            with col_g3:
                impresiones_gap = gaps_df['Impresiones'].sum()
                st.metric("Impresiones en GAPs", f"{impresiones_gap:,}")
            
            st.divider()
            
            # Lista de GAPs prioritarios
            st.markdown("### 🔴 GAPs Prioritarios (Score ≥ 6)")
            
            gaps_prioritarios = gaps_df[gaps_df['Score'] >= 6][['Query', 'Patrón', 'Score', 'Prioridad', 'Clicks', 'Impresiones', 'CTR']]
            
            if len(gaps_prioritarios) > 0:
                st.dataframe(
                    gaps_prioritarios,
                    column_config={
                        "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=10),
                        "CTR": st.column_config.NumberColumn("CTR", format="%.2f%%")
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.info("No hay GAPs con score ≥ 6")
            
            # Sugerencias de contenido
            st.markdown("### 💡 Sugerencias de Contenido")
            
            for _, row in gaps_prioritarios.head(5).iterrows():
                with st.expander(f"**{row['Query']}** - Score: {row['Score']}"):
                    st.markdown(f"""
                    - **Patrón detectado:** {row.get('Patrón', 'N/A')}
                    - **Clicks potenciales:** {row['Clicks']:,}
                    - **Impresiones:** {row['Impresiones']:,}
                    - **CTR actual:** {row['CTR']:.2f}%
                    
                    **Acción sugerida:** Crear contenido optimizado para esta query que responda la intención del usuario y mencione tu oferta formativa.
                    """)
        else:
            st.success("🎉 No se detectaron GAPs. Todas las queries tienen URL mapeada.")
    
    # TAB 4: Análisis
    with tab4:
        st.markdown("### 📈 Análisis Detallado")
        
        col_a1, col_a2 = st.columns(2)
        
        with col_a1:
            st.markdown("#### Score vs Clicks")
            fig_scatter = px.scatter(
                df,
                x='Clicks',
                y='Score',
                color='Prioridad',
                color_discrete_map={'CRÍTICA': '#ef4444', 'ALTA': '#f97316', 'MEDIA': '#eab308', 'BAJA': '#22c55e'},
                hover_data=['Query'],
                size='Impresiones',
                size_max=30
            )
            fig_scatter.update_layout(margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col_a2:
            st.markdown("#### Distribución de Scores")
            fig_hist = px.histogram(
                df,
                x='Score',
                nbins=10,
                color='Prioridad',
                color_discrete_map={'CRÍTICA': '#ef4444', 'ALTA': '#f97316', 'MEDIA': '#eab308', 'BAJA': '#22c55e'}
            )
            fig_hist.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                bargap=0.1
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Análisis por tipo de intención
        if 'Tipo Intención' in df.columns:
            st.markdown("#### Métricas por Tipo de Intención")
            
            tipo_stats = df.groupby('Tipo Intención').agg({
                'Query': 'count',
                'Score': 'mean',
                'Clicks': 'sum',
                'Impresiones': 'sum'
            }).round(2)
            tipo_stats.columns = ['Queries', 'Score Medio', 'Total Clicks', 'Total Impresiones']
            tipo_stats = tipo_stats.sort_values('Score Medio', ascending=False)
            
            st.dataframe(
                tipo_stats,
                column_config={
                    "Score Medio": st.column_config.NumberColumn(format="%.1f"),
                    "Total Clicks": st.column_config.NumberColumn(format="%d"),
                    "Total Impresiones": st.column_config.NumberColumn(format="%d")
                },
                use_container_width=True
            )
        
        # Análisis por patrón
        if 'Patrón' in df.columns:
            st.markdown("#### Métricas por Patrón de Query")
            
            patron_stats = df.groupby('Patrón').agg({
                'Query': 'count',
                'Score': 'mean',
                'Clicks': 'sum',
                'CTR': 'mean'
            }).round(2)
            patron_stats.columns = ['Queries', 'Score Medio', 'Total Clicks', 'CTR Medio']
            patron_stats = patron_stats.sort_values('Score Medio', ascending=False)
            
            fig_patron = px.bar(
                patron_stats.reset_index(),
                x='Patrón',
                y='Score Medio',
                color='Total Clicks',
                color_continuous_scale='Blues'
            )
            fig_patron.update_layout(margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_patron, use_container_width=True)
    
    # TAB 5: Exportar
    with tab5:
        st.markdown("### 💾 Exportar Datos")
        
        col_e1, col_e2, col_e3 = st.columns(3)
        
        with col_e1:
            st.markdown("#### CSV Completo")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar CSV",
                data=csv,
                file_name="geo_scoring_completo.csv",
                mime="text/csv"
            )
        
        with col_e2:
            st.markdown("#### Solo GAPs")
            gaps_csv = df[df['Es GAP'] == True].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar GAPs",
                data=gaps_csv,
                file_name="geo_gaps.csv",
                mime="text/csv"
            )
        
        with col_e3:
            st.markdown("#### Prioritarios (Score ≥ 8)")
            priority_csv = df[df['Score'] >= 8].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar Prioritarios",
                data=priority_csv,
                file_name="geo_prioritarios.csv",
                mime="text/csv"
            )
        
        st.divider()
        
        st.markdown("#### 📋 Resumen para Notion")
        
        resumen = f"""
## Resumen GEO Scoring - {propiedad if propiedad else 'Sin dominio'}

**Fecha:** {pd.Timestamp.now().strftime('%Y-%m-%d')}

### Métricas Generales
- Total queries analizadas: {len(df):,}
- Queries críticas: {len(df[df['Prioridad'] == 'CRÍTICA'])}
- Queries alta prioridad: {len(df[df['Prioridad'] == 'ALTA'])}
- GAPs detectados: {len(df[df['Es GAP'] == True])}

### Top 5 Oportunidades
"""
        for i, row in df.nlargest(5, 'Score').iterrows():
            resumen += f"\n- **{row['Query']}** (Score: {row['Score']}, Clicks: {row['Clicks']:,})"
        
        st.code(resumen, language="markdown")
        
        st.download_button(
            label="📥 Descargar Resumen",
            data=resumen,
            file_name="resumen_geo.md",
            mime="text/markdown"
        )

else:
    # Estado inicial - Sin datos
    st.info("👈 **Sube tus archivos CSV de Google Search Console** en la barra lateral para comenzar el análisis.")
    
    st.markdown("---")
    
    st.markdown("""
    ### 📖 Cómo usar esta app
    
    1. **Exporta datos de GSC:**
       - Ve a Google Search Console → Rendimiento
       - Selecciona el rango de fechas deseado
       - Exporta "Consultas" como CSV
       - Exporta "Páginas" como CSV (opcional, para mapeo de URLs)
    
    2. **Sube los archivos:**
       - Usa los botones en la barra lateral
       - La app detectará automáticamente las columnas
    
    3. **Analiza:**
       - Revisa el dashboard con métricas clave
       - Filtra por prioridad, patrón o tipo de intención
       - Identifica GAPs de contenido
    
    4. **Exporta:**
       - Descarga CSV filtrados
       - Genera resumen para Notion
    
    ---
    
    ### 🎯 Sistema de Scoring
    
    Cada query recibe una puntuación de 0-10 basada en:
    
    | Criterio | Descripción |
    |----------|-------------|
    | **Intención comercial** | ¿La query implica búsqueda de formación? |
    | **Competencia SERP** | ¿Hay baja competencia en Google? |
    | **Tema recomendable** | ¿Los LLMs suelen recomendar cursos aquí? |
    | **Autoridad temática** | ¿Tu sitio es referente en el tema? |
    
    **Prioridades:**
    - 🔴 CRÍTICA (8-10): Optimizar urgente
    - 🟠 ALTA (6-7): Priorizar
    - 🟡 MEDIA (4-5): Oportunidad moderada
    - 🟢 BAJA (0-3): Baja probabilidad
    """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #64748b;'>GEO Scoring App v1.0 | Desarrollado para optimización de contenido en LLMs</p>",
    unsafe_allow_html=True
)
