import streamlit as st
import pandas as pd
import re
from urllib.parse import urlparse
from collections import Counter

# ==============================================================================
# 🎯 CONFIGURACIÓN Y CONSTANTES
# ==============================================================================

st.set_page_config(
    page_title="GEO Scoring App",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Lazy import de Plotly
@st.cache_resource
def load_plotly():
    import plotly.express as px
    import plotly.graph_objects as go
    return px, go

# ==============================================================================
# 🧠 PATRONES DE INTENCIÓN Y CONFIGURACIÓN GEO
# ==============================================================================

PATTERNS = {
    # TOFU - Informacional
    'qué es': {
        'regex': r'\bqu[ée]\s+es\b',
        'tipo_intencion': 'Informacional - Definición',
        'funnel': 'TOFU',
        'tipo_respuesta': 'Explicación',
        'prompt_template': '¿Qué es {tema} y para qué sirve?',
        'cta_sugerido': 'Añadir: ejemplos prácticos, infografía explicativa, enlace a cursos relacionados',
        'score_base': {'intencion': 0, 'recomendable': 1}
    },
    'qué son': {
        'regex': r'\bqu[ée]\s+son\b',
        'tipo_intencion': 'Informacional - Definición',
        'funnel': 'TOFU',
        'tipo_respuesta': 'Explicación',
        'prompt_template': '¿Qué son {tema}?',
        'cta_sugerido': 'Añadir: lista de tipos, características principales, ejemplos',
        'score_base': {'intencion': 0, 'recomendable': 1}
    },
    'qué estudia': {
        'regex': r'\bqu[ée]\s+estudia\b',
        'tipo_intencion': 'Informacional - Definición',
        'funnel': 'TOFU',
        'tipo_respuesta': 'Explicación',
        'prompt_template': '¿Qué estudia {tema} y cuáles son sus ramas?',
        'cta_sugerido': 'Añadir: salidas profesionales, ramas de especialización, cursos introductorios',
        'score_base': {'intencion': 1, 'recomendable': 1}
    },
    'cómo hacer': {
        'regex': r'\bc[oó]mo\s+(?:hacer|se\s+hace|elaborar|crear)\b',
        'tipo_intencion': 'Informacional - Proceso',
        'funnel': 'TOFU',
        'tipo_respuesta': 'Tutorial paso a paso',
        'prompt_template': '¿Cómo puedo hacer {tema} paso a paso?',
        'cta_sugerido': 'Añadir: pasos numerados, plantillas descargables, vídeo tutorial, curso práctico',
        'score_base': {'intencion': 1, 'recomendable': 2}
    },
    'cómo funciona': {
        'regex': r'\bc[oó]mo\s+funciona\b',
        'tipo_intencion': 'Informacional - Proceso',
        'funnel': 'TOFU',
        'tipo_respuesta': 'Explicación técnica',
        'prompt_template': '¿Cómo funciona {tema}?',
        'cta_sugerido': 'Añadir: diagrama de funcionamiento, ejemplos de uso, casos prácticos',
        'score_base': {'intencion': 1, 'recomendable': 1}
    },
    'para qué sirve': {
        'regex': r'\bpara\s+qu[ée]\s+sirve\b',
        'tipo_intencion': 'Informacional - Utilidad',
        'funnel': 'TOFU',
        'tipo_respuesta': 'Lista de beneficios',
        'prompt_template': '¿Para qué sirve {tema} y cuáles son sus aplicaciones?',
        'cta_sugerido': 'Añadir: casos de uso reales, beneficios concretos, testimonios',
        'score_base': {'intencion': 1, 'recomendable': 1}
    },
    'ejemplos de': {
        'regex': r'\bejemplos?\s+de\b',
        'tipo_intencion': 'Informacional - Ejemplos',
        'funnel': 'TOFU',
        'tipo_respuesta': 'Lista con ejemplos',
        'prompt_template': 'Dame ejemplos de {tema}',
        'cta_sugerido': 'Añadir: mínimo 10 ejemplos variados, plantillas, recursos descargables',
        'score_base': {'intencion': 0, 'recomendable': 1}
    },
    'tipos de': {
        'regex': r'\btipos?\s+de\b',
        'tipo_intencion': 'Informacional - Clasificación',
        'funnel': 'TOFU',
        'tipo_respuesta': 'Lista categorizada',
        'prompt_template': '¿Cuáles son los tipos de {tema}?',
        'cta_sugerido': 'Añadir: tabla comparativa, características de cada tipo, cuál elegir según necesidad',
        'score_base': {'intencion': 0, 'recomendable': 1}
    },
    
    # MOFU - Investigación Comercial
    'requisitos para': {
        'regex': r'\brequisitos\s+para\b',
        'tipo_intencion': 'Investigación Comercial',
        'funnel': 'MOFU',
        'tipo_respuesta': 'Lista de requisitos',
        'prompt_template': '¿Cuáles son los requisitos para {tema}?',
        'cta_sugerido': 'Añadir: checklist descargable, pasos para cumplir requisitos, formación necesaria con enlaces',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    'qué necesito': {
        'regex': r'\bqu[ée]\s+(?:necesito|se\s+necesita)\s+para\b',
        'tipo_intencion': 'Investigación Comercial',
        'funnel': 'MOFU',
        'tipo_respuesta': 'Lista de requisitos',
        'prompt_template': '¿Qué necesito para {tema}?',
        'cta_sugerido': 'Añadir: lista de requisitos, formación recomendada, cursos que te preparan',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    'quiero ser': {
        'regex': r'\bquiero\s+ser\b',
        'tipo_intencion': 'Investigación Comercial',
        'funnel': 'MOFU',
        'tipo_respuesta': 'Guía de carrera',
        'prompt_template': 'Quiero ser {tema}, ¿qué pasos debo seguir?',
        'cta_sugerido': 'Añadir: roadmap de carrera, formación paso a paso, historias de éxito, cursos recomendados',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    'cómo ser': {
        'regex': r'\bc[oó]mo\s+(?:ser|convertirse|llegar\s+a\s+ser)\b',
        'tipo_intencion': 'Investigación Comercial',
        'funnel': 'MOFU',
        'tipo_respuesta': 'Guía de carrera',
        'prompt_template': '¿Cómo puedo ser {tema}?',
        'cta_sugerido': 'Añadir: pasos claros, formación necesaria, tiempo estimado, inversión, cursos',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    'mejores': {
        'regex': r'\bmejor(?:es)?\s+(?:carreras?|cursos?|pa[ií]ses?|universidades?|opciones?)\b',
        'tipo_intencion': 'Investigación Comercial',
        'funnel': 'MOFU',
        'tipo_respuesta': 'Ranking/Comparativa',
        'prompt_template': '¿Cuáles son los mejores {tema}?',
        'cta_sugerido': 'Añadir: ranking actualizado con criterios claros, pros/contras, recomendación final',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    'diferencia entre': {
        'regex': r'\bdiferencias?\s+entre\b',
        'tipo_intencion': 'Investigación Comercial',
        'funnel': 'MOFU',
        'tipo_respuesta': 'Comparativa',
        'prompt_template': '¿Cuál es la diferencia entre {tema}?',
        'cta_sugerido': 'Añadir: tabla comparativa clara, cuándo elegir cada opción, recomendación',
        'score_base': {'intencion': 1, 'recomendable': 2}
    },
    'vs': {
        'regex': r'\bvs\.?\b|\bversus\b',
        'tipo_intencion': 'Investigación Comercial',
        'funnel': 'MOFU',
        'tipo_respuesta': 'Comparativa',
        'prompt_template': '¿Qué es mejor, {tema}?',
        'cta_sugerido': 'Añadir: tabla comparativa, veredicto final, para quién es cada opción',
        'score_base': {'intencion': 1, 'recomendable': 2}
    },
    'carreras': {
        'regex': r'\bcarreras?\s+(?:mejor|m[aá]s|cortas?|universitarias?|relacionadas?)\b',
        'tipo_intencion': 'Investigación Comercial',
        'funnel': 'MOFU',
        'tipo_respuesta': 'Lista con detalles',
        'prompt_template': '¿Cuáles son las {tema}?',
        'cta_sugerido': 'Añadir: duración, salario medio, demanda laboral, dónde estudiar',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    'qué estudiar para': {
        'regex': r'\bqu[ée]\s+estudiar\s+para\b',
        'tipo_intencion': 'Investigación Comercial',
        'funnel': 'MOFU',
        'tipo_respuesta': 'Guía formativa',
        'prompt_template': '¿Qué debo estudiar para {tema}?',
        'cta_sugerido': 'Añadir: itinerario formativo completo, opciones (FP, grado, máster), cursos online',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    'salidas profesionales': {
        'regex': r'\bsalidas?\s+profesionales?\b',
        'tipo_intencion': 'Investigación Comercial',
        'funnel': 'MOFU',
        'tipo_respuesta': 'Lista de opciones',
        'prompt_template': '¿Cuáles son las salidas profesionales de {tema}?',
        'cta_sugerido': 'Añadir: lista de puestos, salarios, sectores, formación complementaria recomendada',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    'cuánto gana': {
        'regex': r'\bcu[aá]nto\s+gana\b',
        'tipo_intencion': 'Investigación Comercial',
        'funnel': 'MOFU',
        'tipo_respuesta': 'Datos salariales',
        'prompt_template': '¿Cuánto gana un {tema}?',
        'cta_sugerido': 'Añadir: rangos salariales por experiencia/país, cómo aumentar salario, formación que sube el sueldo',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    'cuánto dura': {
        'regex': r'\bcu[aá]nto\s+(?:dura|tiempo|tarda)\b',
        'tipo_intencion': 'Investigación Comercial',
        'funnel': 'MOFU',
        'tipo_respuesta': 'Información práctica',
        'prompt_template': '¿Cuánto dura {tema}?',
        'cta_sugerido': 'Añadir: duración exacta, modalidades (intensivo, part-time), opciones más cortas',
        'score_base': {'intencion': 2, 'recomendable': 1}
    },
    'vale la pena': {
        'regex': r'\bvale\s+la\s+pena\b|\bmerece\s+la\s+pena\b',
        'tipo_intencion': 'Investigación Comercial',
        'funnel': 'MOFU',
        'tipo_respuesta': 'Análisis pros/contras',
        'prompt_template': '¿Vale la pena {tema}?',
        'cta_sugerido': 'Añadir: análisis honesto, ROI, testimonios reales, alternativas',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    
    # BOFU - Transaccional
    'curso de': {
        'regex': r'\bcursos?\s+(?:de|en|para|sobre)\b',
        'tipo_intencion': 'Transaccional - Curso',
        'funnel': 'BOFU',
        'tipo_respuesta': 'Recomendación directa',
        'prompt_template': '¿Qué curso de {tema} me recomiendas?',
        'cta_sugerido': 'Añadir: precio, duración, certificación, temario, opiniones alumnos, botón matrícula',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    'curso online': {
        'regex': r'\bcursos?\s+online\b',
        'tipo_intencion': 'Transaccional - Curso',
        'funnel': 'BOFU',
        'tipo_respuesta': 'Recomendación directa',
        'prompt_template': '¿Dónde puedo hacer un curso online de {tema}?',
        'cta_sugerido': 'Añadir: ventajas modalidad online, flexibilidad, plataforma, soporte, demo gratis',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    'curso gratis': {
        'regex': r'\bcursos?\s+gratis\b',
        'tipo_intencion': 'Transaccional - Curso',
        'funnel': 'BOFU',
        'tipo_respuesta': 'Lista de opciones',
        'prompt_template': '¿Hay cursos gratis de {tema}?',
        'cta_sugerido': 'Añadir: opciones gratuitas, qué incluyen, limitaciones, opción premium con más valor',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    'curso homologado': {
        'regex': r'\bcursos?\s+homologados?\b',
        'tipo_intencion': 'Transaccional - Curso',
        'funnel': 'BOFU',
        'tipo_respuesta': 'Recomendación con garantías',
        'prompt_template': '¿Dónde puedo hacer un curso homologado de {tema}?',
        'cta_sugerido': 'Añadir: entidad que homologa, validez, puntuación oposiciones, certificado oficial',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    'máster': {
        'regex': r'\bm[aá]ster(?:es)?\s+(?:de|en|online|universitario)?\b',
        'tipo_intencion': 'Transaccional - Máster',
        'funnel': 'BOFU',
        'tipo_respuesta': 'Recomendación directa',
        'prompt_template': '¿Qué máster de {tema} me recomiendas?',
        'cta_sugerido': 'Añadir: titulación, ECTS, precio, financiación, salidas profesionales, ranking',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    'formación': {
        'regex': r'\bformaci[oó]n\s+(?:en|de|online|profesional)\b',
        'tipo_intencion': 'Transaccional - Formación',
        'funnel': 'BOFU',
        'tipo_respuesta': 'Recomendación directa',
        'prompt_template': '¿Qué formación necesito en {tema}?',
        'cta_sugerido': 'Añadir: itinerario formativo, opciones según nivel, certificaciones, precios',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    'dónde estudiar': {
        'regex': r'\bd[oó]nde\s+(?:estudiar|hacer|sacar|puedo)\b',
        'tipo_intencion': 'Transaccional - Ubicación',
        'funnel': 'BOFU',
        'tipo_respuesta': 'Recomendación directa',
        'prompt_template': '¿Dónde puedo estudiar {tema}?',
        'cta_sugerido': 'Añadir: opciones presencial/online, comparativa centros, por qué elegirnos',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    'precio': {
        'regex': r'\bprecio(?:s)?\b|\bcu[áa]nto\s+cuesta\b|\bcoste\b',
        'tipo_intencion': 'Transaccional - Precio',
        'funnel': 'BOFU',
        'tipo_respuesta': 'Información de precio',
        'prompt_template': '¿Cuánto cuesta {tema}?',
        'cta_sugerido': 'Añadir: precio claro, opciones de financiación, descuentos, qué incluye, ROI',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    'opiniones': {
        'regex': r'\bopiniones?\b|\breseñas?\b|\bvaloraciones?\b',
        'tipo_intencion': 'Transaccional - Validación',
        'funnel': 'BOFU',
        'tipo_respuesta': 'Testimonios',
        'prompt_template': '¿Qué opiniones tiene {tema}?',
        'cta_sugerido': 'Añadir: testimonios reales con nombre, puntuación, casos de éxito verificables',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
}

# Umbrales de prioridad
PRIORITY_THRESHOLDS = {'CRÍTICA': 8, 'ALTA': 6, 'MEDIA': 4, 'BAJA': 0}

# ==============================================================================
# 🛠️ FUNCIONES AUXILIARES
# ==============================================================================

def extract_terms(text: str) -> list:
    """Extrae términos significativos eliminando stopwords."""
    stopwords = {
        'de', 'la', 'el', 'en', 'y', 'a', 'los', 'las', 'un', 'una',
        'que', 'es', 'por', 'con', 'para', 'del', 'al', 'como', 'se',
        'su', 'mas', 'pero', 'sus', 'le', 'ya', 'o', 'este', 'si',
        'porque', 'esta', 'cuando', 'muy', 'sin', 'sobre', 'ser', 'tiene',
        'tambien', 'fue', 'siendo', 'son', 'entre', 'todo', 'hacer',
        'qué', 'cómo', 'cuál', 'cuánto', 'dónde'
    }
    terms = re.findall(r'\b[a-záéíóúñü]+\b', str(text).lower())
    return [t for t in terms if t not in stopwords and len(t) > 2]

def clean_url_fragments(url):
    """Elimina fragmentos (#...) de las URLs."""
    if pd.isna(url):
        return url
    url_str = str(url)
    return url_str.split('#')[0] if '#' in url_str else url_str

def get_priority(score: float) -> str:
    """Asigna prioridad basada en score."""
    if score >= PRIORITY_THRESHOLDS['CRÍTICA']: return 'CRÍTICA'
    if score >= PRIORITY_THRESHOLDS['ALTA']: return 'ALTA'
    if score >= PRIORITY_THRESHOLDS['MEDIA']: return 'MEDIA'
    return 'BAJA'

def get_position_status(pos: float) -> str:
    """Clasifica el estado según posición en Google."""
    if pd.isna(pos) or pos == 0:
        return '⚪ Sin datos'
    elif pos <= 3:
        return '🟢 Top 3'
    elif pos <= 10:
        return '🟡 Top 10'
    elif pos <= 20:
        return '🟠 Striking Distance'
    else:
        return '🔴 Posición débil'

def extract_topic_from_query(query: str, pattern_name: str) -> str:
    """Extrae el tema de la query eliminando el patrón."""
    query_lower = str(query).lower()
    
    # Patrones a eliminar para extraer el tema
    remove_patterns = [
        r'^qu[ée]\s+es\s+(?:un[ao]?\s+|el\s+|la\s+)?',
        r'^qu[ée]\s+son\s+(?:los?\s+|las?\s+)?',
        r'^qu[ée]\s+estudia\s+(?:la\s+|el\s+)?',
        r'^c[oó]mo\s+(?:hacer|se\s+hace|elaborar|crear|funciona|ser)\s+(?:un[ao]?\s+|el\s+|la\s+)?',
        r'^para\s+qu[ée]\s+sirve\s+(?:la\s+|el\s+|un\s+)?',
        r'^requisitos\s+para\s+(?:ser\s+|trabajar\s+)?',
        r'^qu[ée]\s+(?:necesito|se\s+necesita)\s+para\s+',
        r'^quiero\s+ser\s+',
        r'^c[oó]mo\s+(?:ser|convertirse\s+en|llegar\s+a\s+ser)\s+',
        r'^mejores?\s+',
        r'^diferencias?\s+entre\s+',
        r'^carreras?\s+',
        r'^qu[ée]\s+estudiar\s+para\s+',
        r'^salidas?\s+profesionales?\s+(?:de\s+|del?\s+)?',
        r'^cu[aá]nto\s+(?:gana|dura|tiempo|cuesta)\s+(?:un[ao]?\s+|el\s+|la\s+)?',
        r'^cursos?\s+(?:de|en|para|sobre|online|gratis|homologados?)?\s*',
        r'^m[aá]ster(?:es)?\s+(?:de|en|online|universitario)?\s*',
        r'^formaci[oó]n\s+(?:en|de|online|profesional)?\s*',
        r'^d[oó]nde\s+(?:estudiar|hacer|sacar|puedo)\s+',
        r'^precio(?:s)?\s+(?:de|del?)?\s*',
        r'^opiniones?\s+(?:de|del?|sobre)?\s*',
        r'^ejemplos?\s+de\s+',
        r'^tipos?\s+de\s+',
    ]
    
    topic = query_lower
    for pattern in remove_patterns:
        topic = re.sub(pattern, '', topic, flags=re.IGNORECASE)
    
    return topic.strip()

def generate_llm_prompt(query: str, pattern_config: dict, topic: str) -> str:
    """Genera el prompt equivalente para LLM."""
    if pattern_config and 'prompt_template' in pattern_config:
        return pattern_config['prompt_template'].format(tema=topic)
    # Fallback: convertir query en pregunta natural
    return f"¿Puedes ayudarme con {query}?"

# ==============================================================================
# 📊 FUNCIONES DE PROCESAMIENTO
# ==============================================================================

def clean_ctr_column(series):
    """Limpia la columna CTR."""
    if series.dtype == 'object':
        cleaned = series.astype(str).str.replace('%', '', regex=False)
        cleaned = cleaned.str.replace(',', '.', regex=False).str.strip()
        cleaned = pd.to_numeric(cleaned, errors='coerce').fillna(0)
        if cleaned.max() < 1:
            cleaned = cleaned * 100
        return cleaned
    else:
        return series * 100 if series.max() < 1 else series

def normalize_columns(df):
    """Normaliza nombres de columnas."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    
    column_mapping = {
        'top queries': 'query', 'consultas principales': 'query',
        'top pages': 'page', 'páginas principales': 'page', 'paginas principales': 'page', 'url': 'page',
        'clics': 'clicks',
        'impressions': 'impresiones',
        'position': 'posicion', 'posición': 'posicion',
    }
    return df.rename(columns=column_mapping)

def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Detecta patrones y genera campos GEO."""
    df = df.copy()
    query_lower = df['query'].astype(str).str.lower()
    
    # Inicializar columnas
    df['Patrón'] = 'Otros'
    df['Tipo Intención'] = 'Indefinido'
    df['Funnel'] = 'Indefinido'
    df['Tipo Respuesta'] = 'General'
    df['Prompt LLM'] = df['query'].apply(lambda q: f"¿Puedes ayudarme con {q}?")
    df['CTA Sugerido'] = 'Añadir: información relevante, llamada a la acción clara'
    df['Score Intención'] = 0
    df['Score Recomendable'] = 1
    
    for pattern_name, config in PATTERNS.items():
        mask = query_lower.str.contains(config['regex'], regex=True, na=False) & (df['Patrón'] == 'Otros')
        
        if mask.any():
            df.loc[mask, 'Patrón'] = pattern_name
            df.loc[mask, 'Tipo Intención'] = config['tipo_intencion']
            df.loc[mask, 'Funnel'] = config['funnel']
            df.loc[mask, 'Tipo Respuesta'] = config['tipo_respuesta']
            df.loc[mask, 'CTA Sugerido'] = config['cta_sugerido']
            df.loc[mask, 'Score Intención'] = config['score_base']['intencion']
            df.loc[mask, 'Score Recomendable'] = config['score_base']['recomendable']
            
            # Generar Prompt LLM
            for idx in df[mask].index:
                topic = extract_topic_from_query(df.loc[idx, 'query'], pattern_name)
                df.loc[idx, 'Prompt LLM'] = generate_llm_prompt(df.loc[idx, 'query'], config, topic)
    
    return df

def calculate_scores(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Calcula scores GEO."""
    df = df.copy()
    
    # Score Competencia (basado en posición - striking distance es mejor)
    def position_score(pos):
        if pd.isna(pos) or pos == 0: return 1
        elif pos <= 3: return 1  # Ya estás arriba
        elif pos <= 20: return 2  # Striking distance - oportunidad
        elif pos <= 50: return 1
        else: return 0
    
    df['Score Competencia'] = df['posicion'].apply(position_score)
    
    # Score Autoridad (basado en CTR y clicks)
    ctr_median = df['ctr'].median()
    clicks_median = df['clicks'].median()
    
    def authority_score(row):
        if row['ctr'] > ctr_median and row['clicks'] > clicks_median: return 2
        elif row['ctr'] > ctr_median * 0.5 or row['clicks'] > clicks_median * 0.5: return 1
        return 0
    
    df['Score Autoridad'] = df.apply(authority_score, axis=1)
    
    # Score Total
    df['Score'] = (
        df['Score Intención'] * weights.get('intencion', 1.0) +
        df['Score Competencia'] * weights.get('competencia', 1.0) +
        df['Score Recomendable'] * weights.get('recomendable', 1.0) +
        df['Score Autoridad'] * weights.get('autoridad', 1.0)
    )
    
    # Normalizar a 0-10
    max_possible = 2 * sum(weights.values())
    df['Score'] = (df['Score'] / max_possible * 10).round(1).clip(0, 10)
    
    # Prioridad y Estado
    df['Prioridad'] = df['Score'].apply(get_priority)
    df['Estado Posición'] = df['posicion'].apply(get_position_status)
    
    return df

def generate_clusters(df: pd.DataFrame, min_frequency: int = 3) -> pd.DataFrame:
    """Genera clusters temáticos basados en n-grams."""
    # Extraer bigramas de queries prioritarias
    priority_queries = df[df['Prioridad'].isin(['CRÍTICA', 'ALTA'])]['query']
    
    bigrams = []
    for query in priority_queries:
        terms = extract_terms(str(query))
        if len(terms) >= 2:
            for i in range(len(terms) - 1):
                bigrams.append(f"{terms[i]} {terms[i+1]}")
    
    # Contar frecuencias
    bigram_counts = Counter(bigrams)
    
    # Filtrar por frecuencia mínima
    clusters = [(bigram, count) for bigram, count in bigram_counts.most_common(30) if count >= min_frequency]
    
    return pd.DataFrame(clusters, columns=['Cluster Temático', 'Frecuencia'])

def assign_clusters_to_queries(df: pd.DataFrame, clusters_df: pd.DataFrame) -> pd.DataFrame:
    """Asigna cluster temático a cada query."""
    df = df.copy()
    df['Cluster'] = 'Sin cluster'
    
    if clusters_df.empty:
        return df
    
    clusters = clusters_df['Cluster Temático'].tolist()
    
    for idx, row in df.iterrows():
        query_lower = str(row['query']).lower()
        for cluster in clusters:
            if cluster in query_lower:
                df.loc[idx, 'Cluster'] = cluster
                break
    
    return df

# ==============================================================================
# 📦 CARGA DE DATOS
# ==============================================================================

@st.cache_data(show_spinner="Cargando CSV...")
def load_data(file_content):
    """Carga y preprocesa el CSV."""
    if file_content is None:
        return None, "No se proporcionó archivo"
    
    try:
        try:
            df = pd.read_csv(file_content, encoding='utf-8', dtype={'query': 'str', 'page': 'str'}, low_memory=True)
        except:
            file_content.seek(0)
            df = pd.read_csv(file_content, encoding='latin-1', dtype={'query': 'str', 'page': 'str'}, low_memory=True)
        
        df = normalize_columns(df)
        
        required = ['page', 'query', 'clicks', 'impresiones', 'ctr', 'posicion']
        missing = [col for col in required if col not in df.columns]
        if missing:
            return None, f"Faltan columnas: {missing}. Disponibles: {list(df.columns)}"
        
        # Limpiar datos
        df['ctr'] = clean_ctr_column(df['ctr'])
        df['clicks'] = pd.to_numeric(df['clicks'], errors='coerce').fillna(0).astype(int)
        df['impresiones'] = pd.to_numeric(df['impresiones'], errors='coerce').fillna(0).astype(int)
        df['posicion'] = pd.to_numeric(df['posicion'], errors='coerce').fillna(0)
        df['page'] = df['page'].apply(clean_url_fragments)
        
        # Agrupar
        df_grouped = df.groupby(['page', 'query']).agg({
            'clicks': 'sum', 'impresiones': 'sum', 'ctr': 'mean', 'posicion': 'mean'
        }).reset_index()
        
        return df_grouped, None
        
    except Exception as e:
        return None, f"Error: {str(e)}"

def process_data(df_raw, weights, exclude_terms):
    """Procesa datos completos."""
    # Detectar patrones
    df = detect_patterns(df_raw)
    
    # Calcular scores
    df = calculate_scores(df, weights)
    
    # Generar clusters
    clusters_df = generate_clusters(df)
    
    # Asignar clusters a queries
    df = assign_clusters_to_queries(df, clusters_df)
    
    # Filtrar términos de marca
    if exclude_terms:
        terms = [t.strip().lower() for t in exclude_terms.split(',') if t.strip()]
        if terms:
            mask = df['query'].apply(lambda x: any(term in str(x).lower() for term in terms))
            df = df[~mask].copy()
    
    return df, clusters_df

# ==============================================================================
# 🎨 INTERFAZ
# ==============================================================================

# CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1e293b; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #64748b; margin-bottom: 2rem; }
    .metric-card { background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); padding: 1rem; border-radius: 0.5rem; color: white; }
    .stTabs [data-baseweb="tab-list"] { gap: 1rem; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🎯 GEO Scoring App v2</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Prioriza y optimiza contenido para aparecer en respuestas de LLMs</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📁 Cargar datos")
    uploaded_file = st.file_uploader("CSV de GSC (Page + Query)", type=['csv'])
    
    st.divider()
    
    st.header("⚙️ Configuración")
    with st.expander("Ajustar pesos del scoring"):
        peso_intencion = st.slider("Intención comercial", 0.0, 2.0, 1.0, 0.1)
        peso_competencia = st.slider("Oportunidad posición", 0.0, 2.0, 1.0, 0.1)
        peso_recomendable = st.slider("Tema recomendable LLM", 0.0, 2.0, 1.0, 0.1)
        peso_autoridad = st.slider("Autoridad actual", 0.0, 2.0, 1.0, 0.1)
    
    st.divider()
    
    exclude_terms = st.text_input("🧹 Excluir marca", placeholder="euroinnova, euro innova")
    
    st.divider()
    
    st.markdown("### ℹ️ Formato esperado")
    st.code("page, query, clicks, impressions, ctr, position", language=None)

# Estado de sesión
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'clusters_df' not in st.session_state:
    st.session_state.clusters_df = None

# Procesar datos
if uploaded_file is not None:
    with st.spinner("Procesando datos..."):
        df_raw, error = load_data(uploaded_file)
        
        if error:
            st.error(error)
            st.stop()
        
        weights = {
            'intencion': peso_intencion,
            'competencia': peso_competencia,
            'recomendable': peso_recomendable,
            'autoridad': peso_autoridad
        }
        
        df_processed, clusters_df = process_data(df_raw, weights, exclude_terms)
        st.session_state.df_processed = df_processed
        st.session_state.clusters_df = clusters_df

# ==============================================================================
# 📊 VISUALIZACIÓN
# ==============================================================================

if st.session_state.df_processed is not None:
    df = st.session_state.df_processed
    clusters_df = st.session_state.clusters_df
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "🎯 Oportunidades GEO", "🧩 Clusters", "📋 Acciones", "💾 Exportar"])
    
    # TAB 1: Dashboard
    with tab1:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Queries", f"{len(df):,}")
        col2.metric("🔴 Críticas", len(df[df['Prioridad'] == 'CRÍTICA']))
        col3.metric("🟠 Altas", len(df[df['Prioridad'] == 'ALTA']))
        col4.metric("🟡 Striking Distance", len(df[df['Estado Posición'] == '🟠 Striking Distance']))
        col5.metric("Total Clicks", f"{df['clicks'].sum():,}")
        
        st.divider()
        
        px, go = load_plotly()
        
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.markdown("### Distribución por Prioridad")
            priority_counts = df['Prioridad'].value_counts()
            fig = px.pie(values=priority_counts.values, names=priority_counts.index,
                        color=priority_counts.index,
                        color_discrete_map={'CRÍTICA': '#ef4444', 'ALTA': '#f97316', 'MEDIA': '#eab308', 'BAJA': '#22c55e'},
                        hole=0.4)
            fig.update_layout(margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)
        
        with col_c2:
            st.markdown("### Distribución por Funnel")
            funnel_counts = df['Funnel'].value_counts()
            fig = px.bar(x=funnel_counts.index, y=funnel_counts.values, 
                        color=funnel_counts.index,
                        color_discrete_map={'BOFU': '#22c55e', 'MOFU': '#f97316', 'TOFU': '#3b82f6', 'Indefinido': '#94a3b8'})
            fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🏆 Top 15 Oportunidades GEO")
        top_df = df.nlargest(15, 'Score')[['query', 'Prompt LLM', 'Prioridad', 'Score', 'Funnel', 'Estado Posición', 'clicks', 'page']]
        st.dataframe(top_df, use_container_width=True, hide_index=True,
                    column_config={
                        "query": "Query GSC",
                        "Prompt LLM": "Prompt equivalente LLM",
                        "Score": st.column_config.ProgressColumn("Score GEO", min_value=0, max_value=10),
                        "page": st.column_config.LinkColumn("URL a optimizar", display_text="Ver")
                    })
    
    # TAB 2: Oportunidades GEO
    with tab2:
        st.markdown("### 🎯 Tabla de Oportunidades GEO")
        
        col_f1, col_f2, col_f3, col_f4 = st.columns([1, 1, 1, 2])
        
        with col_f1:
            filter_priority = st.multiselect("Prioridad", ['CRÍTICA', 'ALTA', 'MEDIA', 'BAJA'], default=['CRÍTICA', 'ALTA'])
        with col_f2:
            filter_funnel = st.selectbox("Funnel", ['Todos', 'BOFU', 'MOFU', 'TOFU'])
        with col_f3:
            filter_tipo = st.selectbox("Tipo Respuesta", ['Todos'] + sorted(df['Tipo Respuesta'].unique().tolist()))
        with col_f4:
            search = st.text_input("🔍 Buscar query")
        
        df_filtered = df.copy()
        if filter_priority:
            df_filtered = df_filtered[df_filtered['Prioridad'].isin(filter_priority)]
        if filter_funnel != 'Todos':
            df_filtered = df_filtered[df_filtered['Funnel'] == filter_funnel]
        if filter_tipo != 'Todos':
            df_filtered = df_filtered[df_filtered['Tipo Respuesta'] == filter_tipo]
        if search:
            df_filtered = df_filtered[df_filtered['query'].str.contains(search, case=False, na=False)]
        
        st.caption(f"Mostrando {len(df_filtered):,} de {len(df):,} queries")
        
        display_cols = ['query', 'Prompt LLM', 'Tipo Respuesta', 'Score', 'Prioridad', 'Funnel', 'Estado Posición', 'CTA Sugerido', 'clicks', 'posicion', 'page']
        
        st.dataframe(
            df_filtered.sort_values('Score', ascending=False)[display_cols],
            use_container_width=True, height=500, hide_index=True,
            column_config={
                "query": "Query GSC",
                "Prompt LLM": "Prompt LLM",
                "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=10),
                "posicion": st.column_config.NumberColumn("Pos.", format="%.1f"),
                "page": st.column_config.LinkColumn("URL", display_text="Ver")
            }
        )
    
    # TAB 3: Clusters
    with tab3:
        st.markdown("### 🧩 Clusters Temáticos")
        st.markdown("Agrupa queries relacionadas para trabajar por bloques de contenido.")
        
        if clusters_df is not None and not clusters_df.empty:
            col_cl1, col_cl2 = st.columns([1, 2])
            
            with col_cl1:
                st.markdown("#### Temas detectados")
                st.dataframe(clusters_df, use_container_width=True, hide_index=True)
            
            with col_cl2:
                st.markdown("#### Queries por cluster")
                selected_cluster = st.selectbox("Selecciona un cluster", clusters_df['Cluster Temático'].tolist())
                
                cluster_queries = df[df['Cluster'] == selected_cluster].sort_values('Score', ascending=False)
                st.caption(f"{len(cluster_queries)} queries en este cluster")
                
                st.dataframe(
                    cluster_queries[['query', 'Prompt LLM', 'Score', 'Prioridad', 'clicks', 'page']].head(20),
                    use_container_width=True, hide_index=True,
                    column_config={
                        "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=10),
                        "page": st.column_config.LinkColumn("URL", display_text="Ver")
                    }
                )
        else:
            st.info("No hay suficientes queries prioritarias para detectar clusters (mínimo 3 repeticiones).")
    
    # TAB 4: Acciones
    with tab4:
        st.markdown("### 📋 Plan de Acción GEO")
        
        st.markdown("#### 🔴 Acciones Inmediatas (CRÍTICA + Striking Distance)")
        immediate = df[(df['Prioridad'] == 'CRÍTICA') & (df['Estado Posición'] == '🟠 Striking Distance')].sort_values('clicks', ascending=False)
        
        if len(immediate) > 0:
            for _, row in immediate.head(10).iterrows():
                with st.expander(f"**{row['query']}** — Score: {row['Score']} | {row['clicks']} clicks | Pos: {row['posicion']:.1f}"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"""
                        **Tipo de respuesta esperada:** {row['Tipo Respuesta']}
                        
                        **Prompt que usará el usuario en LLM:**
                        > {row['Prompt LLM']}
                        
                        **URL a optimizar:** [{row['page']}]({row['page']})
                        """)
                    with col_b:
                        st.markdown(f"""
                        **Qué añadir al contenido:**
                        
                        {row['CTA Sugerido']}
                        
                        **Cluster:** {row['Cluster']}
                        """)
        else:
            st.info("No hay queries CRÍTICAS en striking distance. ¡Buen trabajo!")
        
        st.divider()
        
        st.markdown("#### 🟠 Oportunidades BOFU (mayor conversión)")
        bofu = df[(df['Funnel'] == 'BOFU') & (df['Prioridad'].isin(['CRÍTICA', 'ALTA']))].sort_values('Score', ascending=False)
        
        if len(bofu) > 0:
            st.dataframe(
                bofu[['query', 'Prompt LLM', 'Tipo Respuesta', 'Score', 'Estado Posición', 'CTA Sugerido', 'page']].head(15),
                use_container_width=True, hide_index=True,
                column_config={"page": st.column_config.LinkColumn("URL", display_text="Ver")}
            )
        else:
            st.info("No hay queries BOFU prioritarias.")
    
    # TAB 5: Exportar
    with tab5:
        st.markdown("### 💾 Exportar Datos")
        
        col_e1, col_e2, col_e3 = st.columns(3)
        
        with col_e1:
            st.markdown("#### Completo")
            st.download_button("📥 CSV Completo", df.to_csv(index=False).encode('utf-8'), "geo_scoring_completo.csv", "text/csv")
        
        with col_e2:
            st.markdown("#### Prioritarios")
            priority_df = df[df['Score'] >= 6]
            st.download_button("📥 Score ≥ 6", priority_df.to_csv(index=False).encode('utf-8'), "geo_prioritarios.csv", "text/csv")
        
        with col_e3:
            st.markdown("#### Plan de Acción")
            action_df = df[(df['Prioridad'].isin(['CRÍTICA', 'ALTA'])) & (df['Estado Posición'] == '🟠 Striking Distance')]
            st.download_button("📥 Acciones Inmediatas", action_df.to_csv(index=False).encode('utf-8'), "geo_acciones.csv", "text/csv")
        
        st.divider()
        
        st.markdown("#### 📋 Resumen ejecutivo")
        resumen = f"""# Resumen GEO Scoring

**Fecha:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

## Métricas Generales
- Total queries: {len(df):,}
- CRÍTICA: {len(df[df['Prioridad'] == 'CRÍTICA'])}
- ALTA: {len(df[df['Prioridad'] == 'ALTA'])}
- Striking Distance: {len(df[df['Estado Posición'] == '🟠 Striking Distance'])}
- BOFU prioritarias: {len(df[(df['Funnel'] == 'BOFU') & (df['Prioridad'].isin(['CRÍTICA', 'ALTA']))])}

## Clusters principales
{clusters_df.head(10).to_string() if clusters_df is not None and not clusters_df.empty else 'No detectados'}

## Top 5 Acciones Inmediatas
"""
        for _, row in df.nlargest(5, 'Score').iterrows():
            resumen += f"\n- **{row['query']}** (Score: {row['Score']}) → {row['CTA Sugerido'][:50]}..."
        
        st.code(resumen, language="markdown")
        st.download_button("📥 Descargar Resumen", resumen, "geo_resumen.md", "text/markdown")

else:
    st.markdown("---")
    st.markdown("""
    ### 👈 Sube tu CSV de Google Search Console
    
    **El archivo debe contener:** `page, query, clicks, impressions, ctr, position`
    
    ---
    
    ### 🎯 ¿Qué hace esta app?
    
    | Función | Descripción |
    |---------|-------------|
    | **Score GEO** | Calcula probabilidad de que un LLM mencione tu contenido |
    | **Prompt LLM** | Transforma la query en cómo preguntaría un usuario a ChatGPT |
    | **Tipo Respuesta** | ¿Lista, tutorial, comparativa, recomendación? |
    | **CTA Sugerido** | Qué añadir al contenido para ser citado por LLMs |
    | **Clusters** | Agrupa queries por tema para trabajar en bloques |
    | **Plan de Acción** | Prioriza qué optimizar primero |
    """)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #94a3b8;'>🎯 GEO Scoring App v2.0 | Optimiza para LLMs</p>", unsafe_allow_html=True)
