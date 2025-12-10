import pandas as pd
import re

# Patrones de queries conversacionales
PATTERNS = {
    # Informacional - Definición
    'qué es': {
        'regex': r'\bqu[ée]\s+es\b',
        'tipo': 'Informacional - Definición',
        'funnel': 'TOFU',
        'score_base': {'intencion': 0, 'recomendable': 1}
    },
    'qué estudia': {
        'regex': r'\bqu[ée]\s+estudia\b',
        'tipo': 'Informacional - Definición',
        'funnel': 'TOFU',
        'score_base': {'intencion': 1, 'recomendable': 1}
    },
    'qué son': {
        'regex': r'\bqu[ée]\s+son\b',
        'tipo': 'Informacional - Definición',
        'funnel': 'TOFU',
        'score_base': {'intencion': 0, 'recomendable': 1}
    },
    'cuál es': {
        'regex': r'\bcu[aá]l\s+es\b',
        'tipo': 'Informacional - Definición',
        'funnel': 'TOFU',
        'score_base': {'intencion': 0, 'recomendable': 1}
    },
    
    # Informacional - Proceso
    'cómo': {
        'regex': r'\bc[oó]mo\s+(?:hacer|se\s+hace|elaborar|crear|funciona)\b',
        'tipo': 'Informacional - Proceso',
        'funnel': 'TOFU',
        'score_base': {'intencion': 1, 'recomendable': 2}
    },
    'pasos para': {
        'regex': r'\bpasos\s+para\b',
        'tipo': 'Informacional - Proceso',
        'funnel': 'TOFU',
        'score_base': {'intencion': 1, 'recomendable': 2}
    },
    
    # Informacional - Utilidad
    'para qué sirve': {
        'regex': r'\bpara\s+qu[ée]\s+sirve\b',
        'tipo': 'Informacional - Utilidad',
        'funnel': 'TOFU',
        'score_base': {'intencion': 1, 'recomendable': 1}
    },
    'por qué es importante': {
        'regex': r'\bpor\s+qu[ée]\s+es\s+importante\b',
        'tipo': 'Informacional - Utilidad',
        'funnel': 'TOFU',
        'score_base': {'intencion': 1, 'recomendable': 1}
    },
    
    # Investigación Comercial - Requisitos
    'requisitos para': {
        'regex': r'\brequisitos\s+para\b',
        'tipo': 'Investigación Comercial',
        'funnel': 'MOFU',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    'qué necesito para': {
        'regex': r'\bqu[ée]\s+necesito\s+para\b',
        'tipo': 'Investigación Comercial',
        'funnel': 'MOFU',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    'qué se necesita': {
        'regex': r'\bqu[ée]\s+se\s+necesita\b',
        'tipo': 'Investigación Comercial',
        'funnel': 'MOFU',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    
    # Investigación Comercial - Aspiracional
    'quiero ser': {
        'regex': r'\bquiero\s+ser\b',
        'tipo': 'Investigación Comercial',
        'funnel': 'MOFU',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    'cómo ser': {
        'regex': r'\bc[oó]mo\s+ser\b',
        'tipo': 'Investigación Comercial',
        'funnel': 'MOFU',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    'por qué quiero ser': {
        'regex': r'\bpor\s+qu[ée]\s+quiero\s+ser\b',
        'tipo': 'Investigación Comercial',
        'funnel': 'MOFU',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    
    # Investigación Comercial - Comparativas
    'mejor': {
        'regex': r'\bmejor(?:es)?\s+(?:carreras?|cursos?|pa[ií]ses?|universidades?)\b',
        'tipo': 'Investigación Comercial',
        'funnel': 'MOFU',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    'diferencia entre': {
        'regex': r'\bdiferencias?\s+entre\b',
        'tipo': 'Investigación Comercial',
        'funnel': 'MOFU',
        'score_base': {'intencion': 1, 'recomendable': 1}
    },
    'ventajas y desventajas': {
        'regex': r'\bventajas\s+y\s+desventajas\b',
        'tipo': 'Investigación Comercial',
        'funnel': 'MOFU',
        'score_base': {'intencion': 1, 'recomendable': 2}
    },
    
    # Investigación Comercial - Carreras
    'carreras': {
        'regex': r'\bcarreras?\s+(?:mejor|m[aá]s|cortas?|universitarias?|relacionadas?|de)\b',
        'tipo': 'Investigación Comercial',
        'funnel': 'MOFU',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    'qué estudiar para': {
        'regex': r'\bqu[ée]\s+estudiar\s+para\b',
        'tipo': 'Investigación Comercial',
        'funnel': 'MOFU',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    'salidas profesionales': {
        'regex': r'\bsalidas?\s+profesionales?\b',
        'tipo': 'Investigación Comercial',
        'funnel': 'MOFU',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    'cuánto gana': {
        'regex': r'\bcu[aá]nto\s+gana\b',
        'tipo': 'Investigación Comercial',
        'funnel': 'MOFU',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    'cuánto dura': {
        'regex': r'\bcu[aá]nto\s+(?:dura|tiempo)\b',
        'tipo': 'Investigación Comercial',
        'funnel': 'MOFU',
        'score_base': {'intencion': 2, 'recomendable': 1}
    },
    
    # Transaccional - Cursos
    'curso': {
        'regex': r'\bcursos?\s+(?:de|online|gratis|homologados?)\b',
        'tipo': 'Transaccional - Curso',
        'funnel': 'BOFU',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    'máster': {
        'regex': r'\bm[aá]ster(?:es)?\s+(?:de|en|online)\b',
        'tipo': 'Transaccional - Curso',
        'funnel': 'BOFU',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    'formación': {
        'regex': r'\bformaci[oó]n\s+(?:en|de|online|profesional)\b',
        'tipo': 'Transaccional - Curso',
        'funnel': 'BOFU',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    'certificado': {
        'regex': r'\b(?:curso|formaci[oó]n).*\bcertificado\b|\bcertificado\s+(?:de|en)\b',
        'tipo': 'Transaccional - Curso',
        'funnel': 'BOFU',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    
    # Transaccional - Ubicación
    'dónde estudiar': {
        'regex': r'\bd[oó]nde\s+(?:estudiar|hacer|sacar)\b',
        'tipo': 'Transaccional - Ubicación',
        'funnel': 'BOFU',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    'dónde puedo': {
        'regex': r'\bd[oó]nde\s+puedo\b',
        'tipo': 'Transaccional - Ubicación',
        'funnel': 'BOFU',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    
    # Otros patrones útiles
    'ejemplos de': {
        'regex': r'\bejemplos?\s+de\b',
        'tipo': 'Informacional - Ejemplos',
        'funnel': 'TOFU',
        'score_base': {'intencion': 0, 'recomendable': 1}
    },
    'tipos de': {
        'regex': r'\btipos?\s+de\b',
        'tipo': 'Informacional - Clasificación',
        'funnel': 'TOFU',
        'score_base': {'intencion': 0, 'recomendable': 1}
    },
    'características de': {
        'regex': r'\bcaracter[ií]sticas?\s+de\b',
        'tipo': 'Informacional - Características',
        'funnel': 'TOFU',
        'score_base': {'intencion': 0, 'recomendable': 1}
    },
    'funciones de': {
        'regex': r'\bfunciones?\s+de\b',
        'tipo': 'Informacional - Funciones',
        'funnel': 'TOFU',
        'score_base': {'intencion': 0, 'recomendable': 1}
    },
    'se puede': {
        'regex': r'\bse\s+puede\b',
        'tipo': 'Investigación Comercial',
        'funnel': 'MOFU',
        'score_base': {'intencion': 1, 'recomendable': 1}
    },
}


def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta patrones conversacionales en las queries y añade columnas de clasificación.
    
    Args:
        df: DataFrame con columna 'Query'
    
    Returns:
        DataFrame con columnas adicionales: Patrón, Tipo Intención, Fase Funnel, Score Base
    """
    
    if 'Query' not in df.columns:
        raise ValueError("El DataFrame debe tener una columna 'Query'")
    
    # Inicializar columnas
    df = df.copy()
    df['Patrón'] = 'Otros'
    df['Tipo Intención'] = 'Sin clasificar'
    df['Fase Funnel'] = 'N/A'
    df['Score Intención'] = 0
    df['Score Recomendable'] = 0
    
    # Detectar patrones
    for pattern_name, pattern_config in PATTERNS.items():
        regex = pattern_config['regex']
        mask = df['Query'].str.contains(regex, case=False, na=False, regex=True)
        
        # Solo actualizar si no tiene patrón asignado (priorizar primeros matches)
        update_mask = mask & (df['Patrón'] == 'Otros')
        
        df.loc[update_mask, 'Patrón'] = pattern_name
        df.loc[update_mask, 'Tipo Intención'] = pattern_config['tipo']
        df.loc[update_mask, 'Fase Funnel'] = pattern_config['funnel']
        df.loc[update_mask, 'Score Intención'] = pattern_config['score_base'].get('intencion', 0)
        df.loc[update_mask, 'Score Recomendable'] = pattern_config['score_base'].get('recomendable', 0)
    
    return df


def get_pattern_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera estadísticas agregadas por patrón.
    
    Args:
        df: DataFrame procesado con patrones
    
    Returns:
        DataFrame con estadísticas por patrón
    """
    
    stats = df.groupby('Patrón').agg({
        'Query': 'count',
        'Clicks': 'sum',
        'Impresiones': 'sum',
        'CTR': 'mean',
        'Posición': 'mean'
    }).round(2)
    
    stats.columns = ['Queries', 'Total Clicks', 'Total Impresiones', 'CTR Medio', 'Posición Media']
    stats = stats.sort_values('Total Clicks', ascending=False)
    
    return stats


def extract_topic(query: str) -> str:
    """
    Extrae el tema principal de una query eliminando el patrón.
    
    Args:
        query: String de la query
    
    Returns:
        Tema extraído
    """
    
    # Patrones a eliminar
    remove_patterns = [
        r'^qu[ée]\s+es\s+(?:un[ao]?\s+)?',
        r'^qu[ée]\s+son\s+(?:los?\s+|las?\s+)?',
        r'^qu[ée]\s+estudia\s+(?:la\s+)?',
        r'^c[oó]mo\s+(?:hacer|se\s+hace|elaborar)\s+(?:un[ao]?\s+)?',
        r'^para\s+qu[ée]\s+sirve\s+(?:la\s+|el\s+)?',
        r'^requisitos\s+para\s+(?:ser\s+)?',
        r'^d[oó]nde\s+(?:estudiar|puedo)\s+',
        r'^cursos?\s+(?:de\s+|en\s+)?',
    ]
    
    topic = query.lower()
    for pattern in remove_patterns:
        topic = re.sub(pattern, '', topic, flags=re.IGNORECASE)
    
    return topic.strip()
