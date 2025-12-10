import pandas as pd
import numpy as np

# Configuración de scoring por defecto
DEFAULT_WEIGHTS = {
    'intencion': 1.0,
    'contenido': 1.0,
    'competencia': 1.0,
    'recomendable': 1.0,
    'autoridad': 1.0
}

# Umbrales para clasificación de prioridad
PRIORITY_THRESHOLDS = {
    'CRÍTICA': 8,
    'ALTA': 6,
    'MEDIA': 4,
    'BAJA': 0
}


def calculate_score(df: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    """
    Calcula el score de probabilidad de mención en LLM para cada query.
    
    El score se basa en:
    - Intención comercial (0-2): ¿La query implica búsqueda de formación?
    - Contenido existente (0-2): ¿Hay URL que responda? (se calcula después con URL matching)
    - Competencia SERP (0-2): Basado en posición actual
    - Tema recomendable (0-2): ¿Los LLMs suelen recomendar formación aquí?
    - Autoridad temática (0-2): Basado en CTR y clicks
    
    Args:
        df: DataFrame con queries procesadas
        weights: Diccionario con pesos para cada criterio
    
    Returns:
        DataFrame con columnas de scoring añadidas
    """
    
    if weights is None:
        weights = DEFAULT_WEIGHTS
    
    df = df.copy()
    
    # Score de Intención Comercial (ya viene del pattern detector)
    if 'Score Intención' not in df.columns:
        df['Score Intención'] = 0
    
    # Score de Contenido Existente (placeholder - se actualiza con URL matching)
    if 'Score Contenido' not in df.columns:
        df['Score Contenido'] = 1  # Por defecto asumimos contenido parcial
    
    # Score de Competencia SERP (basado en posición)
    df['Score Competencia'] = calculate_competition_score(df['Posición'])
    
    # Score de Tema Recomendable (ya viene del pattern detector)
    if 'Score Recomendable' not in df.columns:
        df['Score Recomendable'] = 1
    
    # Score de Autoridad Temática (basado en CTR y clicks)
    df['Score Autoridad'] = calculate_authority_score(df)
    
    # Calcular score total ponderado
    df['Score'] = (
        df['Score Intención'] * weights.get('intencion', 1.0) +
        df['Score Contenido'] * weights.get('contenido', 1.0) +
        df['Score Competencia'] * weights.get('competencia', 1.0) +
        df['Score Recomendable'] * weights.get('recomendable', 1.0) +
        df['Score Autoridad'] * weights.get('autoridad', 1.0)
    )
    
    # Normalizar a escala 0-10
    max_possible = 2 * sum(weights.values())
    df['Score'] = (df['Score'] / max_possible * 10).round(1)
    
    # Asegurar que el score esté entre 0 y 10
    df['Score'] = df['Score'].clip(0, 10)
    
    return df


def calculate_competition_score(position: pd.Series) -> pd.Series:
    """
    Calcula score de competencia basado en la posición en SERP.
    
    - Posición 1-3: Alta competencia (score 0) - difícil competir
    - Posición 4-10: Media competencia (score 1)
    - Posición 11+: Baja competencia (score 2) - oportunidad
    
    Args:
        position: Serie con posiciones
    
    Returns:
        Serie con scores de competencia
    """
    
    def position_to_score(pos):
        if pd.isna(pos):
            return 1
        elif pos <= 3:
            return 0  # Alta competencia (ya estás arriba, difícil que LLM te mencione más)
        elif pos <= 10:
            return 1  # Media
        elif pos <= 20:
            return 2  # Oportunidad - visible pero mejorable
        else:
            return 1  # Muy abajo - menos probabilidad
    
    return position.apply(position_to_score)


def calculate_authority_score(df: pd.DataFrame) -> pd.Series:
    """
    Calcula score de autoridad temática basado en CTR y clicks.
    
    - CTR alto + clicks altos: Autoridad establecida (score 2)
    - CTR medio o clicks medios: Algo de autoridad (score 1)
    - CTR bajo y clicks bajos: Sin autoridad (score 0)
    
    Args:
        df: DataFrame con columnas CTR y Clicks
    
    Returns:
        Serie con scores de autoridad
    """
    
    # Calcular percentiles
    ctr_median = df['CTR'].median() if 'CTR' in df.columns else 2.0
    clicks_median = df['Clicks'].median() if 'Clicks' in df.columns else 100
    
    def calc_authority(row):
        ctr = row.get('CTR', 0) or 0
        clicks = row.get('Clicks', 0) or 0
        
        ctr_high = ctr > ctr_median * 1.5
        ctr_med = ctr > ctr_median * 0.5
        clicks_high = clicks > clicks_median * 2
        clicks_med = clicks > clicks_median * 0.5
        
        if ctr_high and clicks_high:
            return 2  # Alta autoridad
        elif ctr_high or clicks_high or (ctr_med and clicks_med):
            return 1  # Media autoridad
        else:
            return 0  # Baja autoridad
    
    return df.apply(calc_authority, axis=1)


def get_priority(score: float) -> str:
    """
    Asigna prioridad basada en el score.
    
    Args:
        score: Score de 0-10
    
    Returns:
        String con la prioridad
    """
    
    if score >= PRIORITY_THRESHOLDS['CRÍTICA']:
        return 'CRÍTICA'
    elif score >= PRIORITY_THRESHOLDS['ALTA']:
        return 'ALTA'
    elif score >= PRIORITY_THRESHOLDS['MEDIA']:
        return 'MEDIA'
    else:
        return 'BAJA'


def update_content_score(df: pd.DataFrame, has_url: pd.Series) -> pd.DataFrame:
    """
    Actualiza el score de contenido existente basado en si hay URL mapeada.
    
    Args:
        df: DataFrame con scores
        has_url: Serie booleana indicando si hay URL
    
    Returns:
        DataFrame con score de contenido actualizado
    """
    
    df = df.copy()
    
    # 0 = GAP (no hay URL), 1 = parcial, 2 = existe
    df.loc[has_url == False, 'Score Contenido'] = 0
    df.loc[has_url == True, 'Score Contenido'] = 2
    
    # Recalcular score total
    df['Score'] = (
        df['Score Intención'] +
        df['Score Contenido'] +
        df['Score Competencia'] +
        df['Score Recomendable'] +
        df['Score Autoridad']
    )
    
    # Normalizar
    df['Score'] = (df['Score'] / 10 * 10).round(1).clip(0, 10)
    
    return df


def get_score_breakdown(row: pd.Series) -> dict:
    """
    Genera un desglose del score para una query específica.
    
    Args:
        row: Fila del DataFrame
    
    Returns:
        Diccionario con el desglose
    """
    
    return {
        'query': row.get('Query', ''),
        'score_total': row.get('Score', 0),
        'prioridad': get_priority(row.get('Score', 0)),
        'desglose': {
            'Intención Comercial': {
                'score': row.get('Score Intención', 0),
                'max': 2,
                'descripcion': 'Nivel de intención de búsqueda de formación'
            },
            'Contenido Existente': {
                'score': row.get('Score Contenido', 0),
                'max': 2,
                'descripcion': 'Si existe URL que responda la query'
            },
            'Competencia SERP': {
                'score': row.get('Score Competencia', 0),
                'max': 2,
                'descripcion': 'Nivel de competencia en resultados de Google'
            },
            'Tema Recomendable': {
                'score': row.get('Score Recomendable', 0),
                'max': 2,
                'descripcion': 'Probabilidad de que LLMs recomienden formación'
            },
            'Autoridad Temática': {
                'score': row.get('Score Autoridad', 0),
                'max': 2,
                'descripcion': 'Nivel de autoridad basado en CTR y clicks'
            }
        }
    }
