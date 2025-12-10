import pandas as pd
import re
from difflib import SequenceMatcher
from urllib.parse import urlparse

def match_urls_to_queries(df_queries: pd.DataFrame, df_urls: pd.DataFrame) -> pd.DataFrame:
    """
    Mapea URLs existentes a queries usando fuzzy matching.
    
    Estrategias de matching:
    1. Coincidencia exacta de términos en URL
    2. Fuzzy matching del tema de la query con el slug de la URL
    3. Matching por categoría/vertical
    
    Args:
        df_queries: DataFrame con queries procesadas
        df_urls: DataFrame con URLs y métricas
    
    Returns:
        DataFrame de queries con columna 'URL Mapeada' añadida
    """
    
    df = df_queries.copy()
    
    if df_urls is None or len(df_urls) == 0:
        df['URL Mapeada'] = None
        return df
    
    # Asegurar que existe la columna URL
    url_col = None
    for col in ['URL', 'Page', 'Páginas principales', 'Top pages']:
        if col in df_urls.columns:
            url_col = col
            break
    
    if url_col is None:
        df['URL Mapeada'] = None
        return df
    
    # Crear índice de URLs para búsqueda eficiente
    url_index = create_url_index(df_urls[url_col].tolist())
    
    # Mapear cada query
    df['URL Mapeada'] = df['Query'].apply(lambda q: find_best_url_match(q, url_index))
    
    return df


def create_url_index(urls: list) -> dict:
    """
    Crea un índice de URLs para búsqueda eficiente.
    
    Args:
        urls: Lista de URLs
    
    Returns:
        Diccionario con índice de términos -> URLs
    """
    
    index = {
        'urls': [],
        'slugs': [],
        'terms': {}
    }
    
    for url in urls:
        if pd.isna(url):
            continue
            
        url = str(url)
        index['urls'].append(url)
        
        # Extraer slug (última parte de la URL)
        slug = extract_slug(url)
        index['slugs'].append(slug)
        
        # Indexar términos del slug
        terms = extract_terms(slug)
        for term in terms:
            if term not in index['terms']:
                index['terms'][term] = []
            index['terms'][term].append(len(index['urls']) - 1)
    
    return index


def extract_slug(url: str) -> str:
    """
    Extrae el slug de una URL (última parte significativa).
    
    Args:
        url: URL completa
    
    Returns:
        Slug limpio
    """
    
    try:
        parsed = urlparse(url)
        path = parsed.path.rstrip('/')
        
        # Obtener última parte del path
        if '/' in path:
            slug = path.split('/')[-1]
        else:
            slug = path
        
        # Limpiar extensiones y parámetros
        slug = re.sub(r'\.[a-z]+$', '', slug)
        slug = re.sub(r'\?.*$', '', slug)
        
        # Reemplazar guiones/underscores por espacios
        slug = re.sub(r'[-_]', ' ', slug)
        
        return slug.lower()
        
    except:
        return ''


def extract_terms(text: str) -> list:
    """
    Extrae términos significativos de un texto.
    
    Args:
        text: Texto a procesar
    
    Returns:
        Lista de términos
    """
    
    # Stopwords en español
    stopwords = {
        'de', 'la', 'el', 'en', 'y', 'a', 'los', 'las', 'un', 'una',
        'que', 'es', 'por', 'con', 'para', 'del', 'al', 'como', 'se',
        'su', 'mas', 'pero', 'sus', 'le', 'ya', 'o', 'este', 'si',
        'porque', 'esta', 'cuando', 'muy', 'sin', 'sobre', 'ser', 'tiene',
        'tambien', 'fue', 'siendo', 'son', 'entre', 'todo', 'hacer',
        'articulos', 'blog', 'www', 'com', 'https', 'http'
    }
    
    # Tokenizar
    terms = re.findall(r'\b[a-záéíóúñü]+\b', text.lower())
    
    # Filtrar stopwords y términos cortos
    terms = [t for t in terms if t not in stopwords and len(t) > 2]
    
    return terms


def find_best_url_match(query: str, url_index: dict, threshold: float = 0.4) -> str:
    """
    Encuentra la mejor URL que coincide con una query.
    
    Args:
        query: Query de búsqueda
        url_index: Índice de URLs
        threshold: Umbral mínimo de similitud
    
    Returns:
        URL que mejor coincide o None
    """
    
    if not url_index['urls']:
        return None
    
    query_terms = extract_terms(query)
    
    if not query_terms:
        return None
    
    # Estrategia 1: Buscar coincidencias exactas de términos
    candidate_scores = {}
    
    for term in query_terms:
        if term in url_index['terms']:
            for idx in url_index['terms'][term]:
                if idx not in candidate_scores:
                    candidate_scores[idx] = 0
                candidate_scores[idx] += 1
    
    # Estrategia 2: Fuzzy matching para los mejores candidatos
    best_match = None
    best_score = threshold
    
    # Si hay candidatos por términos, evaluar esos primero
    if candidate_scores:
        # Ordenar por número de términos coincidentes
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        
        for idx, term_count in sorted_candidates[:10]:  # Top 10 candidatos
            slug = url_index['slugs'][idx]
            
            # Calcular similitud
            similarity = calculate_similarity(query.lower(), slug)
            
            # Bonus por términos coincidentes
            similarity += term_count * 0.1
            
            if similarity > best_score:
                best_score = similarity
                best_match = url_index['urls'][idx]
    
    # Si no hay buenos candidatos, hacer búsqueda exhaustiva en top URLs
    if best_match is None:
        for idx, slug in enumerate(url_index['slugs'][:100]):  # Top 100 URLs
            similarity = calculate_similarity(query.lower(), slug)
            
            if similarity > best_score:
                best_score = similarity
                best_match = url_index['urls'][idx]
    
    return best_match


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calcula similitud entre dos textos usando múltiples métricas.
    
    Args:
        text1: Primer texto
        text2: Segundo texto
    
    Returns:
        Score de similitud (0-1)
    """
    
    # SequenceMatcher (similitud de secuencias)
    seq_ratio = SequenceMatcher(None, text1, text2).ratio()
    
    # Jaccard (términos en común)
    terms1 = set(extract_terms(text1))
    terms2 = set(extract_terms(text2))
    
    if terms1 and terms2:
        jaccard = len(terms1 & terms2) / len(terms1 | terms2)
    else:
        jaccard = 0
    
    # Combinar métricas
    return (seq_ratio * 0.4) + (jaccard * 0.6)


def get_url_coverage_stats(df: pd.DataFrame) -> dict:
    """
    Genera estadísticas de cobertura de URLs.
    
    Args:
        df: DataFrame con queries y URLs mapeadas
    
    Returns:
        Diccionario con estadísticas
    """
    
    total = len(df)
    with_url = df['URL Mapeada'].notna().sum()
    without_url = total - with_url
    
    return {
        'total_queries': total,
        'con_url': with_url,
        'sin_url': without_url,
        'cobertura_pct': round(with_url / total * 100, 1) if total > 0 else 0,
        'gaps': without_url
    }


def suggest_url_for_gap(query: str, existing_urls: list) -> dict:
    """
    Sugiere una estructura de URL para una query sin contenido.
    
    Args:
        query: Query sin URL
        existing_urls: Lista de URLs existentes para inferir patrón
    
    Returns:
        Diccionario con sugerencias
    """
    
    # Extraer términos clave
    terms = extract_terms(query)
    
    # Generar slug sugerido
    slug = '-'.join(terms[:5])  # Max 5 términos
    
    # Detectar patrón de URL más común
    base_path = '/blog/'
    
    if existing_urls:
        # Analizar patrones
        paths = [urlparse(u).path for u in existing_urls if pd.notna(u)]
        if paths:
            # Encontrar path base más común
            path_counts = {}
            for p in paths:
                parts = p.strip('/').split('/')
                if parts:
                    base = '/' + parts[0] + '/'
                    path_counts[base] = path_counts.get(base, 0) + 1
            
            if path_counts:
                base_path = max(path_counts, key=path_counts.get)
    
    return {
        'query': query,
        'slug_sugerido': slug,
        'url_sugerida': f'{base_path}{slug}',
        'terminos_clave': terms
    }
