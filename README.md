# 🎯 GEO Scoring App

Aplicación para analizar queries de Google Search Console y priorizar contenido para **Generative Engine Optimization (GEO)**.

## 🚀 Demo

[Ver app en Streamlit Cloud](https://geo-scoring-app.streamlit.app) *(actualizar con tu URL)*

## 📋 Características

- **Detección automática de patrones**: Identifica queries informacionales, transaccionales y de investigación comercial
- **Sistema de scoring**: Calcula probabilidad de mención en LLMs (0-10)
- **Mapeo de URLs**: Cruza queries con contenido existente
- **Identificación de GAPs**: Detecta oportunidades de contenido
- **Dashboard interactivo**: Visualiza métricas y filtra datos
- **Exportación**: CSV, Excel y resumen para Notion

## 🎯 Sistema de Scoring

Cada query recibe una puntuación basada en:

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| Intención comercial | 0-2 | ¿Busca formación? |
| Contenido existente | 0-2 | ¿Hay URL que responda? |
| Competencia SERP | 0-2 | ¿Baja competencia? |
| Tema recomendable | 0-2 | ¿LLMs recomiendan aquí? |
| Autoridad temática | 0-2 | ¿Eres referente? |

**Prioridades:**
- 🔴 CRÍTICA (8-10): Optimizar urgente
- 🟠 ALTA (6-7): Priorizar
- 🟡 MEDIA (4-5): Oportunidad moderada
- 🟢 BAJA (0-3): Baja probabilidad

## 📦 Instalación Local

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/geo-scoring-app.git
cd geo-scoring-app

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o: venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar
streamlit run app.py
```

## ☁️ Despliegue en Streamlit Cloud

1. **Sube el código a GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/tu-usuario/geo-scoring-app.git
   git push -u origin main
   ```

2. **Conecta en Streamlit Cloud**
   - Ve a [share.streamlit.io](https://share.streamlit.io)
   - Clic en "New app"
   - Selecciona tu repositorio
   - Branch: `main`
   - Main file: `app.py`
   - Clic en "Deploy"

3. **¡Listo!** Tu app estará en `https://tu-usuario-geo-scoring-app.streamlit.app`

## 📊 Cómo Usar

### 1. Exportar datos de GSC

Desde Google Search Console:
- Ve a **Rendimiento**
- Selecciona rango de fechas (recomendado: 12-18 meses)
- Pestaña **Consultas** → Exportar CSV
- Pestaña **Páginas** → Exportar CSV

### 2. Cargar en la app

- Sube el CSV de consultas (obligatorio)
- Sube el CSV de páginas (opcional, para mapeo de URLs)

### 3. Analizar

- **Dashboard**: Métricas generales y distribución
- **Scoring**: Tabla filtrable con todas las queries
- **GAPs**: Queries sin contenido (oportunidades)
- **Análisis**: Gráficos detallados

### 4. Exportar

- Descarga CSV filtrado
- Genera resumen para documentación

## 🔧 Configuración Avanzada

### Ajustar pesos del scoring

En la barra lateral, expande "Ajustar pesos" para modificar la importancia de cada criterio.

### Añadir patrones personalizados

Edita `utils/pattern_detector.py` y añade nuevos patrones al diccionario `PATTERNS`:

```python
PATTERNS = {
    'mi_patron': {
        'regex': r'\bmi\s+patrón\b',
        'tipo': 'Mi Tipo',
        'funnel': 'MOFU',
        'score_base': {'intencion': 2, 'recomendable': 2}
    },
    # ...
}
```

## 📁 Estructura del Proyecto

```
geo-scoring-app/
├── app.py                    # Aplicación principal
├── requirements.txt          # Dependencias
├── README.md                 # Este archivo
└── utils/
    ├── __init__.py
    ├── pattern_detector.py   # Detección de patrones
    ├── scoring.py            # Cálculo de scores
    └── url_matcher.py        # Mapeo de URLs
```

## 🤝 Contribuir

1. Fork del repositorio
2. Crea una rama (`git checkout -b feature/nueva-funcionalidad`)
3. Commit (`git commit -am 'Añade nueva funcionalidad'`)
4. Push (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 📄 Licencia

MIT License - Libre para uso comercial y personal.

## 👤 Autor

Desarrollado para optimización de contenido educativo.

---

**¿Preguntas?** Abre un issue en GitHub.
