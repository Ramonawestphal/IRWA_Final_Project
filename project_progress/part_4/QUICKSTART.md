# 🚀 Quick Start Guide - Fashion Search Engine

## Paso 1: Preparar el Entorno

### 1.1 Estructura de Carpetas
Crea la siguiente estructura de carpetas:

```
tu_proyecto/
├── app.py
├── search_engine.py
├── analytics.py
├── config.py
├── requirements.txt
├── convert_parquet_to_json.py
├── test_search.py
├── data/
│   └── (aquí irá fashion_products_dataset.json)
└── templates/
    ├── base.html
    ├── index.html
    ├── results.html
    ├── product.html
    ├── analytics.html
    └── error.html
```

### 1.2 Instalar Dependencias

```bash
pip install -r requirements.txt
```

Esto instalará:
- Flask (web framework)
- pandas (data manipulation)
- numpy (numerical operations)
- scikit-learn (TF-IDF vectorization)
- sentence-transformers (semantic search)
- requests (HTTP requests for RAG)
- pyarrow (Parquet file support)

## Paso 2: Preparar los Datos

### Opción A: Si tienes `fashion_products_dataset.json`
Simplemente colócalo en la carpeta `data/`:
```
data/fashion_products_dataset.json
```

### Opción B: Si tienes `products_clean.parquet`
1. Coloca el archivo Parquet en `data/processed/`:
```
data/processed/products_clean.parquet
```

2. Ejecuta el script de conversión:
```bash
python convert_parquet_to_json.py
```

Esto creará automáticamente `data/fashion_products_dataset.json`

### Verificar el archivo JSON

El archivo debe tener esta estructura:
```json
[
  {
    "pid": "TKPFCZ9EA7H5FYZH",
    "title": "Solid Women Multicolor Track Pants",
    "brand": "York",
    "category": "Clothing and Accessories",
    "sub_category": "Bottomwear",
    "selling_price": "921",
    "actual_price": "2,999",
    "discount": "69% off",
    "average_rating": "3.9",
    "description": "Yorker trackpants made from 100% cotton...",
    "images": ["url1", "url2"],
    "product_details": [...],
    "url": "https://...",
    "seller": "Shyam Enterprises",
    "out_of_stock": false
  },
  ...
]
```

## Paso 3: Verificar la Instalación

Ejecuta el script de prueba:

```bash
python test_search.py
```

Deberías ver:
```
✓ All required packages installed
✓ Found: data/fashion_products_dataset.json
✓ Contains 28099 products
✓ Loaded 28099 products
✓ TFIDF: Found 5 results
✓ BM25: Found 5 results
✓ CUSTOM: Found 5 results
✓ SEMANTIC: Found 5 results
✅ Everything is ready! Start the app with: python app.py
```

## Paso 4: Iniciar la Aplicación

```bash
python app.py
```

Verás:
```
Loading search engine...
Building sentence embeddings...
Batches: 100%|████████| 878/878 [01:23<00:00, 10.51it/s]
Search engine ready with 28099 products
Loaded 28099 products
Starting Flask server...
 * Running on http://0.0.0.0:5000
```

**Nota**: La primera vez tardará 1-2 minutos en cargar los embeddings semánticos. Las siguientes veces serán más rápidas.

## Paso 5: Usar la Aplicación

### 5.1 Página Principal
Abre tu navegador en: **http://localhost:5000**

Verás:
- Caja de búsqueda central
- Selector de algoritmos
- Ejemplos de búsquedas
- Información sobre los algoritmos

### 5.2 Realizar una Búsqueda

1. **Escribe una consulta**, por ejemplo:
   - "women full sleeve sweatshirt cotton"
   - "men slim jeans blue"
   - "denim jacket"

2. **Selecciona un algoritmo**:
   - **Custom** (recomendado): Combina TF-IDF con features de producto
   - **TF-IDF**: Ranking basado en texto clásico
   - **BM25**: Ranking probabilístico
   - **Semantic**: Búsqueda semántica con IA

3. **Haz clic en "Search"**

### 5.3 Ver Resultados

La página de resultados muestra:
- **Resumen con IA** (arriba): Resumen generado automáticamente
- **Grid de productos**: Imagen, título, marca, precio, rating
- **Paginación**: 20 productos por página
- **Botón "View Details"**: Ver información completa del producto

### 5.4 Detalles del Producto

Al hacer clic en un producto verás:
- Imágenes del producto (con thumbnails)
- Precio, descuento, rating
- Descripción completa
- Especificaciones técnicas
- Productos similares
- Enlace al sitio original

### 5.5 Analytics Dashboard

Visita: **http://localhost:5000/analytics**

Verás estadísticas en tiempo real:
- Total de búsquedas y queries únicos
- Visualizaciones de productos
- Uso de algoritmos
- Top queries y productos
- Distribución de búsquedas por hora
- Actividad reciente

## Paso 6: API REST (Opcional)

### Búsqueda básica
```bash
curl "http://localhost:5000/api/search?q=blue+jeans"
```

### Con algoritmo específico
```bash
curl "http://localhost:5000/api/search?q=blue+jeans&algorithm=bm25"
```

### Limitar resultados
```bash
curl "http://localhost:5000/api/search?q=blue+jeans&top_k=10"
```

### Respuesta JSON
```json
{
  "query": "blue jeans",
  "algorithm": "custom",
  "total": 245,
  "results": [
    {
      "pid": "JEAF...",
      "title": "Slim Men Blue Jeans",
      "brand": "Levi's",
      "selling_price": "1979",
      "average_rating": "4.3",
      "score": 2.339,
      ...
    }
  ]
}
```

## Troubleshooting

### ❌ Error: "Module 'flask' not found"
```bash
pip install -r requirements.txt
```

### ❌ Error: "Data file not found"
Verifica que el archivo esté en: `data/fashion_products_dataset.json`

```bash
ls -la data/
```

### ❌ Error: "Cannot import name 'SearchEngine'"
Asegúrate de que todos los archivos Python están en la carpeta raíz del proyecto.

### ❌ La carga es muy lenta
La primera carga tarda 1-2 minutos construyendo embeddings. Esto es normal.

### ❌ Los resúmenes RAG no funcionan
El RAG usa una API gratuita que puede tener limitaciones. La búsqueda sigue funcionando sin resúmenes.

### ❌ Error de memoria (MemoryError)
Si tienes < 2GB RAM disponible:
1. Reduce el dataset
2. O desactiva semantic search en `config.py`

## Personalización

### Cambiar pesos del algoritmo Custom

Edita `search_engine.py`, línea ~235:

```python
final_score = (
    tfidf_score +
    0.5 * title_boost +      # Peso del título
    0.3 * price_score +      # Peso del precio
    0.3 * rating_score +     # Peso del rating
    0.2 * brand_score        # Peso de la marca
)
```

### Cambiar resultados por página

Edita `app.py`, línea 23:

```python
per_page = 20  # Cambia a 10, 30, 50, etc.
```

### Modificar colores y diseño

Edita los archivos en `templates/`. Todo el CSS está incluido en los templates.

## Queries de Prueba Recomendadas

Prueba estos queries para verificar que todo funciona:

1. `women full sleeve sweatshirt cotton` - Búsqueda específica
2. `men slim jeans blue` - Filtro por género y color
3. `denim jacket` - Búsqueda general
4. `cotton shirt man regular fit` - Múltiples atributos
5. `brand blend fabric` - Búsqueda por material
6. `high rating discount` - Búsqueda por características

## Estructura de Datos Analíticos

Los datos se guardan en `analytics_log.json`:

```json
{
  "searches": [
    {
      "timestamp": "2024-01-15T10:30:00",
      "query": "blue jeans",
      "algorithm": "custom",
      "type": "search"
    }
  ],
  "product_views": [
    {
      "timestamp": "2024-01-15T10:31:00",
      "pid": "JEAF...",
      "type": "product_view"
    }
  ]
}
```

## Siguiente Paso: Evaluación

Para evaluar tu motor de búsqueda:

1. Usa las 7 queries de validación del proyecto
2. Compara resultados entre algoritmos
3. Revisa el analytics dashboard para ver cuál es más usado
4. Documenta los mejores resultados para tu informe

## Recursos Adicionales

- **Logs de la aplicación**: Se muestran en la consola
- **Datos analíticos**: `analytics_log.json`
- **Configuración**: `config.py`
- **Documentación API**: README.md

## Contacto y Soporte

Si tienes problemas:
1. Revisa los mensajes de error en la consola
2. Ejecuta `python test_search.py` para diagnóstico
3. Verifica que todos los archivos están en su lugar
4. Comprueba que las dependencias están instaladas

---

**¡Listo! 🎉** Tu motor de búsqueda está funcionando con:
- ✅ 4 algoritmos de ranking
- ✅ Interfaz web completa
- ✅ RAG con resúmenes IA
- ✅ Analytics en tiempo real
- ✅ API REST
- ✅ 28,000+ productos fashion

**Disfruta buscando! 🛍️**