# Proyecto Integrador 3: Sistema Multi-Agente RAG con Observabilidad

Este repositorio contiene la entrega final del Proyecto Integrador 3 para el módulo de AI Engineering. 
Implementa un sistema de soporte automatizado que clasifica la intención del usuario y enruta la consulta a uno de 3 agentes RAG especializados (HR, Tech, Finanzas). Todo el pipeline está instrumentado usando **Langfuse** para garantizar la observabilidad total, y evaluado mediante un *Golden Dataset*.

## Entregables Incluidos

- `multi_agent_system.ipynb`: Jupyter Notebook principal con la implementación estructurada paso a paso.
- `data/`: Contiene los documentos simulados (txt) separados por dominio (HR, Tech, Finance).
- `test_queries.json`: Golden Dataset con 12 consultas curadas, abarcando intenciones regulares y casos ambiguos/edge-cases.
- `.env.example`: Plantilla de variables de entorno necesarias.
- `requirements.txt`: Lista de dependencias del proyecto.

## Instalación y Configuración

1. **Clonar e Inicializar**
   Clona este repositorio y crea un entorno virtual de Python.
   ```bash
   uv venv
   # o bien: python -m venv venv
   ```

2. **Instalar Dependencias**
   ```bash
   uv pip install -r requirements.txt
   ```

3. **Configurar Variables de Entorno**
   Copia el archivo `.env.example` a `.env` y completa tus credenciales reales:
   - `OPENAI_API_KEY`: Tu clave de OpenAI (requerida para el modelo `gpt-4o-mini` y embeddings).
   - `LANGFUSE_PUBLIC_KEY` y `LANGFUSE_SECRET_KEY`: Credenciales del proyecto en Langfuse.
   - `LANGFUSE_HOST`: URL de Langfuse (usualmente `https://us.cloud.langfuse.com` o el host de EU).

## Ejecución del Proyecto

1. Abre el archivo `multi_agent_system.ipynb` en tu entorno Jupyter favorito (VS Code o JupyterLab).
2. Ejecuta las celdas en orden de arriba hacia abajo.
   - La **Sección 1** inicializará los clientes.
   - La **Sección 2** cargará los documentos de prueba desde `data/` y construirá la base de datos vectorial local con FAISS.
   - La **Sección 3** inicializará las cadenas RAG.
   - La **Sección 4** configura el Orquestador y el Enrutamiento condicional.
   - La **Sección 5** ejecutará iterativamente las preguntas de `test_queries.json` y enviará las métricas (Trace) a Langfuse.
   - La **Sección 6 (Bonus)** ejecuta el Evaluador que puntuará las respuestas del LLM automáticamente y guardará el score en Langfuse.

## Notas Técnicas
- **LangChain:** Se utiliza la API moderna de LangChain (`create_stuff_documents_chain`, `create_retrieval_chain`) por su robustez en producción.
- **Instrumentación:** Los decoradores `@observe` aseguran que el contexto y los identificadores de traza se propaguen correctamente hacia Langfuse, habilitando el análisis del *routing accuracy*.
