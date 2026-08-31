# ---
import os
from dotenv import load_dotenv
from langfuse.decorators import observe, langfuse_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langfuse import Langfuse
import json
import logging
logging.getLogger("faiss").setLevel(logging.CRITICAL)

# Cargar variables de entorno
load_dotenv()

# Inicializar cliente de Langfuse para evaluacion (score)
langfuse = Langfuse()

# Inicializar LLM y Embeddings
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ---
def load_and_index_docs(domain, data_path):
    loader = DirectoryLoader(data_path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    docs = loader.load()
    
    # Text Splitter agresivo para asegurar que se generen bastantes chunks (req: >50 por dominio)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = text_splitter.split_documents(docs)
    print(f"[{domain}] Se generaron {len(chunks)} chunks.")
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever

hr_retriever = load_and_index_docs("HR", "data/hr_docs")
tech_retriever = load_and_index_docs("IT", "data/tech_docs")
finance_retriever = load_and_index_docs("FINANCE", "data/finance_docs")

# ---
def create_agent_chain(system_prompt, retriever):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt + "\n\nContexto:\n{context}"),
        ("human", "{input}")
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain

hr_chain = create_agent_chain("Eres el agente de Recursos Humanos. Responde basandote solo en las politicas dadas.", hr_retriever)
tech_chain = create_agent_chain("Eres el agente de Soporte IT. Responde basandote solo en el manual tecnico dado.", tech_retriever)
finance_chain = create_agent_chain("Eres el agente de Finanzas. Responde basandote solo en la politica de gastos dada.", finance_retriever)

# Funcion wraper instrumentada para cada agente
@observe(name="Agent-Execution")
def run_agent(intent, query):
    if intent == "hr":
        res = hr_chain.invoke({"input": query})
    elif intent == "tech":
        res = tech_chain.invoke({"input": query})
    elif intent == "finance":
        res = finance_chain.invoke({"input": query})
    else:
        return "Lo siento, no puedo procesar consultas sobre ese tema."
    return res["answer"]

# ---
@observe(name="SupportBotFlow")
def support_orchestrator(query: str, expected_intent: str = None):
    # Paso 1: Routing
    intent = classify_intent(query)
    
    # Paso 2: Ejecucion Condicional
    final_response = run_agent(intent, query)
    
    # Anadimos la respuesta final a la traza padre y los tags si venimos de la evaluacion
    if expected_intent:
        langfuse_context.update_current_trace(
            output=final_response,
            tags=["golden-run-v1", expected_intent],
            metadata={"expected_intent": expected_intent}
        )
    else:
        langfuse_context.update_current_trace(output=final_response)
        
    trace_id = langfuse_context.get_current_trace_id()
    return intent, final_response, trace_id

@observe(name="Orchestrator-Routing")
def classify_intent(query: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Clasifica la consulta en uno de estos dominios: 'hr', 'tech', 'finance', 'unknown'. "
                   "Ejemplo de HR: licencias, obra social. "
                   "Ejemplo de tech: VPN, equipos rotos, pantallas azules, robo de notebook, extravio. "
                   "Ejemplo de finance: reembolsos, sueldos, tarjetas.\n"
                   "Devuelve SOLO la palabra exacta."),
        ("human", "{input}")
    ])
    
    router_chain = prompt | llm
    res = router_chain.invoke({"input": query})
    
    intent = res.content.strip().lower()
    if intent not in ["hr", "tech", "finance"]:
         intent = "unknown"
         
    return intent

# ---
def run_evaluation():
    with open("test_queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
    
    correct_routes = 0
    
    for item in queries:
        query = item["query"]
        expected = item["expected_intent"]
        
        predicted_intent, response, trace_id = support_orchestrator(query, expected_intent=expected)
        
        # Flush para asegurar que el trace ID esta disponible en el backend
        langfuse_context.flush()
        
        if predicted_intent == expected:
            correct_routes += 1
            print(f"✅ Pass | Query: '{query}' -> {predicted_intent}")
        else:
            print(f"❌ Fail | Query: '{query}' -> Expected: {expected}, Got: {predicted_intent}")
            
        # Corremos el agente evaluador de Langfuse (Bonus)
        if trace_id:
            evaluate_response_quality(trace_id, query, response)
        
    print(f"\nResumen de Ruteo: {correct_routes}/{len(queries)} correctos.")

# ---
def evaluate_response_quality(trace_id, query, response):
    if not trace_id:
        return
        
    evaluator_prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un juez imparcial. Puntua la calidad de la respuesta a la pregunta del usuario en una escala del 1 al 10.\n"
                   "Devuelve SOLO el numero entero. Criterios: Relevancia y Precision."),
        ("human", "Pregunta: {query}\nRespuesta: {response}")
    ])
    
    eval_chain = evaluator_prompt | llm
    score_res = eval_chain.invoke({"query": query, "response": response})
    
    try:
        score_value = int(score_res.content.strip())
    except ValueError:
        score_value = 5 # Default on error
        
    # Enviar score a Langfuse
    langfuse.score(
        trace_id=trace_id,
        name="response-quality",
        value=score_value,
        comment="Evaluacion automatizada por LLM-as-a-judge"
    )

if __name__ == "__main__":
    print("\n=======================================================")
    print("¡Bienvenido al Bot de Soporte Multi-Agente (HR/IT/Finanzas)!")
    print("Escribe 'salir' para terminar.")
    print("Tambien puedes escribir 'evaluar' para correr el Golden Dataset.")
    print("=======================================================\n")
    while True:
        try:
            user_input = input("👤 Tu: ")
        except EOFError:
            break
        if user_input.lower() in ["salir", "exit", "quit"]:
            break
        if user_input.lower() == "evaluar":
            print("Corriendo evaluacion del Golden Dataset...")
            run_evaluation()
            continue
        
        intent, response, _ = support_orchestrator(user_input)
        langfuse_context.flush()
        print(f"🤖 Bot [enrutado a {intent.upper()}]: {response}\n")
