from typing import TypedDict, List

from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tavily import TavilyClient
import os

# INITIALIZE LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# INITIALIZE MEMORY
memory = ChatMessageHistory()

# INITIALIZE TAVILY
os.environ["TAVILY_API_KEY"] = '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'
tavily_client = TavilyClient(
    api_key=os.environ["TAVILY_API_KEY"]
)


# VECTOR STORE SETUP
def create_vectorstore():

    pdf_files = [
        "dataset/THE_CENTRE_FOR_HUMANITARIAN_DATA.pdf",
        "dataset/Peer_Review_Framework_for_Predictive_Analytics_in_Humaniarian_Data.pdf"
    ]

    documents = []

    # Load PDFs
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        docs = loader.load()
        documents.extend(docs)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = splitter.split_documents(documents)

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Create FAISS vector DB
    vectorstore = FAISS.from_documents(
        split_docs,
        embeddings
    )

    return vectorstore


vectorstore = create_vectorstore()


# LANGGRAPH STATE

class AgentState(TypedDict):

    query: str

    route: str

    research_data: str

    final_response: str

    chat_history: List


# ROUTER AGENT

def router_agent(state: AgentState):

    query = state["query"]

    router_prompt = f"""
    You are a routing agent.

    Your job is to classify the user query into ONE category:

    Categories:
    - web
    - rag
    - llm

    Routing Rules:
    - Use 'web' for latest/current/recent/news/trending information
    - Use 'rag' for internal knowledge base questions
    - Use 'llm' for reasoning/general knowledge questions

    Query:
    {query}

    Return ONLY one word.
    """

    response = llm.invoke(router_prompt)

    route = response.content.strip().lower()

    print(f"\n[Router Decision]: {route}")

    return {
        "route": route
    }


# WEB RESEARCH AGENT

def web_research_agent(state: AgentState):

    query = state["query"]

    print("\n[Web Research Agent Activated]")

    response = tavily_client.search(
        query=query,
        search_depth="advanced",
        max_results=5
    )

    results = response["results"]

    formatted_results = []

    for result in results:

        formatted_results.append(
            f"""
Title:
{result.get("title")}

Content:
{result.get("content")}

URL:
{result.get("url")}
"""
        )

    combined_results = "\n\n".join(formatted_results)

    return {
        "research_data": combined_results
    }


# RAG AGENT

def rag_agent(state: AgentState):

    query = state["query"]

    print("\n[RAG Agent Activated]")

    # Better retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    )

    docs = retriever.invoke(query)

    # If nothing retrieved
    if not docs:

        return {
            "research_data": "NO_RELEVANT_RAG_DATA"
        }

    retrieved_context = "\n\n".join(
        [
            f"""
            Source: {doc.metadata.get('source')}
            
            Content:
            {doc.page_content}
            """
            for doc in docs
        ]
    )

    # LLM relevance verification
    relevance_check_prompt = f"""
    You are a strict relevance evaluator.

    User Question:
    {query}

    Retrieved Context:
    {retrieved_context}

    TASK:
    Determine whether the retrieved context contains
    enough relevant information to answer the question.

    RULES:
    - Return ONLY 'RELEVANT'
    - Return ONLY 'NOT_RELEVANT'
    - Be VERY strict
    - If context is weak, unrelated, partial,
      or insufficient -> return NOT_RELEVANT
    """

    relevance_response = llm.invoke(
        relevance_check_prompt
    )

    decision = relevance_response.content.strip()

    print(f"\nRelevance Decision: {decision}")

    if decision != "RELEVANT":

        return {
            "research_data": "NO_RELEVANT_RAG_DATA"
        }

    return {
        "research_data": retrieved_context
    }


# LLM REASONING AGENT

def llm_agent(state: AgentState):

    query = state["query"]

    print("\n[LLM Agent Activated]")

    response = llm.invoke(
        f"""
        Answer the following query clearly and accurately.

        Query:
        {query}
        """
    )

    return {
        "research_data": response.content
    }


# SUMMARIZATION AGENT

def summarization_agent(state: AgentState):

    query = state["query"]

    research_data = state["research_data"]

    chat_history = state.get("chat_history", [])

    print("\n[Summarization Agent Activated]")

    prompt = f"""
    You are a professional AI summarization assistant.

    User Query:
    {query}

    Chat History:
    {chat_history}

    Gathered Information:
    {research_data}

    Create a structured response with:

    1. Overview
    2. Detailed Explanation
    3. Key Takeaways
    4. Final Summary

    Ensure the response is:
    - Clear
    - Accurate
    - Well-structured
    - Professional
    """

    response = llm.invoke(prompt)

    return {
        "final_response": response.content
    }


# CONDITIONAL ROUTING FUNCTION

def route_decision(state: AgentState):

    return state["route"]


# BUILD LANGGRAPH
workflow = StateGraph(AgentState)


# ADD NODES
workflow.add_node(
    "router",
    router_agent
)

workflow.add_node(
    "web_research",
    web_research_agent
)

workflow.add_node(
    "rag",
    rag_agent
)

workflow.add_node(
    "llm",
    llm_agent
)

workflow.add_node(
    "summarizer",
    summarization_agent
)


# SET ENTRY POINT
workflow.set_entry_point("router")


# CONDITIONAL EDGES
workflow.add_conditional_edges(
    "router",
    route_decision,
    {
        "web": "web_research",
        "rag": "rag",
        "llm": "llm"
    }
)


# CONNECT TO SUMMARIZER
workflow.add_edge(
    "web_research",
    "summarizer"
)

workflow.add_edge(
    "rag",
    "summarizer"
)

workflow.add_edge(
    "llm",
    "summarizer"
)


# END GRAPH
workflow.add_edge(
    "summarizer",
    END
)


# COMPILE GRAPH
graph = workflow.compile()


# MAIN APPLICATION
def run_chatbot():

    print("\n" + "=" * 60)
    print("MULTI-AGENT RESEARCH SYSTEM")
    print("=" * 60)

    print("\nType 'exit' to quit.\n")

    while True:

        user_query = input("You: ")

        if user_query.lower() == "exit":
            print("\nGoodbye!")
            break

        # Load conversation history
        history = memory.messages

        # Invoke graph
        result = graph.invoke(
            {
                "query": user_query,
                "chat_history": history
            }
        )

        final_response = result["final_response"]

        # Save conversation
        memory.add_user_message(user_query)
        memory.add_ai_message(final_response)

        print("\n" + "=" * 60)
        print("ASSISTANT RESPONSE")
        print("=" * 60)

        print(f"\n{final_response}\n")


# RUN APPLICATION
if __name__ == "__main__":

    run_chatbot()
