# agents.py
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_experimental.tools import PythonREPLTool
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from json_query import get_llm, get_qa_chain, get_vector_store, get_qdrant_client, get_embeddings

# RAG Tool
def rag_query_tool(query: str) -> str:
    """Query the document vector store for financial data."""
    client = get_qdrant_client()
    embeddings = get_embeddings()
    vector_store = get_vector_store(client, embeddings)
    qa_chain = get_qa_chain(get_llm(), vector_store)
    result = qa_chain({"query": query})
    return result.get("result", "No data found")

rag_tool = Tool(
    name="RAG_Query",
    func=rag_query_tool,
    description="Query ingested financial documents for raw data like revenues, expenses, liabilities."
)

# Python REPL Tool for calculations
python_repl_tool = PythonREPLTool()

# Analysis Agent: Performs financial calculations
analysis_prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    template="""
    You are a financial analysis agent. Use the provided tools to retrieve data and perform calculations.
    Available tools: {tools}
    Tool names: {tool_names}
    Tasks:
    - Calculate cash flow (revenue - expenses), burn rate (monthly expenses), runway (cash / burn rate).
    - Detect trends (e.g., MoM expense growth) and inefficiencies (e.g., vendor concentration).
    - Flag risks: negative margins, rising debt, liquidity issues, runway < 6 months.
    Use the Python REPL for computations and RAG for data retrieval.
    Input: {input}
    {agent_scratchpad}
    """
)

llm = get_llm()
analysis_agent = create_react_agent(llm, [rag_tool, python_repl_tool], analysis_prompt)
analysis_executor = AgentExecutor(agent=analysis_agent, tools=[rag_tool, python_repl_tool], verbose=True)

# Advisor Agent: Produces explanations and recommendations
# Since no tools are used, use a simpler prompt without ReAct
advisor_prompt = PromptTemplate(
    input_variables=["input", "analysis_output"],
    template="""
    You are a CFO advisor agent. Use the analysis results to provide clear, traceable explanations and recommendations.
    - Summarize in plain language.
    - Explain calculations (e.g., 'Runway is 5 months because cash is $250k and burn rate is $50k/month').
    - Flag risks (e.g., short runway, expense spikes) and suggest actions (e.g., 'Reduce vendor spend').
    - Link to data sources for traceability.
    Analysis results: {analysis_output}
    User query: {input}
    """
)

# Use LLM directly for Advisor Agent (no tools, no ReAct)
from langchain.chains import LLMChain
advisor_chain = LLMChain(llm=llm, prompt=advisor_prompt)
advisor_executor = AgentExecutor.from_agent_and_tools(
    agent=advisor_chain, tools=[], verbose=True
)

# LangGraph Workflow
class AgentState(TypedDict):
    input: str
    analysis_output: str
    final_output: str
    sources: List[dict]

def analysis_node(state: AgentState) -> AgentState:
    result = analysis_executor.invoke({"input": state["input"]})
    state["analysis_output"] = result["output"]
    # Extract sources from RAG
    client = get_qdrant_client()
    embeddings = get_embeddings()
    vector_store = get_vector_store(client, embeddings)
    results = vector_store.similarity_search_with_score(state["input"], k=3)
    state["sources"] = [
        {"page": doc.metadata.get("pages", "N/A"), "snippet": doc.page_content[:400]}
        for doc, score in results if score >= 0.37
    ]
    return state

def advisor_node(state: AgentState) -> AgentState:
    result = advisor_executor.invoke({"input": state["input"], "analysis_output": state["analysis_output"]})
    state["final_output"] = result["output"]
    return state

workflow = StateGraph(AgentState)
workflow.add_node("analysis", analysis_node)
workflow.add_node("advisor", advisor_node)
workflow.add_edge("analysis", "advisor")
workflow.add_edge("advisor", END)
workflow.set_entry_point("analysis")

multi_agent_chain = workflow.compile()