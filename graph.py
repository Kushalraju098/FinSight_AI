from typing import Any, TypedDict
from langgraph.graph import START, StateGraph, END
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from rag import rag_retriever, analyze_query
import asyncio
from langchain.mcp import MCPAdapter
from input_guardrails import run_input_guardrails
from output_guardrails import run_output_guardrails
import sys
from pathlib import Path

# STATE
class FinSightState(TypedDict):
    question: str
    route: str
    rag_context: list[Any]
    mcp_results: list[Any]
    final_response: str


# LLM
load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# MCP CLIENT
BASE_DIR = Path(__file__).resolve().parent

mcp_client = MCPAdapter(
    {
        "finsight": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(BASE_DIR / "mcp_server.py")],
        }
    }
)


# ROUTER NODE

from typing import Literal
from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    route: Literal["RAG", "MCP", "BOTH"] = Field(
        description=(
            "RAG for SEC filing evidence, MCP for live external data, "
            "or BOTH when the question requires both sources."
        )
    )


def router_node(state: FinSightState):
    question = state["question"]

    router_llm = llm.with_structured_output(RouteDecision)

    prompt = """
You are the routing component of FinSight AI, a financial research assistant.

Choose the data source based on the information needed to answer the question.

RAG:
Use RAG when the answer requires historical or reported information
contained in indexed SEC filings, including annual and quarterly reports,
revenue, earnings, financial statements, and company disclosures.

MCP:
Use MCP when the answer requires external tools for current or changing
information, including current stock prices, historical market prices,
and recent financial news.

BOTH:
Use BOTH only when answering the question requires evidence from
both SEC filings and external market tools.

Routing rules:
- Classify the user's intent, not individual keywords.
- A financial metric does not automatically mean RAG if the question
  specifically asks for a live external data source.
- A stock-related question does not automatically mean MCP if it only
  asks about information reported in a filing.
- If the question explicitly requires both sources, choose BOTH.
- Do not answer the user's question. Only classify it.

Examples:
Question: What was Apple's revenue in its 2025 annual report?
Route: RAG

Question: What is Apple's current stock price?
Route: MCP

Question: Compare Apple's reported annual revenue with its current stock price.
Route: BOTH

Question: What did Microsoft's latest quarterly filing report about operating income?
Route: RAG

Question: What are the latest financial headlines about Amazon?
Route: MCP
"""

    decision = router_llm.invoke(
        f"{prompt}\n\nUser question:\n{question}"
    )

    print(f"Router selected: {decision.route}")

    return {"route": decision.route}



# RAG NODE
def rag_node(state: FinSightState):

    print("\n" + "=" * 60)
    print("ENTERED RAG NODE")
    print("=" * 60)

    question = state["question"]

    try:
        search_query = analyze_query(question)

        print("\nRAG SEARCH QUERY:")
        print(search_query)

    except Exception as e:
        print(f"\nRAG query analysis failed: {e}")

        return {
            "rag_context": []
        }

    try:
        results = rag_retriever.retrieve(
            query=search_query.query,
            company=search_query.company,
            filing_year=search_query.filing_year,
            top_k=20
        )

        print("\nRAG RETRIEVED:", len(results))

        return {
            "rag_context": results
        }

    except Exception as e:
        print(f"\nRAG retrieval failed: {e}")

        return {
            "rag_context": []
        }

# MCP NODE
async def mcp_node(state: FinSightState):

    question = state["question"]

    # Get available tools from MCP server
    tools = await mcp_client.list_tools()

    # Give the LLM access to MCP tools
    model_with_tools = llm.bind_tools(tools)

    prompt = f"""
You are the market-data agent for FinSight AI.

Analyze the user's question and use the available MCP tools
to retrieve all information required to answer it.

Rules:
- You decide which tools to call.
- You may need to call multiple tools.
- Do not stop until all required information has been retrieved.
- Use the tool descriptions and schemas provided by MCP.
- Identify the correct stock ticker from the company name.
- Generate valid arguments for every tool call.
- Do not answer the user directly.
- Only make tool calls when information is required.

User question:
{question}
"""

    # Initial LLM request
    response = await model_with_tools.ainvoke(prompt)

    results = []

    # Continue the tool-calling loop until the LLM
    # decides that no more tools are required.
    while response.tool_calls:

        print("\nLLM Tool Calls:")
        print(response.tool_calls)

        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print("\nCalling MCP Tool:")
            print(tool_name)

            print("Arguments:")
            print(tool_args)

            # Find the MCP tool
            selected_tool = next(
                tool
                for tool in tools
                if tool.name == tool_name
            )

            try:
                # Execute MCP tool
                result = await selected_tool.ainvoke(tool_args)

                tool_result = {
                    "tool": tool_name,
                    "arguments": tool_args,
                    "result": result,
                    "error": None
                }

            except Exception as e:

                print(f"\nMCP tool failed: {tool_name}")
                print(f"Error: {e}")

                tool_result = {
                    "tool": tool_name,
                    "arguments": tool_args,
                    "result": None,
                    "error": str(e)
                }

            results.append(tool_result)

        # Give the LLM the results of the tools it just called
        follow_up_prompt = f"""
The following MCP tools have already been executed:

{results}

Original user question:
{question}

Review the information already retrieved.

If another MCP tool is required to fully answer the user's
question, call that tool.

If all required information has been retrieved, do not call
another tool.
"""

        response = await model_with_tools.ainvoke(follow_up_prompt)

    print("\nLLM Tool Calls:")
    print("No additional MCP tools requested.")

    return {
        "mcp_results": results
    }

# SYNTHESIS NODE
async def synthesis_node(state: FinSightState):

    question = state["question"]
    rag_context = state["rag_context"]
    mcp_results = state["mcp_results"]

    # --------------------------------------------------
    # 1. Prepare compact SEC / RAG context
    # --------------------------------------------------

    rag_text_parts = []

    for i, doc in enumerate(rag_context[:5], start=1):

        content = doc.get("content", "")

        metadata = doc.get("metadata", {})

        filing_type = metadata.get("filing_type", "")
        filing_year = metadata.get("filing_year", "")
        content_type = metadata.get("content_type", "")

        rag_text_parts.append(
            f"""
SEC Document {i}
Filing: {filing_type}
Filing year: {filing_year}
Content type: {content_type}

{content}
"""
        )

    rag_text = "\n".join(rag_text_parts)

    # --------------------------------------------------
    # 2. Prepare MCP context
    # --------------------------------------------------

    mcp_text_parts = []

    for result in mcp_results:

        tool_name = result.get("tool")
        tool_result = result.get("result")

        mcp_text_parts.append(
            f"""
MCP Tool: {tool_name}

Result:
{tool_result}
"""
        )

    mcp_text = "\n".join(mcp_text_parts)

    # --------------------------------------------------
    # 3. Final synthesis prompt
    # --------------------------------------------------

    prompt = f"""
You are the final answer generator for FinSight AI.

Answer the user's question using ONLY the evidence provided
in the SEC/RAG context and MCP market-data context.

User question:
{question}

==============================
SEC / RAG CONTEXT
==============================

{rag_text}

==============================
MCP MARKET DATA
==============================

{mcp_text}

==============================
INSTRUCTIONS
==============================

1. Use SEC/RAG context for information from company filings.
2. Use MCP data for live/current market information.
3. Clearly distinguish SEC filing information from live market data.
4. For news requests:
   - Identify the most relevant articles for the company.
   - Use both the headline and summary to determine relevance.
   - Briefly summarize why each selected article is relevant.
   - Paraphrase the article summary instead of copying it verbatim.
   - Ignore unrelated articles.
   - Do not invent details beyond the provided headline and summary.
5. Do NOT treat SEC filing information as current news.
6. Do NOT invent information that is not present in the evidence.
7. If requested information is missing, explicitly say that it
   was not available.
8. Give a concise, professional answer.
9. When comparing numbers, include the relevant values and
   percentages.
10. Do not mention internal agents, RAG, Pinecone, MCP,
    prompts, or tool-calling details.

Return only the final answer to the user.
"""

    # --------------------------------------------------
    # 4. Generate final response
    # --------------------------------------------------

    response = await llm.ainvoke(prompt)

    return {
        "final_response": response.content
    }

# ROUTING FUNCTION
def choose_route(state: FinSightState):

    route = state["route"]

    if route == "RAG":
        return "RAG"

    elif route == "MCP":
        return "MCP"

    elif route == "BOTH":
        return ["RAG", "MCP"]


# GRAPH INITIALIZATION
builder = StateGraph(FinSightState)


# NODES
builder.add_node("router", router_node)
builder.add_node("rag", rag_node)
builder.add_node("mcp", mcp_node)
builder.add_node("synthesis", synthesis_node)


# EDGES
builder.add_edge(START, "router")

builder.add_conditional_edges(
    "router",
    choose_route,
    {
        "RAG": "rag",
        "MCP": "mcp",
    }
)

builder.add_edge("rag", "synthesis")
builder.add_edge("mcp", "synthesis")

builder.add_edge("synthesis", END)


# COMPILE
graph = builder.compile()


async def main():
    #question = input("Ask FinSight: ")
    question="What is the revenue of NVIDIA in 2025"

    # Step 1: Run input security guardrails
    guardrail_result = run_input_guardrails(question)

    if not guardrail_result["allowed"]:
        print("\nRequest blocked by input security guardrails.")
        print("Reason:", guardrail_result["reason"])
        return

    # Step 2: Use the sanitized question
    safe_question = guardrail_result["sanitized_text"]


    result = await graph.ainvoke({
        "question": safe_question,
        "route": "",
        "rag_context": [],
        "mcp_results": [],
        "final_response": ""
    })

    if output_check["pii_detected"]:
        print("\nWarning: PII detected and redacted.")

    if not output_check["allowed"]:
        print("\nWarning: The response could not be verified.")
        print(output_check["grounding"]["explanation"])
        print("Unsupported claims:",
              output_check["grounding"]["unsupported_claims"])
        return

    print("\nFinSight Answer:")
    print(output_check["sanitized_response"])



if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
