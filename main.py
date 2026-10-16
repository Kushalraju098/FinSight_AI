
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from graph import graph
from input_guardrails import run_input_guardrails
from output_guardrails import run_output_guardrails


# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finsight_api")


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="FinSight AI",
    description="AI-powered financial research API",
    version="1.0.0",
)


# --------------------------------------------------
# Request and response schemas
# --------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
    )


class ChatResponse(BaseModel):
    answer: str


# --------------------------------------------------
# Health endpoints
# --------------------------------------------------

@app.get("/")
async def root():
    return {
        "message": "FinSight AI API is running",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


# --------------------------------------------------
# Chat endpoint
# --------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    try:
        # 1. Input guardrails
        logger.info("Running input guardrails")

        input_check = run_input_guardrails(question)

        if not input_check.get("allowed", False):
            logger.warning("Input guardrails rejected a request")

            raise HTTPException(
                status_code=400,
                detail="Request blocked by input guardrails.",
            )

        safe_question = input_check.get("sanitized_text")

        if not isinstance(safe_question, str) or not safe_question.strip():
            raise RuntimeError(
                "Input guardrails returned invalid sanitized text"
            )

        # 2. Execute the LangGraph workflow
        logger.info("Invoking FinSight AI graph")

        result = await graph.ainvoke({
            "question": safe_question,
            "route": "",
            "rag_context": [],
            "mcp_results": [],
            "final_response": "",
        })

        logger.info("FinSight AI graph completed")

        # 3. Validate graph output
        if not isinstance(result, dict):
            raise RuntimeError("Graph returned an invalid result")

        answer = result.get("final_response")

        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError(
                "Graph did not return a valid final_response"
            )

        # 4. Output guardrails
        logger.info("Running output guardrails")

        output_check = run_output_guardrails(
            response=answer,
            rag_context=result.get("rag_context", []),
            mcp_results=result.get("mcp_results", []),
        )
        logger.info("Output guardrail result: %r", output_check)

        if not output_check.get("allowed", False):
            logger.warning("Output guardrails rejected the response")

            raise HTTPException(
                status_code=422,
                detail="The generated response could not be verified.",
            )

        safe_answer = output_check.get("sanitized_response")

        if not isinstance(safe_answer, str) or not safe_answer.strip():
            raise RuntimeError(
                "Output guardrails returned an invalid response"
            )

        # 5. Return the answer
        return ChatResponse(answer=safe_answer)

    except HTTPException:
        raise

    except Exception:
        # Full traceback appears in the backend terminal.
        # Internal details are not returned to the client.
        logger.exception("FinSight AI /chat request failed")

        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing your request.",
        )
