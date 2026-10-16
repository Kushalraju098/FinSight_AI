import asyncio
import json
import random
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from graph import graph

load_dotenv()

# The evaluator shares your Groq account's limits with calls made by graph.py.
evaluator = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=250,
)

DATASET_PATH = Path("evaluation_dataset.json")
OUTPUT_PATH = Path("evaluation_results.json")
MAX_EVIDENCE_CHARS = 6000
MAX_DOC_CHARS = 900
MAX_ANSWER_CHARS = 2500
MAX_RETRIES = 4


def load_dataset() -> list[dict[str, Any]]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH.resolve()}. "
            "Keep evaluation_dataset.json beside evaluate.py."
        )
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)
    if not isinstance(dataset, list):
        raise ValueError("evaluation_dataset.json must contain a JSON list.")
    required = {"question", "expected_answer", "expected_evidence"}
    for index, case in enumerate(dataset, start=1):
        missing = required - set(case)
        if missing:
            raise ValueError(f"Test case {index} is missing keys: {sorted(missing)}")
    return dataset


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            str(item.get("text", item)) if isinstance(item, dict) else str(item)
            for item in value
        )
    if isinstance(value, (dict, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


async def invoke_with_retry(callable_, label: str):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await callable_()
        except Exception as exc:
            message = str(exc).lower()
            retryable = any(token in message for token in (
                "429", "rate limit", "rate_limit", "too many requests",
                "temporarily unavailable", "503", "502", "500"
            ))
            if not retryable or attempt >= MAX_RETRIES:
                raise
            delay = min(60, 5 * (2 ** attempt)) + random.uniform(0, 1.5)
            print(f"{label}: temporary limit/server error; retrying in {delay:.1f}s "
                  f"({attempt + 1}/{MAX_RETRIES})")
            await asyncio.sleep(delay)


async def run_finsight(question: str) -> dict[str, Any]:
    result = await invoke_with_retry(
        lambda: graph.ainvoke({
            "question": question,
            "route": "",
            "rag_context": [],
            "mcp_results": [],
            "final_response": "",
        }),
        label="FinSight graph",
    )
    return {
        "answer": safe_text(result.get("final_response", "")),
        "rag_context": result.get("rag_context", []) or [],
        "mcp_results": result.get("mcp_results", []) or [],
    }


def format_evidence(result: dict[str, Any]) -> str:
    parts = []
    for index, doc in enumerate(result.get("rag_context", []), start=1):
        if isinstance(doc, dict):
            content = safe_text(doc.get("content", ""))[:MAX_DOC_CHARS]
            metadata = doc.get("metadata", {})
            source = ""
            if isinstance(metadata, dict):
                source = safe_text(metadata.get("source") or metadata.get("company")
                                   or metadata.get("filing") or "")
            parts.append(f"RAG evidence {index} {source}:\n{content}")
        else:
            parts.append(f"RAG evidence {index}:\n{safe_text(doc)[:MAX_DOC_CHARS]}")

    for index, item in enumerate(result.get("mcp_results", []), start=1):
        if isinstance(item, dict):
            parts.append(
                f"MCP evidence {index}:\nTool: {safe_text(item.get('tool', ''))}\n"
                f"Arguments: {safe_text(item.get('arguments', ''))}\n"
                f"Result: {safe_text(item.get('result', ''))[:1200]}"
            )
        else:
            parts.append(f"MCP evidence {index}:\n{safe_text(item)[:1200]}")
    return ("\n\n".join(parts) or "No evidence was returned.")[:MAX_EVIDENCE_CHARS]


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise TypeError("Evaluator response must be a JSON object.")
    return value


def normalize_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if 0 <= score <= 1 else None


async def evaluate_answer(test_case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    prompt = f"""Evaluate this financial QA answer using actual evidence. Penalize unsupported claims.
QUESTION: {safe_text(test_case.get("question"))[:1000]}
EXPECTED ANSWER: {safe_text(test_case.get("expected_answer"))[:1200]}
EXPECTED EVIDENCE: {safe_text(test_case.get("expected_evidence"))[:1200]}
ACTUAL EVIDENCE:
{format_evidence(result)}
GENERATED ANSWER:
{result["answer"][:MAX_ANSWER_CHARS]}

Score each 0.0–1.0:
faithfulness = claims supported by actual evidence
relevance = answers the question
correctness = agrees with expected answer when fixed; for live data, use actual MCP evidence
Return ONLY JSON: {{"faithfulness":0.0,"relevance":0.0,"correctness":0.0,"reason":"brief reason"}}
Keep reason under 25 words."""
    response = await invoke_with_retry(
        lambda: evaluator.ainvoke(prompt),
        label="Answer evaluator",
    )
    try:
        parsed = extract_json(safe_text(response.content))
        scores = {
            "faithfulness": normalize_score(parsed.get("faithfulness")),
            "relevance": normalize_score(parsed.get("relevance")),
            "correctness": normalize_score(parsed.get("correctness")),
            "reason": safe_text(parsed.get("reason", ""))[:500],
        }
        if any(scores[key] is None for key in ("faithfulness", "relevance", "correctness")):
            raise ValueError("A score is missing or outside the 0–1 range.")
        return scores
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return {
            "faithfulness": None, "relevance": None, "correctness": None,
            "reason": f"Evaluator response parsing failed: {exc}",
            "raw_evaluator_response": safe_text(response.content)[:1000],
        }


async def main() -> None:
    dataset = load_dataset()
    results = []
    print(f"Loaded {len(dataset)} test cases. Running sequentially to reduce rate-limit pressure.\n")

    for index, case in enumerate(dataset, start=1):
        question = safe_text(case["question"])
        print(f"[{index}/{len(dataset)}] {question}")
        row = {
            "test_id": case.get("id", index),
            "question": question,
            "expected_answer": case.get("expected_answer"),
        }
        try:
            result = await run_finsight(question)
            row["answer"] = result["answer"]
            row["rag_evidence_count"] = len(result["rag_context"])
            row["mcp_evidence_count"] = len(result["mcp_results"])
            await asyncio.sleep(2)
            row["evaluation"] = await evaluate_answer(case, result)
            evaluation = row["evaluation"]
            if evaluation["faithfulness"] is None:
                print(f"  Evaluation issue: {evaluation['reason']}")
            else:
                print("  Scores — " + ", ".join(
                    f"{metric}: {evaluation[metric]:.2f}"
                    for metric in ("faithfulness", "relevance", "correctness")
                ))
        except Exception as exc:
            row["answer"] = ""
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["evaluation"] = {
                "faithfulness": None, "relevance": None, "correctness": None,
                "reason": "Test failed before a complete evaluation.",
            }
            print(f"  FAILED: {row['error'][:500]}")

        results.append(row)
        # Save after each case so an interrupted run keeps completed results.
        with OUTPUT_PATH.open("w", encoding="utf-8") as file:
            json.dump(results, file, indent=2, ensure_ascii=False)
        if index < len(dataset):
            await asyncio.sleep(3)

    valid = [
        row["evaluation"] for row in results
        if all(row.get("evaluation", {}).get(key) is not None
               for key in ("faithfulness", "relevance", "correctness"))
    ]
    print("\n" + "=" * 60)
    print("FINSIGHT AI EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Test cases: {len(results)}")
    print(f"Successfully scored: {len(valid)}")
    for metric in ("faithfulness", "relevance", "correctness"):
        if valid:
            print(f"Average {metric}: {sum(item[metric] for item in valid) / len(valid):.3f}")
    print(f"Results saved to: {OUTPUT_PATH.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
