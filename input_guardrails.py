
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

# -----------------------------
# 1. Presidio PII protection
# -----------------------------
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()


def detect_and_redact_pii(text: str) -> dict:
    # Only analyze entities relevant to personal or sensitive information.
    # DATE_TIME is intentionally excluded because financial terms such
    # as "quarterly" can be incorrectly classified as dates.
    pii_entities = [
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "IP_ADDRESS",
        "CREDIT_CARD",
        "IBAN_CODE",
        "US_SSN",
        "US_PASSPORT",
        "US_DRIVER_LICENSE",
    ]

    results = analyzer.analyze(
        text=text,
        language="en",
        entities=pii_entities,
    )

    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
    )

    return {
        "original_text": text,
        "sanitized_text": anonymized.text,
        "pii_detected": len(results) > 0,
        "entities": list(
            set(result.entity_type for result in results)
        ),
    }




# -----------------------------
# 2. Load PIGuard
# -----------------------------
MODEL_NAME = "leolee99/PIGuard"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

injection_classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    truncation=True,
)


# -----------------------------
# 3. Resolve classifier labels
# -----------------------------
def resolve_label(label: str) -> str:
    """
    Convert the model's prediction into a recognized class.
    Raise an error rather than silently guessing if unknown.
    """
    normalized = label.strip().lower()

    if any(word in normalized for word in (
        "injection", "malicious", "attack"
    )):
        return "attack"

    if any(word in normalized for word in (
        "benign", "safe", "legitimate"
    )):
        return "benign"

    # Handle generic LABEL_0 / LABEL_1 outputs only when
    # the model config provides meaningful class names.
    if normalized.startswith("label_"):
        try:
            class_id = int(normalized.split("_")[-1])
            mapped = model.config.id2label.get(class_id)

            if mapped and mapped.lower() != normalized:
                return resolve_label(mapped)
        except (ValueError, AttributeError):
            pass

    raise ValueError(
        f"Unrecognized PIGuard label: {label!r}. "
        f"Model mapping: {model.config.id2label}"
    )


# -----------------------------
# 4. Detect prompt injection
# -----------------------------
def detect_prompt_injection(text: str) -> dict:
    result = injection_classifier(
        text,
        truncation=True,
        max_length=512,
    )[0]

    predicted_label = result["label"]
    confidence = float(result["score"])
    resolved = resolve_label(predicted_label)

    return {
        "injection_detected": resolved == "attack",
        "label": predicted_label,
        "resolved_label": resolved,
        "confidence": confidence,
    }


# -----------------------------
# 5. Combined input guardrails
# -----------------------------
def run_input_guardrails(text: str) -> dict:
    pii_result = detect_and_redact_pii(text)
    safe_text = pii_result["sanitized_text"]

    # Test PIGuard against the redacted text.
    injection_result = detect_prompt_injection(safe_text)

    if injection_result["injection_detected"]:
        return {
            "allowed": False,
            "reason": "Potential prompt injection detected",
            "sanitized_text": safe_text,
            "pii_detected": pii_result["pii_detected"],
            "injection": injection_result,
        }

    return {
        "allowed": True,
        "reason": None,
        "sanitized_text": safe_text,
        "pii_detected": pii_result["pii_detected"],
        "injection": injection_result,
    }
