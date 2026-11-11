import json
import math
import os
from pathlib import Path
from urllib.error import URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import joblib
from flask import Flask, jsonify, render_template, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from knowledge_base import retrieve_retention_protocols


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
INDUSTRIES = {"ecommerce"}
MAX_CHAT_MESSAGE = 1200


def load_env_file():
    env_file = ROOT / ".env"
    if env_file.is_file():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip("'\"")
                        if key and key not in os.environ:
                            os.environ[key] = value
        except Exception:
            pass


load_env_file()

if os.environ.get("GEMINI_API_KEY"):
    print("[RetainX] GEMINI_API_KEY is detected.")
else:
    print("[RetainX] Note: GEMINI_API_KEY is not set. Retention Advisor will run in Preview Mode.")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024
limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri=os.environ.get("RATELIMIT_STORAGE_URI", "memory://"),
    default_limits=[],
)


def load_model(industry):
    model_path = MODEL_DIR / f"{industry}.joblib"
    if not model_path.is_file():
        return None
    return joblib.load(model_path)


MODELS = {industry: load_model(industry) for industry in INDUSTRIES}
REVIEW_MODEL = load_model("reviews")


class AIProviderError(RuntimeError):
    pass


def error_response(message, status):
    return jsonify({"error": message}), status


def clean_number(value, name, minimum, maximum, integer=False):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number.")
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    if number < minimum or number > maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}.")
    if integer and not number.is_integer():
        raise ValueError(f"{name} must be a whole number.")
    return int(number) if integer else number


def normalize_customer(industry, customer):
    if not isinstance(customer, dict):
        raise ValueError("Provide customer details.")
    if industry != "ecommerce":
        raise ValueError("Invalid industry. Only 'ecommerce' is supported.")
    recency = clean_number(customer.get("days_since_last_order"), "Days since last order", 0, 3650, True)
    order_count = clean_number(customer.get("orders_in_history", 1), "Orders in history", 1, 500, True)
    tenure = clean_number(customer.get("days_since_first_order", recency), "Customer history days", recency, 3650, True)
    average_review = clean_number(customer.get("average_review_score", 0), "Average review score", 0, 5)
    last_review = clean_number(customer.get("last_review_score", 0), "Last review score", 0, 5)
    avg_delay = clean_number(customer.get("average_delivery_delay_days", 0), "Average delivery delay", -365, 365)
    avg_duration = clean_number(customer.get("average_delivery_duration_days", 0), "Average delivery duration", 0, 365)
    average_value = clean_number(customer.get("average_order_value", 0), "Average order value", 0, 100000)
    spend = clean_number(customer.get("total_historical_spend", 0), "Historical spend", 0, 10000000)
    items = clean_number(customer.get("average_items_per_order", 0), "Average items per order", 0, 500)
    last_value = clean_number(customer.get("last_order_value", 0), "Last order value", 0, 100000)
    last_freight = clean_number(customer.get("last_freight_value", 0), "Last freight value", 0, 50000)
    installments = clean_number(customer.get("last_payment_installments", 1), "Last payment installments", 1, 36, True)
    ptype = customer.get("last_payment_type", "unknown")
    if ptype not in {"credit_card", "boleto", "voucher", "debit_card", "unknown"}:
        raise ValueError("Choose a valid payment method.")
    return {
        "days_since_last_order": recency,
        "orders_in_history": order_count,
        "days_since_first_order": tenure,
        "average_order_value": average_value,
        "total_historical_spend": spend,
        "average_items_per_order": items,
        "last_order_value": last_value,
        "last_freight_value": last_freight,
        "last_payment_installments": installments,
        "last_payment_type": ptype,
        "average_review_score": average_review,
        "last_review_score": last_review,
        "average_delivery_delay_days": avg_delay,
        "average_delivery_duration_days": avg_duration,
    }


def analyze_review(text):
    if text in (None, ""):
        return None
    if not isinstance(text, str):
        raise ValueError("Review text must be plain text.")
    if len(text) > 2500:
        raise ValueError("Review text must be 2,500 characters or fewer.")
    text = text.strip()
    if not text:
        return None
    if REVIEW_MODEL is None:
        return {
            "sentiment": "unavailable",
            "message": "The review sentiment model has not been installed.",
        }
    vectorized = REVIEW_MODEL["vectorizer"].transform([text])
    positive_score = float(REVIEW_MODEL["classifier"].predict_proba(vectorized)[0][1])
    sentiment = "positive" if positive_score >= 0.6 else "negative" if positive_score <= 0.4 else "mixed"
    return {
        "sentiment": sentiment,
        "positive_score_percent": round(positive_score * 100, 1),
        "label_basis": "NLP Model trained on real Olist customer review messages.",
        "limitations": REVIEW_MODEL.get("limitations", "Trained on e-commerce customer review text."),
    }


def make_evidence(model_bundle, features):
    vectorized = model_bundle["vectorizer"].transform([features])
    scaled = model_bundle["scaler"].transform(vectorized)
    coef = model_bundle["classifier"].coef_[0]
    
    if hasattr(scaled, "toarray"):
        scaled_row = scaled.toarray()[0]
    elif hasattr(scaled, "toarray") or isinstance(scaled, (list, tuple)):
        scaled_row = list(scaled)[0]
    else:
        scaled_row = scaled[0]
        
    contributions = scaled_row * coef
    names = model_bundle["vectorizer"].get_feature_names_out()
    evidence = []
    for name, value in zip(names, contributions):
        if abs(value) < 0.01:
            continue
        evidence.append({
            "feature": str(name),
            "direction": "raises" if value > 0 else "lowers",
            "strength": round(float(abs(value)), 2),
        })
    evidence.sort(key=lambda item: item["strength"], reverse=True)
    return evidence[:5]


def model_case(industry, customer):
    bundle = MODELS.get(industry)
    if bundle is None:
        raise RuntimeError(f"The {industry} model has not been trained or installed.")
    features = normalize_customer(industry, customer)
    vectorized = bundle["vectorizer"].transform([features])
    scaled = bundle["scaler"].transform(vectorized)
    probability = float(bundle["classifier"].predict_proba(scaled)[0][1])
    threshold = float(bundle["decision_threshold"])
    evidence = make_evidence(bundle, features)
    review_text = customer.get("review_text") if isinstance(customer, dict) else None
    
    case = {
        "industry": industry,
        "probability": round(probability, 4),
        "risk_percent": round(probability * 100, 1),
        "risk_flag": probability >= threshold,
        "decision_threshold_percent": round(threshold * 100, 1),
        "positive_snapshot_rate_percent": round(bundle.get("positive_snapshot_rate", 0) * 100, 2),
        "label_name": bundle["label_name"],
        "horizon": bundle["horizon"],
        "evidence": evidence,
        "metrics": bundle["metrics"],
        "evaluation_split": bundle.get(
            "evaluation_split",
            "Held-out 20% test split from Olist e-commerce orders.",
        ),
        "limitations": bundle["limitations"],
        "days_since_last_order": features["days_since_last_order"],
        "inactivity_days": bundle.get("inactivity_days", 90),
        "churn_status": "inactive_90_days" if features["days_since_last_order"] >= bundle.get("inactivity_days", 90) else "active_under_90_days",
        "review_signal": analyze_review(review_text),
    }

    # Retrieve guidance using the actual customer-history fields shown in the interface.
    case["retention_protocols"] = retrieve_retention_protocols(
        {**features, "inactivity_days": case["inactivity_days"]}, evidence
    )
    case["customer_history"] = features
    return case


def fallback_recommendation(case):
    days = case.get("days_since_last_order", 0)
    threshold = case.get("inactivity_days", 90)
    evidence = case.get("evidence", [])
    reasons = [item["feature"].replace("_", " ") for item in evidence[:3]]
    reason_text = ", ".join(reasons) if reasons else "limited prior order history"
    if case.get("churn_status") == "inactive_90_days":
        return (
            f"What: The customer is inactive at {days} days since their last order. "
            f"Why: The project inactivity rule is {threshold} days; the main supporting signals are {reason_text}. "
            "How: Check prior order and delivery history, then send a personalized re-engagement message with a relevant next step."
        )
    remaining = max(0, threshold - days)
    return (
        f"What: The customer is active but approaching the {threshold}-day inactivity mark ({days} days since the last order; {remaining} days remain). "
        f"Why: The model’s strongest signals are {reason_text}. "
        "How: Review any past service issue and send a timely, relevant reminder or offer before the inactivity threshold."
    )


def gemini_reply(case, question, history=None, mode="chat"):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    context = {
        "business_domain": "E-Commerce Customer Retention (Olist Dataset)",
        "model_churn_score_percent": case["risk_percent"],
        "customer_history": case.get("customer_history", {}),
        "days_since_last_order": case.get("days_since_last_order"),
        "inactivity_threshold_days": case.get("inactivity_days", 90),
        "deterministic_inactivity_status": case.get("churn_status"),
        "above_decision_threshold": case["risk_flag"],
        "decision_threshold_percent": case["decision_threshold_percent"],
        "target_definition": case["label_name"],
        "prediction_horizon": case["horizon"],
        "key_risk_drivers": case["evidence"],
        "customer_review_sentiment": case.get("review_signal"),
        "rag_retrieved_retention_protocols": case.get("retention_protocols", []),
    }

    if mode == "recommendation":
        system_prompt = (
            "You are RetainX, a practical e-commerce retention advisor speaking to a business owner. "
            "Write a useful initial briefing with four short sections: WHAT the data says, WHY (cite the customer's actual values and strongest feature evidence), "
            "HOW the evidence relates to the 90-day inactivity rule, and WHAT TO DO next (one or two feasible actions grounded in the provided protocols). "
            "Use plain business language, avoid model jargon, avoid repeating generic advice, and do not invent customer history or claim correlations are causes. "
            "State when the customer is already inactive at 90 days. Keep the whole answer under 180 words. "
            "Use only the supplied customer context and retrieved protocols. Context (JSON): " + json.dumps(context)
        )
    else:
        system_prompt = (
            "You are RetainX, a concise business retention assistant. Answer only the user's latest question using relevant supplied context. "
            "Do not repeat the full customer briefing or add unrelated recommendations. Follow explicit format and length requests exactly, including one-line or bullet-only requests. "
            "If the user asks for a short answer, give a short answer. Use plain business language, do not invent facts, and distinguish model associations from causes. "
            "Use conversation history only to resolve references. Context (JSON): " + json.dumps(context)
        )

    contents = []
    for item in (history or [])[-8:]:
        if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}:
            continue
        content = item.get("content")
        if isinstance(content, str) and len(content) <= MAX_CHAT_MESSAGE:
            contents.append({
                "role": "model" if item["role"] == "assistant" else "user",
                "parts": [{"text": content}],
            })
    contents.append({"role": "user", "parts": [{"text": question}]})

    model = quote(os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"), safe="")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?"
        + urlencode({"key": api_key})
    )
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 650 if mode == "recommendation" else 500, "temperature": 0.25},
    }
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-goog-api-key": api_key},
        method="POST",
    )

    last_error = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=45) as response:
                result = json.loads(response.read().decode("utf-8"))
            break
        except Exception as exc:
            last_error = exc
            error_body = ""
            if hasattr(exc, "read"):
                try:
                    error_body = exc.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    pass
            app.logger.warning(
                "Gemini attempt %d/%d failed: %s %s %s",
                attempt + 1, 3, type(exc).__name__, exc, error_body,
            )
            if attempt < 2:
                import time
                time.sleep(1.5 * (attempt + 1))
                request = Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json", "X-goog-api-key": api_key},
                    method="POST",
                )
    else:
        raise AIProviderError(f"Gemini request failed after 3 attempts: {last_error}") from None

    if not isinstance(result, dict):
        raise AIProviderError("Gemini returned an empty response.")
    candidates = result.get("candidates", [])
    if not candidates or not isinstance(candidates[0], dict):
        raise AIProviderError("Gemini returned an empty response.")
    content = candidates[0].get("content")
    if not isinstance(content, dict):
        raise AIProviderError("Gemini returned an empty response.")
    parts = content.get("parts", [])
    if not isinstance(parts, list):
        raise AIProviderError("Gemini returned an empty response.")
    answer = " ".join(
        part["text"].strip()
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"].strip()
    )
    if not answer:
        raise AIProviderError("Gemini returned an empty response.")
    return answer


@app.get("/")
def home():
    available = {industry: MODELS.get(industry) is not None for industry in INDUSTRIES}
    return render_template(
        "index.html",
        model_availability=available,
        ai_configured=bool(os.environ.get("GEMINI_API_KEY")),
        review_metrics=REVIEW_MODEL["metrics"] if REVIEW_MODEL else None,
    )


@app.get("/healthz")
def health():
    available = {industry: MODELS.get(industry) is not None for industry in INDUSTRIES}
    available["reviews"] = REVIEW_MODEL is not None
    return jsonify({
        "status": "ok" if any(available.values()) else "degraded",
        "models": available,
        "ai_configured": bool(os.environ.get("GEMINI_API_KEY")),
    }), 200 if any(available.values()) else 503


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return error_response("Send a JSON request body.", 400)
    industry = data.get("industry", "ecommerce")
    if industry not in INDUSTRIES:
        return error_response("Choose a valid industry.", 400)
    try:
        case = model_case(industry, data.get("customer"))
    except ValueError as exc:
        return error_response(str(exc), 400)
    except RuntimeError as exc:
        return error_response(str(exc), 503)
    return jsonify(case)


@app.post("/api/recommendations")
@limiter.shared_limit("30 per hour", scope="ai-advisor")
def recommendations():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or data.get("industry") not in INDUSTRIES:
        return error_response("Choose a valid business domain.", 400)
    if data.get("consent") is not True:
        return jsonify({
            "answer": "Enable the AI Advisor option to synthesize the retention playbook with Gemini 3.5.",
            "provider": "consent_required",
        })
    try:
        case = model_case(data["industry"], data.get("customer"))
        prompt = (
            "Provide an executive retention briefing for this customer based on the risk score, "
            "review message, and RAG-retrieved retention protocols. Translate any Portuguese review into English."
        )
        answer = gemini_reply(case, prompt, mode="recommendation")
    except ValueError as exc:
        return error_response(str(exc), 400)
    except AIProviderError:
        app.logger.error("Gemini recommendation request failed.")
        return error_response("Gemini could not generate a recommendation. Please retry.", 502)
    if answer is None:
        return jsonify({
            "answer": fallback_recommendation(case),
            "provider": "preview",
        })
    return jsonify({
        "answer": answer,
        "provider": "gemini",
        "retention_protocols": case.get("retention_protocols", []),
    })


@app.post("/api/chat")
@limiter.shared_limit("30 per hour", scope="ai-advisor")
def chat():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or data.get("industry") not in INDUSTRIES:
        return error_response("Choose a valid business domain.", 400)
    question = data.get("question")
    if not isinstance(question, str) or not question.strip() or len(question) > MAX_CHAT_MESSAGE:
        return error_response(f"Enter a question of 1 to {MAX_CHAT_MESSAGE} characters.", 400)
    if data.get("consent") is not True:
        return error_response("Enable AI Advisor before starting a chat.", 403)
    try:
        case = model_case(data["industry"], data.get("customer"))
        answer = gemini_reply(case, question.strip(), data.get("history"), mode="chat")
    except ValueError as exc:
        return error_response(str(exc), 400)
    except AIProviderError:
        app.logger.error("Gemini chat request failed.")
        return error_response("Gemini could not answer right now. Please try again.", 502)
    if answer is None:
        return jsonify({
            "answer": (
                "AI chat is in preview mode because GEMINI_API_KEY is not configured. "
                + fallback_recommendation(case)
            ),
            "provider": "preview",
        })
    return jsonify({"answer": answer, "provider": "gemini"})


@app.errorhandler(413)
def request_too_large(_error):
    return error_response("Request is too large.", 413)


@app.errorhandler(429)
def ai_rate_limit(_error):
    return error_response("AI request limit reached. Try again later.", 429)


if __name__ == "__main__":
    app.run(debug=False)
