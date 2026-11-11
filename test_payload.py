import json
from urllib.request import Request, urlopen

payload = {
    "industry": "ecommerce",
    "consent": True,
    "customer": {
        "order_value": 120.0,
        "freight_value": 45.0,
        "delivery_delay_days": 8.0,
        "delivery_duration_days": 22.0,
        "payment_type": "credit_card",
        "payment_installments": 1,
        "total_items": 1,
        "review_score": 1,
        "review_text": "O produto demorou mais de uma semana além do prazo e a caixa veio toda rasgada. Péssima experiência.",
    },
}

req = Request(
    "http://127.0.0.1:5000/api/recommendations",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

try:
    with urlopen(req, timeout=60) as res:
        data = json.loads(res.read().decode("utf-8"))
        print("Provider:", data.get("provider"))
        print("Answer:\n", data.get("answer", ""))
except Exception as e:
    body = ""
    if hasattr(e, "read"):
        body = e.read().decode("utf-8", errors="replace")[:300]
    print("Error:", e, body)
