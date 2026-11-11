import io
import os
import unittest
from unittest.mock import patch

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import app as retainx


def build_test_bundle():
    rows = [
        {"days_since_last_order": 20, "orders_in_history": 2, "days_since_first_order": 100,
         "average_order_value": 120, "total_historical_spend": 240, "average_items_per_order": 1,
         "last_order_value": 120, "last_freight_value": 20, "last_payment_installments": 1,
         "last_payment_type": "credit_card", "average_review_score": 5, "last_review_score": 5,
         "average_delivery_delay_days": 0, "average_delivery_duration_days": 7},
        {"days_since_last_order": 80, "orders_in_history": 1, "days_since_first_order": 80,
         "average_order_value": 80, "total_historical_spend": 80, "average_items_per_order": 1,
         "last_order_value": 80, "last_freight_value": 25, "last_payment_installments": 1,
         "last_payment_type": "boleto", "average_review_score": 1, "last_review_score": 1,
         "average_delivery_delay_days": 5, "average_delivery_duration_days": 20},
    ] * 4
    labels = [0, 1] * 4
    vectorizer = DictVectorizer(sparse=False)
    transformed = vectorizer.fit_transform(rows)
    scaler = StandardScaler().fit(transformed)
    classifier = LogisticRegression().fit(scaler.transform(transformed), labels)
    return {
        "vectorizer": vectorizer, "scaler": scaler, "classifier": classifier,
        "decision_threshold": 0.5,
        "label_name": "No repeat order within the next 90 days",
        "horizon": "90-day repeat-purchase forecast at an active-customer snapshot",
        "inactivity_days": 90, "metrics": {"roc_auc": 0.85, "f1": 0.72, "precision": 0.8, "recall": 0.7},
        "limitations": "Olist historical data proxy.",
    }


def ecommerce_customer():
    return {
        "days_since_last_order": 80, "orders_in_history": 1, "days_since_first_order": 80,
        "average_order_value": 80, "total_historical_spend": 80, "average_items_per_order": 1,
        "last_order_value": 80, "last_freight_value": 25, "last_payment_installments": 1,
        "last_payment_type": "boleto", "average_review_score": 1, "last_review_score": 1,
        "average_delivery_delay_days": 5, "average_delivery_duration_days": 20,
        "review_text": "O produto demorou muito para chegar e a caixa veio violada.",
    }


class RetainXApiTests(unittest.TestCase):
    def setUp(self):
        self.client = retainx.app.test_client()
        self.bundle = build_test_bundle()
        self.models = {"ecommerce": self.bundle}
        self.patch_models = patch.dict(retainx.MODELS, self.models, clear=True)
        self.patch_models.start()

    def tearDown(self):
        self.patch_models.stop()

    def test_home_renders_brand_and_ecommerce_ui(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"RetainX", response.data)
        self.assertIn(b"OLIST E-COMMERCE AI", response.data)
        self.assertIn(b"Days since last order", response.data)
        self.assertIn(b"Orders in customer history", response.data)
        self.assertNotIn(b"Music streaming", response.data)
        self.assertNotIn(b"Telecom", response.data)

    def test_predict_returns_score_and_feature_evidence(self):
        response = self.client.post(
            "/api/predict",
            json={"industry": "ecommerce", "customer": ecommerce_customer()},
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertGreaterEqual(body["risk_percent"], 0)
        self.assertLessEqual(body["risk_percent"], 100)
        self.assertTrue(body["evidence"])
        self.assertEqual(body["label_name"], "No repeat order within the next 90 days")
        self.assertIn(body["churn_status"], {"active_under_90_days", "inactive_90_days"})

    def test_predict_rejects_non_finite_and_invalid_inputs(self):
        customer = ecommerce_customer()
        customer["days_since_last_order"] = float("nan")
        response = self.client.post("/api/predict", json={"industry": "ecommerce", "customer": customer})
        self.assertEqual(response.status_code, 400)
        self.assertIn("finite", response.get_json()["error"])

        customer["days_since_last_order"] = -50.0
        response = self.client.post("/api/predict", json={"industry": "ecommerce", "customer": customer})
        self.assertEqual(response.status_code, 400)

    def test_unsupported_industry_returns_bad_request(self):
        response = self.client.post(
            "/api/predict",
            json={"industry": "telecom", "customer": {}},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Choose a valid industry", response.get_json()["error"])

    def test_missing_model_returns_service_unavailable(self):
        retainx.MODELS["ecommerce"] = None
        response = self.client.post(
            "/api/predict",
            json={"industry": "ecommerce", "customer": ecommerce_customer()},
        )
        self.assertEqual(response.status_code, 503)

    def test_chat_requires_explicit_ai_consent(self):
        response = self.client.post(
            "/api/chat",
            json={"industry": "ecommerce", "customer": ecommerce_customer(), "question": "What should I do?"},
        )
        self.assertEqual(response.status_code, 403)

    def test_chat_preview_is_explicit_when_gemini_api_key_is_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            response = self.client.post(
                "/api/chat",
                json={
                    "industry": "ecommerce",
                    "customer": ecommerce_customer(),
                    "question": "What should I do?",
                    "consent": True,
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["provider"], "preview")
        self.assertIn("preview mode", response.get_json()["answer"])

    def test_gemini_reply_uses_configured_model_and_returns_generated_text(self):
        response_body = b'{"candidates":[{"content":{"parts":[{"text":"Helpful retention response."}]}}]}'
        with patch.dict(os.environ, {
            "GEMINI_API_KEY": "test-only-gemini-key",
            "GEMINI_MODEL": "gemini-test-model",
        }), patch.object(retainx, "urlopen", return_value=io.BytesIO(response_body)) as send:
            answer = retainx.gemini_reply(
                retainx.model_case("ecommerce", ecommerce_customer()),
                "What should I do?",
            )

        self.assertEqual(answer, "Helpful retention response.")
        request = send.call_args.args[0]
        self.assertIn("gemini-test-model:generateContent", request.full_url)
        self.assertIn("test-only-gemini-key", request.full_url)
        self.assertEqual(send.call_args.kwargs["timeout"], 45)
        payload = __import__("json").loads(request.data.decode("utf-8"))
        self.assertIn("latest question", payload["systemInstruction"]["parts"][0]["text"])
        self.assertLessEqual(payload["generationConfig"]["maxOutputTokens"], 500)

    def test_review_sentiment_is_analyzed_with_rag_playbook(self):
        review_vectorizer = TfidfVectorizer()
        review_texts = [
            "otimo atendimento entrega rapida excelente",
            "pessimo produto veio quebrado e atrasou muito",
        ] * 5
        review_labels = [1, 0] * 5
        review_matrix = review_vectorizer.fit_transform(review_texts)
        review_model = {
            "vectorizer": review_vectorizer,
            "classifier": LogisticRegression().fit(review_matrix, review_labels),
            "limitations": ["Olist star ratings used as binary sentiment supervision."],
        }
        source_text = "pessimo produto veio quebrado e atrasou muito"
        customer = ecommerce_customer()
        customer["review_text"] = source_text
        with patch.object(retainx, "REVIEW_MODEL", review_model), patch.object(
            retainx, "gemini_reply", return_value="Review recovery playbook applied."
        ) as ai_reply:
            response = self.client.post(
                "/api/recommendations",
                json={"industry": "ecommerce", "customer": customer, "consent": True},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["answer"], "Review recovery playbook applied.")
        self.assertEqual(response.get_json()["provider"], "gemini")
        self.assertEqual(ai_reply.call_args.kwargs["mode"], "recommendation")
        context = ai_reply.call_args.args[0]
        self.assertIn(context["review_signal"]["sentiment"], {"negative", "mixed", "positive"})
        self.assertIn("retention_protocols", context)
        self.assertGreater(len(context["retention_protocols"]), 0)


if __name__ == "__main__":
    unittest.main()
