# RetainX: E-Commerce Customer Retention & Churn Intelligence
## Master Presentation & Viva Defense Guide (Prepared for November 16th Presentation)

---

### Executive Overview (60-Second Pitch)

> “Good morning, Ma’am. Our project uses the Olist Brazilian e-commerce data to study customer inactivity. Our operational churn rule is 90 days without an order. The dashboard shows the exact days since the customer’s last order and marks them inactive when they reach 90. For customers still under the threshold, a statistical model estimates whether they will cross it before ordering again, based on historical order frequency, customer tenure, spend, prior ratings, and delivery experience. We evaluate on customers held out from training so repeated snapshots of one customer do not cross between train and test. The dashboard explains the model estimate with feature contributions. In this Olist sample, the model performs worse than a majority-class F1 baseline because very few customers reorder within 90 days. We present the inactivity rule as the main churn decision and the statistical estimate as experimental.”

### 1. Dataset and Churn Definition

- Olist contains order timestamps, customer identifiers, delivery records, reviews, and payment/item data from 2016–2018.
- Stable customer identity is `customer_unique_id`, mapped from each order’s `customer_id`.
- **Inactivity threshold:** 90 days.
- At any current snapshot, calculate `days_inactive = today - last_order_date`. Mark the customer inactive/churned when this value is at least 90 days. For customers below 90 days, the model estimates whether a new order will occur before they cross the threshold.
- Snapshots without a complete 90-day observation window before the dataset ends are excluded.
- This is an operational project definition. Olist has no official churn label and cannot prove the customer never returns after the dataset ends.

### 2. What Information Goes Into the Model?

Only data known on or before the snapshot date is used:

| Signal | Meaning |
| --- | --- |
| Days since last order | Recency of customer activity |
| Orders in history | Number of delivered orders to date |
| Customer history length | Days since first observed order |
| Average/total spend | Prior purchase value |
| Average and last review score | Prior customer feedback, when available |
| Average delivery delay and duration | Prior fulfillment experience |
| Last order payment and order details | Most recent observed transaction |

Future orders are used only to create the label, never as input features. The customer-grouped holdout prevents snapshots of the same customer from appearing on both sides of evaluation.

### 3. Model and Evaluation

- **Model:** The churn model uses one-hot encoded categorical fields, standardized numeric features, and balanced Logistic Regression.
- **Class imbalance:** Threshold crossings are extremely common under this operational labeling: about 99.5% of historical snapshots cross before a new purchase. The current model has ROC-AUC 0.738 and F1 0.776, while a majority-only baseline has F1 0.997. Therefore the classifier is experimental and should not be called a useful churn predictor yet.
- **Evidence shown:** Per-customer feature contributions are input values relative to training averages multiplied by learned coefficients. They describe model associations, not causes.
- **Current evaluation:** Customer-grouped holdout ROC-AUC 0.738; F1 0.776. Majority baseline F1 0.997. This indicates the model is not useful at threshold 0.5. The holdout is customer-grouped, not a later-calendar-period test.

### 4. Honest Limitations

- “No order in 90 days” is a proxy label, not proof of permanent churn.
- Olist ends in 2018; customers without the full outcome window are omitted.
- The dataset is historical and may not represent current shopping behavior.
- Observational feature contributions do not prove that delivery or ratings cause churn.
- The low return rate within 90 days makes churn-positive prediction difficult; review the metrics and base rate together.

### 5. RAG (Retrieval-Augmented Generation) & Gemini 3.5 Copilot
When asked: *"How is Gemini integrated? Why not just use standard prompts?"*

* **The Problem with Raw LLMs:** Standard LLMs hallucinate generic, untargeted advice (e.g. "offer good service").
* **Our RAG Architecture (`knowledge_base.py`):**
  1. When a prediction is generated, the system evaluates the top risk drivers.
  2. The **RAG Retriever** matches the customer's risk profile to verified e-commerce retention protocols:
     * *Logistics Delay Service Recovery Protocol* (triggered if delay $> 0$ days).
     * *High Freight Friction Mitigation Strategy* (triggered if freight ratio $> 20\%$).
     * *Negative Review Customer Success Playbook* (triggered if rating $\le 2$ stars).
  3. The retrieved protocols + customer metrics are passed to **Gemini 3.5 Flash**.
  4. Gemini acts as an expert retention strategist:
     * Understands Portuguese reviews and translates the complaint into English.
     * Cites the exact retrieved retention actions (e.g. freight reimbursement, 15% discount code, VIP courier flag).
     * Drafts ready-to-send apology emails and CS recommendations.

---

### 6. Anticipated Teacher Questions & Bulletproof Answers

#### Q1: "Why are the reviews in Portuguese and how does your app handle that?"
> *"Ma'am, Olist is a Brazilian marketplace, so all customer comments were originally written in Portuguese. Rather than discarding or translating the dataset offline, we built a cross-lingual pipeline: our Scikit-Learn NLP model scores the sentiment directly on Portuguese text, and our Gemini 3.5 AI Copilot natively translates the complaint and delivers all executive briefings and email drafts in fluent English."*

#### Q2: "Can your model explain WHY a customer is predicted to churn?"
> *"Yes, ma'am. In our dashboard, every prediction displays a Feature Attribution Breakdown. By multiplying the standardized feature values by our model's coefficients, we extract the top 5 directional drivers (e.g. 'Delivery Delay Raises Risk by +0.18' vs 'Multiple Items Lowers Risk by -0.16'). The customer success team knows the exact root cause before reaching out."*

#### Q3: "What if the customer didn't write a review comment?"
> *"The system handles missing text seamlessly. If no comment is provided, the model relies on the star rating (1–5) and behavioral order telemetry (delay, freight, installments). The review sentiment component activates dynamically whenever text is present."*

#### Q4: "What makes your UI enterprise-ready?"
> *"We implemented a dual-theme design system following Linear and Stripe benchmarks, with a real-time Light/Dark mode switcher, 1-click test profile presets, responsive layout, and an interactive AI copilot chat where agents can ask follow-up questions and generate email drafts."*
