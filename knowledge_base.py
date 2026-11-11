"""
E-Commerce Retention Knowledge Base & Retrieval System (RAG)
Provides grounded, verified customer retention protocols tailored to specific e-commerce risk signals.
"""

RETENTION_KNOWLEDGE_BASE = {
    "delivery_delay": {
        "title": "Logistics & Delivery Delay Service Recovery Protocol",
        "condition": "Delivery delay exceeds estimated arrival date (>0 days)",
        "protocol": [
            "Proactive Apology & Tracking: Trigger an automated, sincere delivery apology notification acknowledging the carrier delay before the customer complains.",
            "Shipping Fee Compensation: Credit the freight amount ($R$) or provide a 15% discount coupon applicable to their next order as compensation for transit friction.",
            "VIP Fulfillment Flag: Flag the customer's account for priority courier dispatch and automated tracking SMS updates on all subsequent purchases."
        ]
    },
    "freight_friction": {
        "title": "High Shipping/Freight Cost Mitigation Strategy",
        "condition": "Freight fee exceeds 20% of total order value",
        "protocol": [
            "Free Shipping Threshold Incentive: Offer a 'Free Shipping on Orders Over R$ 100' personalized voucher valid for 30 days.",
            "Bundle / Multi-item Recommendation: Present product bundles that amortize shipping costs across multiple items in their preferred category.",
            "Regional Fulfillment Routing: Highlight local merchant inventory within the customer's home state to minimize regional transit surcharges."
        ]
    },
    "low_review_score": {
        "title": "Dissatisfied Buyer & Negative Review Recovery Playbook",
        "condition": "Customer rated order 1 or 2 stars or expressed negative sentiment",
        "protocol": [
            "Priority Customer Success Escalation: Route review feedback to the tier-2 customer support team for personal outreach within 24 hours.",
            "Zero-Friction Resolution: Offer immediate no-return-required replacement or instant store credit if product arrived defective or non-compliant.",
            "Seller Accountability: Audit the merchant fulfillment rating and temporarily down-rank repeat-offending sellers."
        ]
    },
    "single_item_dropoff": {
        "title": "Single-Purchase Conversion & Re-Engagement Cadence",
        "condition": "Customer bought only 1 item and has not repurchased",
        "protocol": [
            "Lifecycle Re-Engagement Campaign: Send a curated replenishment / accessory email on Day 14 and Day 28 post-delivery based on category lifespan.",
            "Loyalty Points Kickstart: Grant introductory reward points (e.g., 50 RetainX reward coins) expiring in 30 days to incentivize a second transaction.",
            "Social Proof Showcase: Highlight top customer favorites and peer reviews in the same category."
        ]
    },
    "high_order_value_retention": {
        "title": "High-Value Account White-Glove Retention Playbook",
        "condition": "Order value is in top tier (> R$ 250)",
        "protocol": [
            "Dedicated Support Channel: Provide priority WhatsApp / email concierge support for any future order inquiries.",
            "Exclusive Tier Invitation: Automatically enroll customer into the Silver/Gold tier with permanent free shipping benefits.",
            "Early Access Privileges: Invite customer to early flash sales and new collection previews."
        ]
    }
}


def retrieve_retention_protocols(customer_data, evidence):
    """Select only guidance supported by fields in the current customer-history form."""
    retrieved = []
    days = customer_data.get("days_since_last_order", 0)
    threshold = customer_data.get("inactivity_days", 90)
    if days >= threshold:
        retrieved.append({
            "title": "90-Day Inactivity Re-Engagement",
            "condition": f"Customer has been inactive for {days} days (threshold: {threshold}).",
            "protocol": [
                "Check the last order and any unresolved service issue before contacting the customer.",
                "Send one personalized re-engagement message tied to the customer’s prior purchase history.",
                "Use a relevant incentive only when it fits the customer’s history and business policy.",
            ],
        })
    elif days >= threshold * 0.7:
        retrieved.append({
            "title": "Approaching 90-Day Inactivity",
            "condition": f"Customer is {days} days since last order, nearing the {threshold}-day threshold.",
            "protocol": [
                "Review the most recent order for a clear friction point.",
                "Send a timely, relevant reminder before the inactivity threshold.",
            ],
        })

    if customer_data.get("average_review_score", 5) and customer_data["average_review_score"] <= 2.5:
        retrieved.append(RETENTION_KNOWLEDGE_BASE["low_review_score"])
    if customer_data.get("average_delivery_delay_days", 0) > 0:
        retrieved.append(RETENTION_KNOWLEDGE_BASE["delivery_delay"])
    if customer_data.get("orders_in_history", 0) <= 1:
        retrieved.append(RETENTION_KNOWLEDGE_BASE["single_item_dropoff"])
    if not retrieved:
        retrieved.append({
            "title": "Maintain Relevant Customer Engagement",
            "condition": "Customer is below the inactivity threshold and no strong prior friction signal is present.",
            "protocol": ["Continue normal, relevant lifecycle communication based on prior purchases."],
        })
    return retrieved
