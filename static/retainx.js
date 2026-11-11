(() => {
  const config = window.RETAINX_CONFIG || { models: {}, aiConfigured: false };
  const form = document.getElementById("customer-form");
  const industrySelect = document.getElementById("industry");
  const domainTabs = document.querySelectorAll(".domain-tab");
  const sections = [...document.querySelectorAll(".industry-fields")];
  const errorBox = document.getElementById("error-message");
  const analyzeButton = document.getElementById("analyze-button");
  const consent = document.getElementById("ai-consent");
  
  const chatForm = document.getElementById("chat-form");
  const chatInput = document.getElementById("chat-input");
  const chatSubmit = document.getElementById("chat-submit");
  const chatMessages = document.getElementById("chat-messages");
  const recommendationCard = document.getElementById("recommendation-card");
  const promptChips = document.querySelectorAll(".chip-btn");

  const history = [];
  let latestCustomer = null;
  let latestIndustry = "ecommerce";

  // Customer history scenarios for the 90-day inactivity definition.
  const presets = {
    ecommerce: {
      critical: { days_since_last_order: 120, orders_in_history: 1, days_since_first_order: 120, average_order_value: 145, total_historical_spend: 145, average_items_per_order: 1, last_order_value: 145, last_freight_value: 38.5, last_payment_installments: 1, last_payment_type: "credit_card", average_review_score: 1, last_review_score: 1, average_delivery_delay_days: 14, average_delivery_duration_days: 28 },
      borderline: { days_since_last_order: 75, orders_in_history: 2, days_since_first_order: 180, average_order_value: 85, total_historical_spend: 170, average_items_per_order: 1, last_order_value: 85, last_freight_value: 24, last_payment_installments: 1, last_payment_type: "boleto", average_review_score: 3, last_review_score: 3, average_delivery_delay_days: 2, average_delivery_duration_days: 15 },
      healthy: { days_since_last_order: 20, orders_in_history: 5, days_since_first_order: 360, average_order_value: 320, total_historical_spend: 1600, average_items_per_order: 3, last_order_value: 320, last_freight_value: 18, last_payment_installments: 6, last_payment_type: "credit_card", average_review_score: 5, last_review_score: 5, average_delivery_delay_days: -4, average_delivery_duration_days: 7 }
    }
  };

  const featureLabels = {
    days_since_last_order: "Days Since Last Order",
    average_delivery_delay_days: "Average Delivery Delay",
    average_delivery_duration_days: "Average Delivery Duration",
    average_items_per_order: "Average Items per Order",
    total_historical_spend: "Historical Spend",
    last_order_value: "Last Order Value",
    last_freight_value: "Last Freight Value",
    last_payment_installments: "Last Payment Installments",
    last_review_score: "Last Review Score",
    orders_in_history: "Orders in History",
    days_since_first_order: "Customer History Length",
    average_order_value: "Average Order Value",
    total_historical_spend: "Historical Spend",
    average_items_per_order: "Average Items per Order",
    last_order_value: "Last Order Value",
    last_freight_value: "Last Freight Value",
    last_payment_installments: "Last Payment Installments",
    average_review_score: "Average Review Score",
    last_review_score: "Last Review Score",
    average_delivery_delay_days: "Average Delivery Delay",
    average_delivery_duration_days: "Average Delivery Duration",
    delivery_delay_days: "Delivery Delay (vs Promised Date)",
    delivery_duration_days: "Total Transit Duration",
    freight_value: "Freight / Shipping Cost",
    freight_ratio: "Freight-to-Order Ratio",
    order_value: "Total Order Spend",
    payment_installments: "Payment Installments",
    total_items: "Items in Order",
    review_score: "Customer Review Rating (Stars)",
    review_has_comment: "Review Feedback Provided",
    payment_type: "Payment Method",
    call_failures: "Call Failures & Dropouts",
    complaints: "Customer Complaint On File",
    subscription_length: "Subscription Length",
    charge_amount: "Monthly Charge Index",
    seconds_of_use: "Airtime Usage (Seconds)",
    frequency_of_use: "Call Frequency",
    frequency_of_sms: "SMS Frequency",
    distinct_called_numbers: "Distinct Contacts Called",
    tariff_plan: "Account Tariff Plan",
    registered_via: "Registration Channel",
    registration_tenure_days: "Account Tenure",
    transaction_count: "Transaction Count",
    avg_plan_days: "Average Plan Length",
    avg_plan_price: "Listed Plan Price",
    avg_amount_paid: "Actual Amount Paid",
    auto_renew_rate: "Auto-Renew Ratio",
    cancel_rate: "Cancellation Ratio",
    days_since_last_transaction: "Days Inactive",
    active_days: "Active Listening Days",
    avg_daily_plays: "Daily Stream Plays",
    total_seconds: "Total Listening Duration",
    completion_rate: "Track Completion Rate",
    avg_unique_songs: "Unique Songs Listened"
  };

  function displayFeature(name) {
    const category = name.match(/^([^=]+)=(.+)$/);
    if (category) {
      const label = featureLabels[category[1]] || category[1].replaceAll("_", " ");
      return `${label}: ${category[2].replaceAll("_", " ")}`;
    }
    return featureLabels[name] || name.replaceAll("_", " ");
  }

  // Lightweight Markdown to HTML Renderer for Executive Cleanliness
  function renderMarkdown(md) {
    if (!md) return "";
    let html = md
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/^### (.*$)/gim, "<h4>$1</h4>")
      .replace(/^## (.*$)/gim, "<h3>$1</h3>")
      .replace(/^# (.*$)/gim, "<h2>$1</h2>")
      .replace(/\*\*(.*?)\*\*/gim, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/gim, "<em>$1</em>")
      .replace(/^\s*[\-\*]\s+(.*$)/gim, "<li>$1</li>")
      .replace(/^\s*\d+\.\s+(.*$)/gim, "<li>$1</li>");

    html = html.replace(/(<li>.*?<\/li>)/gms, "<ul>$1</ul>");
    html = html.replace(/<\/ul>\s*<ul>/g, "");
    
    const paragraphs = html.split(/\n{2,}/).map(p => {
      p = p.trim();
      if (!p) return "";
      if (p.startsWith("<h") || p.startsWith("<ul") || p.startsWith("<ol")) return p;
      return `<p>${p.replace(/\n/g, "<br>")}</p>`;
    });

    return paragraphs.join("");
  }

  function setIndustry(industry) {
    latestIndustry = industry;
    industrySelect.value = industry;

    domainTabs.forEach(tab => {
      tab.classList.toggle("active", tab.dataset.domain === industry);
    });

    sections.forEach(section => {
      const active = section.dataset.industry === industry;
      section.classList.toggle("hidden", !active);
      section.querySelectorAll("input, select").forEach(input => {
        input.disabled = !active;
      });
    });

    const hintEl = document.getElementById("industry-hint");
    if (industry === "ecommerce") {
      hintEl.textContent = "Trained on 98,000+ real orders from the Olist Brazilian E-Commerce dataset.";
    } else if (industry === "telecom") {
      hintEl.textContent = "Calibrated for telecom customer churn detection & intervention.";
    } else {
      hintEl.textContent = "Calibrated for subscription streaming & SaaS retention.";
    }

    analyzeButton.disabled = !config.models[industry];
  }

  domainTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      setIndustry(tab.dataset.domain);
      clearError();
      resetResults();
    });
  });

  function loadPreset(profileType) {
    const currentIndustry = latestIndustry || "ecommerce";
    const data = presets[currentIndustry]?.[profileType];
    if (!data) return;

    const activeSection = sections.find(s => s.dataset.industry === currentIndustry);
    if (!activeSection) return;

    Object.entries(data).forEach(([key, val]) => {
      if (key === "review_text") {
        document.getElementById("review-text").value = val;
        return;
      }
      const field = activeSection.querySelector(`[name="${key}"]`);
      if (field) {
        if (field.tagName === "SELECT") {
          field.value = String(val);
        } else {
          field.value = val;
        }
      }
    });

    form.dispatchEvent(new Event("submit"));
  }

  document.getElementById("preset-critical").addEventListener("click", () => loadPreset("critical"));
  document.getElementById("preset-borderline").addEventListener("click", () => loadPreset("borderline"));
  document.getElementById("preset-healthy").addEventListener("click", () => loadPreset("healthy"));

  function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
  }

  function clearError() {
    errorBox.textContent = "";
    errorBox.classList.add("hidden");
  }

  function resetResults() {
    latestCustomer = null;
    document.getElementById("prediction-result").classList.add("hidden");
    document.getElementById("empty-result").classList.remove("hidden");
    recommendationCard.classList.add("hidden");
    chatInput.disabled = true;
    chatSubmit.disabled = true;
    clearChat();
  }

  async function postJson(url, payload) {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "The service request could not be completed.");
    }
    return data;
  }

  function collectCustomer() {
    const activeSection = sections.find(s => s.dataset.industry === latestIndustry);
    const customer = {};
    activeSection.querySelectorAll("input, select").forEach(input => {
      if (input.type === "number") {
        customer[input.name] = Number(input.value);
      } else if (input.name === "complaints") {
        customer[input.name] = input.value === "true";
      } else {
        customer[input.name] = input.value;
      }
    });
    customer.review_text = document.getElementById("review-text").value.trim();
    return customer;
  }

  function renderEvidence(items) {
    const list = document.getElementById("evidence-list");
    list.replaceChildren();
    if (!items.length) {
      const item = document.createElement("li");
      item.className = "evidence-item";
      item.textContent = "No strong model evidence signals detected for this customer history.";
      list.append(item);
      return;
    }
    items.forEach(evidence => {
      const item = document.createElement("li");
      item.className = "evidence-item";

      const labelWrap = document.createElement("div");
      labelWrap.className = "evidence-label";

      const name = document.createElement("span");
      name.textContent = displayFeature(evidence.feature);
      labelWrap.append(name);

      const delta = document.createElement("div");
      delta.className = `evidence-delta ${evidence.direction}`;
      delta.textContent = evidence.direction === "raises" ? `▲ Raises Churn Risk (+${evidence.strength})` : `▼ Lowers Churn Risk (-${evidence.strength})`;

      item.append(labelWrap, delta);
      list.append(item);
    });
  }

  function renderCase(result) {
    document.getElementById("empty-result").classList.add("hidden");
    document.getElementById("prediction-result").classList.remove("hidden");

    const riskVal = result.risk_percent.toFixed(1);
    document.getElementById("risk-number").textContent = riskVal;

    const badge = document.getElementById("risk-badge");
    const track = document.getElementById("risk-track");
    track.style.width = `${Math.max(2, result.risk_percent)}%`;

    if (result.risk_percent >= 60) {
      badge.textContent = "HIGH 90-DAY NO-RETURN SCORE";
      badge.className = "risk-tag critical";
    } else if (result.risk_percent >= 40) {
      badge.textContent = "ELEVATED 90-DAY SCORE";
      badge.className = "risk-tag moderate";
    } else {
      badge.textContent = "LOWER 90-DAY SCORE";
      badge.className = "risk-tag healthy";
    }

    document.getElementById("target-definition").textContent =
      `${result.churn_status === "inactive_90_days" ? "INACTIVE (90+ days)" : "ACTIVE (<90 days)"} | ${result.label_name}`;

    const reviewSignal = document.getElementById("review-signal");
    if (result.review_signal && result.review_signal.sentiment !== "unavailable") {
      reviewSignal.classList.remove("hidden");
      const sentiment = result.review_signal.sentiment;
      const score = result.review_signal.positive_score_percent;
      reviewSignal.innerHTML = `<strong>Customer Review NLP Sentiment:</strong> ${sentiment.toUpperCase()} (${score}% positive rating indicator)`;
    } else {
      reviewSignal.classList.add("hidden");
    }

    renderEvidence(result.evidence || []);

    const precision = Number(result.metrics.precision).toLocaleString(undefined, { style: "percent", maximumFractionDigits: 0 });
    const recall = Number(result.metrics.recall).toLocaleString(undefined, { style: "percent", maximumFractionDigits: 0 });
    document.getElementById("model-quality").textContent =
      `90-day customer-grouped holdout: ROC-AUC ${Number(result.metrics.roc_auc).toFixed(2)} · Precision ${precision} · Recall ${recall} · F1 ${Number(result.metrics.f1).toFixed(2)} · 90-day no-return base rate ${result.positive_snapshot_rate_percent}%`;
  }

  function clearChat() {
    history.splice(0, history.length);
    chatMessages.replaceChildren();
    const empty = document.createElement("div");
    empty.className = "chat-empty";
    empty.textContent = "Ask any follow-up question regarding this customer's churn signals, delivery delays, or intervention strategy.";
    chatMessages.append(empty);
  }

  function addBubble(role, content) {
    const empty = chatMessages.querySelector(".chat-empty");
    if (empty) empty.remove();
    const bubble = document.createElement("div");
    bubble.className = `bubble ${role}`;
    bubble.innerHTML = renderMarkdown(content);
    chatMessages.append(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  async function requestRecommendation() {
    if (!latestCustomer || !latestIndustry) return;
    recommendationCard.classList.remove("hidden");
    const providerLabel = document.getElementById("recommendation-provider");
    const bodyEl = document.getElementById("recommendation-text");

    providerLabel.textContent = "Preparing a concise customer briefing...";
    bodyEl.innerHTML = "<p>Reviewing customer history and relevant next steps...</p>";

    try {
      const result = await postJson("/api/recommendations", {
        industry: latestIndustry,
        customer: latestCustomer,
        consent: consent.checked
      });

      providerLabel.textContent = result.provider === "gemini" ? "Customer retention briefing" : "Customer retention briefing";
      bodyEl.innerHTML = renderMarkdown(result.answer);
    } catch (error) {
      providerLabel.textContent = "AI Advisory Notice";
      bodyEl.innerHTML = `<p>${error.message}</p>`;
    }
  }

  form.addEventListener("submit", async event => {
    event.preventDefault();
    clearError();
    analyzeButton.disabled = true;
    analyzeButton.querySelector(".btn-text").textContent = "Analyzing Signals...";

    try {
      latestCustomer = collectCustomer();
      latestCustomer.total_historical_spend = Number(latestCustomer.average_order_value || 0) * Number(latestCustomer.orders_in_history || 1);
      latestCustomer.average_items_per_order = Number(latestCustomer.average_items_per_order || 1);
      latestCustomer.last_order_value = Number(latestCustomer.average_order_value || 0);
      latestCustomer.last_freight_value = Number(latestCustomer.last_freight_value || 0);
      latestCustomer.last_payment_installments = Number(latestCustomer.last_payment_installments || 1);
      latestCustomer.last_payment_type = latestCustomer.last_payment_type || "unknown";
      latestCustomer.last_review_score = Number(latestCustomer.average_review_score || 0);
      latestCustomer.average_delivery_duration_days = Number(latestCustomer.average_delivery_duration_days || 0);
      const result = await postJson("/api/predict", {
        industry: latestIndustry,
        customer: latestCustomer
      });

      renderCase(result);
      clearChat();
      chatInput.disabled = false;
      chatSubmit.disabled = false;

      if (consent.checked) {
        await requestRecommendation();
      } else {
        recommendationCard.classList.add("hidden");
      }
    } catch (error) {
      showError(error.message);
    } finally {
      analyzeButton.disabled = false;
      analyzeButton.querySelector(".btn-text").textContent = "Execute E-Commerce Churn Analysis";
    }
  });

  consent.addEventListener("change", () => {
    if (!consent.checked) {
      recommendationCard.classList.add("hidden");
      history.splice(0, history.length);
      return;
    }
    if (latestCustomer) requestRecommendation();
  });

  promptChips.forEach(chip => {
    chip.addEventListener("click", () => {
      if (!latestCustomer) {
        showError("Please run an analysis first to provide customer context.");
        return;
      }
      chatInput.value = chip.dataset.prompt;
      chatForm.dispatchEvent(new Event("submit"));
    });
  });

  chatForm.addEventListener("submit", async event => {
    event.preventDefault();
    const question = chatInput.value.trim();
    if (!question || !latestCustomer || !latestIndustry) return;

    if (!consent.checked) {
      showError("Enable AI Advisor to chat with the retention copilot.");
      return;
    }

    clearError();
    addBubble("user", question);
    chatInput.value = "";
    chatInput.disabled = true;
    chatSubmit.disabled = true;

    try {
      const result = await postJson("/api/chat", {
        industry: latestIndustry,
        customer: latestCustomer,
        question,
        history: history.slice(-8),
        consent: consent.checked
      });
      addBubble("assistant", result.answer);
      history.push({ role: "user", content: question }, { role: "assistant", content: result.answer });
    } catch (error) {
      addBubble("assistant", error.message);
    } finally {
      chatInput.disabled = false;
      chatSubmit.disabled = false;
      chatInput.focus();
    }
  });

  // Theme Toggle (Dark & Light)
  const themeToggle = document.getElementById("theme-toggle");
  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      const current = document.documentElement.getAttribute("data-theme") || "dark";
      const nextTheme = current === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", nextTheme);
      try {
        localStorage.setItem("retainx_theme", nextTheme);
      } catch (e) {}
    });
  }

  setIndustry("ecommerce");
})();
