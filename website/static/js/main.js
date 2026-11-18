// PromiseGuard™ Enterprise Client Logic

document.addEventListener("DOMContentLoaded", () => {
  initSliderBadges();
  initPredictionForm();
  initSimulationSlider();
  initRoiCalculator();
  initFaqAccordion();
  initLiveOrderTicker();
});

// FAQ Accordion Handler
function initFaqAccordion() {
  const faqQuestions = document.querySelectorAll(".faq-question");
  faqQuestions.forEach(q => {
    q.addEventListener("click", () => {
      const parent = q.parentElement;
      const answer = parent.querySelector(".faq-answer");
      const icon = q.querySelector(".faq-icon");
      if (answer) {
        const isHidden = answer.style.display === "none" || !answer.style.display;
        answer.style.display = isHidden ? "block" : "none";
        if (icon) {
          icon.textContent = isHidden ? "▲" : "▼";
        }
      }
    });
  });
}

// 1. Sync sliders with display badges
function initSliderBadges() {
  const sliders = document.querySelectorAll("input[type=range]");
  sliders.forEach(slider => {
    const badge = document.getElementById(slider.id + "-val");
    if (badge) {
      slider.addEventListener("input", () => {
        let suffix = "";
        if (slider.id === "distance_km") suffix = " km";
        else if (slider.id === "order_hour") suffix = ":00";
        else if (slider.id === "Delivery_person_Age") suffix = " yrs";
        badge.textContent = slider.value + suffix;
      });
    }
  });
}

// 2. Interactive Prediction Form AJAX
function initPredictionForm() {
  const form = document.getElementById("eta-prediction-form");
  if (!form) return;

  const calculateBtn = document.getElementById("btn-calculate-eta");
  if (calculateBtn) {
    calculateBtn.addEventListener("click", runPredictionAjax);
  }

  // Auto-recalculate on dropdown select change
  form.addEventListener("change", () => {
    runPredictionAjax();
  });

  // Debounced auto-recalculate on slider drag (sub-200ms reactive feel)
  let debounceTimer = null;
  form.addEventListener("input", (e) => {
    if (e.target.type === "range") {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        runPredictionAjax();
      }, 150);
    }
  });
}

async function runPredictionAjax() {
  const form = document.getElementById("eta-prediction-form");
  if (!form) return;

  const payload = {
    distance_km: parseFloat(document.getElementById("distance_km").value),
    Delivery_person_Age: parseFloat(document.getElementById("Delivery_person_Age").value),
    Delivery_person_Ratings: parseFloat(document.getElementById("Delivery_person_Ratings").value),
    multiple_deliveries: parseFloat(document.getElementById("multiple_deliveries").value),
    pickup_prep_delay_min: parseFloat(document.getElementById("pickup_prep_delay_min").value),
    order_hour: parseInt(document.getElementById("order_hour").value),
    Vehicle_condition: parseInt(document.getElementById("Vehicle_condition").value),
    Weather_conditions: document.getElementById("Weather_conditions").value,
    Road_traffic_density: document.getElementById("Road_traffic_density").value,
    Type_of_order: document.getElementById("Type_of_order").value,
    Type_of_vehicle: document.getElementById("Type_of_vehicle").value,
    Festival: document.getElementById("Festival").value,
    City: document.getElementById("City").value
  };

  const statusEl = document.getElementById("prediction-loading-status");
  if (statusEl) statusEl.textContent = "⚡ Running PromiseGuard enterprise ML inference...";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const res = await response.json();
    if (res.status === "success") {
      updatePredictionUI(res.data);
    }
  } catch (err) {
    console.error("Prediction error:", err);
  } finally {
    if (statusEl) statusEl.textContent = "";
  }
}

function updatePredictionUI(data) {
  // 1. Text elements
  const elEta = document.getElementById("res-pred-eta");
  if (elEta) elEta.textContent = data.predicted_eta;

  const elQLow = document.getElementById("res-q-low");
  const elQHigh = document.getElementById("res-q-high");
  if (elQLow && elQHigh) {
    elQLow.textContent = data.pred_q_low;
    elQHigh.textContent = data.pred_q_high;
  }

  const elConfLow = document.getElementById("res-conf-low");
  const elConfHigh = document.getElementById("res-conf-high");
  if (elConfLow && elConfHigh) {
    elConfLow.textContent = data.conf_low;
    elConfHigh.textContent = data.conf_high;
  }

  const elSla = document.getElementById("res-promised-sla");
  if (elSla) elSla.textContent = data.promised_sla + " min";

  const elRiskVal = document.getElementById("res-risk-percentage");
  if (elRiskVal) elRiskVal.textContent = data.risk_percentage + "%";

  const elRiskBadge = document.getElementById("res-risk-badge");
  if (elRiskBadge) {
    elRiskBadge.textContent = data.risk_level;
    elRiskBadge.className = "badge " + (
      data.risk_class === "danger" ? "badge-danger" :
      data.risk_class === "warning" ? "badge-warning" : "badge-success"
    );
  }

  // 2. Recommendation Box
  const elRecBox = document.getElementById("res-recommendation-box");
  const elRecText = document.getElementById("res-recommendation-text");
  if (elRecBox && elRecText) {
    elRecBox.className = "recommendation-box " + data.risk_class;
    elRecText.textContent = data.recommendation;
  }

  // 3. Risk Contributors List
  const elContribs = document.getElementById("res-contributors-list");
  if (elContribs) {
    elContribs.innerHTML = "";
    data.contributors.forEach(c => {
      const li = document.createElement("li");
      li.textContent = c;
      elContribs.appendChild(li);
    });
  }

  // 4. Interactive Timeline Bar
  updateTimelineVisualization(data);
}

function updateTimelineVisualization(data) {
  const maxMin = 70.0;
  const qLowPct = Math.max(0, Math.min(100, (data.pred_q_low / maxMin) * 100));
  const qHighPct = Math.max(0, Math.min(100, (data.pred_q_high / maxMin) * 100));
  const etaPct = Math.max(0, Math.min(100, (data.predicted_eta / maxMin) * 100));
  const slaPct = Math.max(0, Math.min(100, (data.promised_sla / maxMin) * 100));

  const intervalEl = document.getElementById("timeline-interval");
  if (intervalEl) {
    intervalEl.style.left = qLowPct + "%";
    intervalEl.style.width = Math.max(2, (qHighPct - qLowPct)) + "%";
  }

  const etaEl = document.getElementById("timeline-eta-marker");
  if (etaEl) {
    etaEl.style.left = etaPct + "%";
  }

  const slaEl = document.getElementById("timeline-sla-marker");
  if (slaEl) {
    slaEl.style.left = slaPct + "%";
  }
}

// 3. Dynamic Threshold Simulation Interactive Slider
function initSimulationSlider() {
  const simSlider = document.getElementById("sim-threshold-slider");
  if (!simSlider) return;

  const badge = document.getElementById("sim-threshold-val");

  simSlider.addEventListener("input", async () => {
    const val = parseFloat(simSlider.value);
    if (badge) badge.textContent = val.toFixed(2);

    try {
      const response = await fetch("/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ threshold: val })
      });
      const res = await response.json();
      if (res.status === "success") {
        updateSimulationMetrics(res.data);
      }
    } catch (e) {
      console.error(e);
    }
  });
}

function updateSimulationMetrics(d) {
  const setEl = (id, text) => {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  };

  const trueLate = d.true_positives !== undefined ? d.true_positives : (d.true_late_captured !== undefined ? d.true_late_captured : 399);
  const missedLate = d.false_negatives !== undefined ? d.false_negatives : (d.missed_late_orders !== undefined ? d.missed_late_orders : 41);

  setEl("sim-alerts-sent", `${d.alerts_sent.toLocaleString()} (${d.alert_rate_pct}%)`);
  setEl("sim-recall", `${d.recall_pct}%`);
  setEl("sim-precision", `${d.precision_pct}%`);
  setEl("sim-cost-index", d.illustrative_cost_index);
  setEl("sim-delays-captured", `${trueLate} of 440 delayed orders caught`);
  setEl("sim-missed-delays", `${missedLate}`);
}

// 4. Enterprise ROI & Loss Prevention Calculator (Zomato / Swiggy Scale)
function initRoiCalculator() {
  const orderSlider = document.getElementById("roi-daily-orders");
  const costSlider = document.getElementById("roi-refund-cost");
  if (!orderSlider || !costSlider) return;

  function recalculateRoi() {
    const dailyOrders = parseInt(orderSlider.value);
    const refundCost = parseInt(costSlider.value);

    const orderBadge = document.getElementById("roi-orders-val");
    const costBadge = document.getElementById("roi-cost-val");
    if (orderBadge) orderBadge.textContent = (dailyOrders >= 1000000 ? (dailyOrders/1000000).toFixed(1) + "M orders/day" : (dailyOrders/1000).toFixed(0) + "k orders/day");
    if (costBadge) costBadge.textContent = "₹" + refundCost;

    // Metrics based on verified models:
    // 6.43% base delayed orders rate
    const dailyDelayed = dailyOrders * 0.0643;
    // 90.68% caught by PromiseGuard at T* = 0.57
    const dailyIntercepted = dailyDelayed * 0.9068;
    // Estimated 65% reduction in customer support grievances & refunds through proactive notification/voucher
    const grievancesAvoided = dailyIntercepted * 0.65;
    
    // Financial calculations
    const annualSavingsINR = grievancesAvoided * refundCost * 365;
    const annualSavingsCrores = (annualSavingsINR / 10000000).toFixed(2);
    const annualSavingsUSD = (annualSavingsINR / (86.5 * 1000000)).toFixed(2); // ~86.5 INR/USD

    const elSavings = document.getElementById("roi-annual-savings");
    const elIntercepted = document.getElementById("roi-intercepted-daily");
    const elUsd = document.getElementById("roi-usd-savings");

    if (elSavings) elSavings.textContent = "₹" + annualSavingsCrores + " Cr";
    if (elUsd) elUsd.textContent = "≈ $" + annualSavingsUSD + "M USD / year";
    if (elIntercepted) elIntercepted.textContent = Math.round(dailyIntercepted).toLocaleString() + " late orders/day";
  }

  orderSlider.addEventListener("input", recalculateRoi);
  costSlider.addEventListener("input", recalculateRoi);
  recalculateRoi();
}

// 5. 1-Click Scenario Preset Switcher
function applyScenarioPreset(type) {
  const setVal = (id, val) => {
    const el = document.getElementById(id);
    if (el) {
      el.value = val;
      const badge = document.getElementById(id + "-val");
      if (badge) {
        let suffix = "";
        if (id === "distance_km") suffix = " km";
        else if (id === "order_hour") suffix = ":00";
        else if (id === "Delivery_person_Age") suffix = " yrs";
        badge.textContent = val + suffix;
      }
    }
  };

  if (type === "lunch") {
    setVal("distance_km", 5.0);
    setVal("Road_traffic_density", "Medium");
    setVal("Weather_conditions", "Sunny");
    setVal("pickup_prep_delay_min", "10");
    setVal("Delivery_person_Ratings", 4.8);
    setVal("multiple_deliveries", "0.0");
    setVal("order_hour", 13);
    setVal("Vehicle_condition", "2");
  } else if (type === "monsoon") {
    setVal("distance_km", 9.0);
    setVal("Road_traffic_density", "Jam");
    setVal("Weather_conditions", "Stormy");
    setVal("pickup_prep_delay_min", "15");
    setVal("Delivery_person_Ratings", 4.3);
    setVal("multiple_deliveries", "2.0");
    setVal("order_hour", 20);
    setVal("Vehicle_condition", "1");
  } else if (type === "midnight") {
    setVal("distance_km", 3.0);
    setVal("Road_traffic_density", "Low");
    setVal("Weather_conditions", "Sunny");
    setVal("pickup_prep_delay_min", "5");
    setVal("Delivery_person_Ratings", 4.9);
    setVal("multiple_deliveries", "0.0");
    setVal("order_hour", 23);
    setVal("Vehicle_condition", "2");
  } else if (type === "suburban") {
    setVal("distance_km", 16.0);
    setVal("Road_traffic_density", "High");
    setVal("Weather_conditions", "Fog");
    setVal("pickup_prep_delay_min", "10");
    setVal("Delivery_person_Ratings", 4.6);
    setVal("multiple_deliveries", "1.0");
    setVal("order_hour", 19);
    setVal("Vehicle_condition", "1");
  }
  runPredictionAjax();
}

// 6. Live Rotating Order Telemetry Ticker
function initLiveOrderTicker() {
  const tickerEl = document.getElementById("live-order-ticker");
  if (!tickerEl) return;

  const orders = [
    "🛵 <strong>Order #ZOM-4829</strong> (Bangalore Koramangala) • Predicted: <strong>26 min</strong> • Window: [22–30 min] • Status: <span class=\"badge badge-success\">On-Time 🟢</span>",
    "🛵 <strong>Order #SWG-9104</strong> (Mumbai Bandra West) • Predicted: <strong>41 min</strong> • Window: [35–48 min] • Status: <span class=\"badge badge-warning\">Traffic Jam (Auto-Buffer +6m) 🟡</span>",
    "🛵 <strong>Order #ZPT-3112</strong> (Delhi Connaught Place) • Predicted: <strong>18 min</strong> • Window: [15–22 min] • Status: <span class=\"badge badge-success\">Smooth Transit 🟢</span>",
    "🛵 <strong>Order #BLN-7740</strong> (Hyderabad Hitec City) • Predicted: <strong>52 min</strong> • Window: [44–58 min] • Status: <span class=\"badge badge-danger\">Kitchen Surge Delay (₹30 Proactive Voucher Dispatched) 🔴</span>",
    "🛵 <strong>Order #SWG-2081</strong> (Pune Kothrud) • Predicted: <strong>23 min</strong> • Window: [20–27 min] • Status: <span class=\"badge badge-success\">On-Time 🟢</span>"
  ];

  let idx = 0;
  setInterval(() => {
    idx = (idx + 1) % orders.length;
    tickerEl.innerHTML = orders[idx];
  }, 4500);
}
