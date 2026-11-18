// PromiseGuard Interactive Web Client Logic

document.addEventListener("DOMContentLoaded", () => {
  initSliderBadges();
  initPredictionForm();
  initSimulationSlider();
  initVivaAccordions();
});

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

  // Also auto-recalculate on change if desired
  form.addEventListener("change", (e) => {
    // Only auto-trigger if not rapid sliding
    if (e.target.type !== "range") {
      runPredictionAjax();
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
  if (statusEl) statusEl.textContent = "Calculating ML inferences...";

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
    elRiskBadge.className = "pill " + (
      data.risk_class === "danger" ? "pill-red" :
      data.risk_class === "warning" ? "pill-amber" : "pill-green"
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
  // Max scale: 70 min
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

// 3. Proactive Intervention Simulation Interactive Slider
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

  setEl("sim-alerts-sent", `${d.alerts_sent.toLocaleString()} (${d.alert_rate_pct}%)`);
  setEl("sim-recall", `${d.recall_pct}%`);
  setEl("sim-precision", `${d.precision_pct}%`);
  setEl("sim-cost-index", d.illustrative_cost_index);
  setEl("sim-delays-captured", d.true_late_captured);
  setEl("sim-missed-delays", d.missed_late_orders);
  setEl("sim-false-alarms", d.false_alarms);
}

// 4. Viva Voce Accordion
function initVivaAccordions() {
  const questions = document.querySelectorAll(".viva-question");
  questions.forEach(q => {
    q.addEventListener("click", () => {
      const answer = q.nextElementSibling;
      if (answer) {
        const isShown = answer.style.display === "block";
        answer.style.display = isShown ? "none" : "block";
        const icon = q.querySelector(".viva-icon");
        if (icon) icon.textContent = isShown ? "▼" : "▲";
      }
    });
  });
}
