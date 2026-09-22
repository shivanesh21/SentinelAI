(function () {
  "use strict";

  var API_BASE = "http://localhost:8000/api";
  var REFRESH_MS = 15000;

  var conn = document.getElementById("conn");
  var cards = document.getElementById("cards");

  function setConn(state, label) {
    conn.textContent = label;
    conn.className = "conn " + state;
  }

  function getJSON(path) {
    return fetch(API_BASE + path, { cache: "no-store" }).then(function (res) {
      if (!res.ok) {
        return res.text().then(function (t) { throw new Error(t); });
      }
      return res.json();
    });
  }

  function fmtNumber(n) {
    if (n === null || n === undefined) return "—";
    return n.toLocaleString();
  }

  function renderHealth(health) {
    var card = cards.children[0];
    var value = document.getElementById("v-api");
    value.textContent = health && health.status === "ok" ? "UP" : "DOWN";
    card.className = "card status " + (health && health.status === "ok" ? "ok" : "err");
    document.getElementById("s-rows").textContent = health ? health.service : "unreachable";
  }

  function renderOverview(overview) {
    if (!overview || !overview.available) {
      setConn("ok", "connected — no dataset yet");
      return;
    }
    document.getElementById("v-rows").textContent = fmtNumber(overview.n_rows);
    document.getElementById("s-rows").textContent =
      overview.n_features + " features x " + overview.n_labels + " labels";
    document.getElementById("v-services").textContent = fmtNumber(overview.n_services);
    document.getElementById("v-window").textContent = overview.prediction_window_min;

    var chips = document.getElementById("service-chips");
    chips.innerHTML = "";
    (overview.services || []).slice().sort().forEach(function (s) {
      var el = document.createElement("span");
      el.className = "chip";
      el.textContent = s;
      chips.appendChild(el);
    });
    if (!(overview.services || []).length) {
      chips.innerHTML = '<span class="chip muted">none registered</span>';
    }

    var tbody = document.querySelector("#split-table tbody");
    tbody.innerHTML = "";
    var counts = (overview.splits && overview.splits.counts) || {};
    var rates = (overview.splits && overview.splits.positive_rates) || {};
    var total = 0;
    ["train", "val", "test"].forEach(function (name) {
      total += counts[name] || 0;
    });
    ["train", "val", "test"].forEach(function (name) {
      var rate = rates[name] && rates[name].failure_in_next_10min;
      var share = total ? (((counts[name] || 0) / total) * 100).toFixed(1) + "%" : "—";
      var tr = document.createElement("tr");
      tr.innerHTML =
        "<td>" + name + "</td>" +
        "<td>" + fmtNumber(counts[name]) + " (" + share + ")</td>" +
        "<td>" + (rate === undefined ? "—" : (rate * 100).toFixed(2) + "%") + "</td>";
      tbody.appendChild(tr);
    });
  }

  function refresh() {
    return Promise.all([getJSON("/health"), getJSON("/overview")])
      .then(function (results) {
        setConn("ok", "connected");
        renderHealth(results[0]);
        renderOverview(results[1]);
      })
      .catch(function (err) {
        setConn("err", "backend unreachable — start uvicorn and retry");
        var value = document.getElementById("v-api");
        value.textContent = "DOWN";
        cards.children[0].className = "card status err";
      });
  }

  refresh();
  setInterval(refresh, REFRESH_MS);
})();