/* AgriSense static site.
 * Irrigation seasons are precomputed in Python (agrisense/export_web.py -> web/data/irrigation.json). */
(() => {
  "use strict";

  const RL = "Q-learning agent (RL)", THR = "Sensor threshold rule", FIX = "Fixed schedule", RAIN = "Rainfed (no irrigation)";
  const COLORS = { [RL]: "#2A78D6", [THR]: "#EB6834", [FIX]: "#1BAF7A", [RAIN]: "#EDA100" };
  const SHORT = { [RL]: "RL agent", [THR]: "Sensor rule", [FIX]: "Fixed schedule", [RAIN]: "Rainfed" };
  const INK = "#16201A", INK2 = "#55605A", GRID = "#E9EEE7";
  const RB_COLORS = { 0: "#F4F8FC", 10: "#CDE2FB", 20: "#9EC5F4", 30: "#5598E7", 40: "#256ABF" };

  const state = { tab: "overview", crop: "Maize", city: "Jodhpur", season: null, year: 2023,
                  show: new Set([RL, THR]), rbCrop: "Maize", rbStage: 2, rbEt0: 2 };
  let D = null;                 // irrigation data
  const charts = {};
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const fmt = (v, d = 0) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : Number(v).toFixed(d);

  /* ------------------------------------------------------------------ tabs */
  function moveInk() {
    const btn = $(`.tabs button[data-tab="${state.tab}"]`), ink = $(".tab-ink");
    if (!btn || !ink) return;
    ink.style.width = btn.offsetWidth + "px";
    ink.style.transform = `translateX(${btn.offsetLeft}px)`;
  }
  function selectTab(name) {
    name = { agent: "irrigation", results: "irrigation", about: "overview", leaf: "overview" }[name] || name;   // old links
    if (!$(`[data-panel="${name}"]`)) name = "overview";
    state.tab = name;
    $$(".tabs button").forEach(b => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
    $$("[data-panel]").forEach(p => { p.hidden = p.dataset.panel !== name; });
    moveInk();
    if (location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
    window.scrollTo({ top: 0, behavior: "smooth" });
    requestAnimationFrame(() => {
      if (name === "irrigation") { renderIrrigation(); renderAgent(); renderResults(); }
      Object.values(charts).forEach(c => c && c.resize());
    });
  }
  document.addEventListener("click", e => {
    const t = e.target.closest("[data-tab],[data-goto]");
    if (!t) return;
    e.preventDefault();
    selectTab(t.dataset.tab || t.dataset.goto);
  });
  window.addEventListener("resize", () => { moveInk(); Object.values(charts).forEach(c => c && c.resize()); });
  window.addEventListener("hashchange", () => { const t = location.hash.slice(1); if (D && t && t !== state.tab) selectTab(t); });

  function chart(id) {
    const el = document.getElementById(id);
    if (!charts[id]) charts[id] = echarts.init(el, null, { renderer: "canvas" });
    return charts[id];
  }
  function seg(container, items, active, onPick, label = x => x) {
    container.innerHTML = "";
    items.forEach(it => {
      const b = document.createElement("button");
      b.type = "button"; b.textContent = label(it);
      b.setAttribute("aria-pressed", String(it === active));
      b.addEventListener("click", () => {
        $$("button", container).forEach(x => x.setAttribute("aria-pressed", "false"));
        b.setAttribute("aria-pressed", "true");
        onPick(it);
      });
      container.appendChild(b);
    });
  }
  const axisStyle = {
    axisLine: { lineStyle: { color: GRID } }, axisTick: { show: false },
    axisLabel: { color: INK2, fontSize: 11 }, splitLine: { lineStyle: { color: GRID } },
    nameTextStyle: { color: INK2, fontSize: 11 },
  };

  /* ------------------------------------------------------------------ irrigation */
  function seasonKey() { return `${state.crop}|${state.season}|${state.city}|${state.year}`; }

  function buildControls() {
    const crops = Object.keys(D.crops);
    seg($("#crop-seg"), crops, state.crop, c => {
      state.crop = c;
      if (!(state.season in D.crops[c].seasons)) state.season = Object.keys(D.crops[c].seasons)[0];
      buildSeasonSeg(); renderIrrigation(true);
    });
    buildSeasonSeg();

    const maxRain = Math.max(...Object.values(D.cities).flatMap(c => c.monthly_rain));
    const wrap = $("#city-cards"); wrap.innerHTML = "";
    Object.entries(D.cities).forEach(([name, c]) => {
      const b = document.createElement("button");
      b.type = "button"; b.className = "city"; b.setAttribute("aria-pressed", String(name === state.city));
      const bars = c.monthly_rain.map((v, i) => {
        const h = Math.max(1, 30 * v / maxRain);
        return `<rect x="${i * 6}" y="${32 - h}" width="4" height="${h}" rx="1" fill="#5598E7"/>`;
      }).join("");
      b.innerHTML = `<b>${name}</b><svg viewBox="0 0 72 34" aria-hidden="true">${bars}</svg>
        <small>${c.climate.split(";")[0]} · ${fmt(c.annual_rain)} mm/yr</small>`;
      b.title = c.climate + " (bars: average monthly rain)";
      b.addEventListener("click", () => {
        state.city = name;
        $$(".city", wrap).forEach(x => x.setAttribute("aria-pressed", String(x === b)));
        renderIrrigation(true);
      });
      wrap.appendChild(b);
    });

    const yr = $("#year-range");
    yr.value = state.year;
    yr.addEventListener("input", () => { state.year = +yr.value; renderIrrigation(true); });

    const chips = $("#policy-chips"); chips.innerHTML = "";
    D.policies.forEach(p => {
      const b = document.createElement("button");
      b.type = "button"; b.className = "chip"; b.setAttribute("aria-pressed", String(state.show.has(p)));
      b.innerHTML = `<i style="background:${COLORS[p]}"></i>${SHORT[p]}`;
      b.addEventListener("click", () => {
        state.show.has(p) ? state.show.delete(p) : state.show.add(p);
        if (!state.show.size) state.show.add(RL);
        $$(".chip", chips).forEach((x, i) => x.setAttribute("aria-pressed", String(state.show.has(D.policies[i]))));
        renderIrrigation(true);
      });
      chips.appendChild(b);
    });
    $("#replay").addEventListener("click", () => { charts["chart-season"]?.clear(); renderIrrigation(false, 2600); });
  }
  function buildSeasonSeg() {
    const seasons = Object.keys(D.crops[state.crop].seasons);
    if (!seasons.includes(state.season)) state.season = seasons[0];
    seg($("#season-seg"), seasons, state.season, s => { state.season = s; renderIrrigation(true); });
  }

  function kpi(label, value, delta, cls) {
    return `<div class="kpi"><span>${label}</span><b>${value}</b>${delta ? `<em class="${cls}">${delta}</em>` : ""}</div>`;
  }

  function renderIrrigation(fast = false, duration = 900) {
    if (!D) return;
    const e = D.seasons[seasonKey()];
    const crop = D.crops[state.crop];
    if (!e) return;
    const P = e.policies, rl = P[RL], thr = P[THR], fix = P[FIX];
    $("#year-val").textContent = state.year;
    const pill = $("#year-pill");
    pill.textContent = e.test ? "unseen test year" : "training year";
    pill.className = "pill" + (e.test ? " test" : "");
    const sow = new Date(e.sow + "T00:00:00");
    $("#season-title").textContent = `${state.crop} · ${state.city} · sown ${sow.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" })}`;

    const rainTotal = e.rain.reduce((a, b) => a + (b || 0), 0);
    const dFix = rl.water - fix.water, dThr = rl.yield_pct - thr.yield_pct;
    $("#kpis").innerHTML =
      kpi("RL agent water used", `${fmt(rl.water)} mm`, `${dFix <= 0 ? "" : "+"}${fmt(dFix)} mm vs fixed schedule`, dFix <= 0 ? "good" : "bad") +
      kpi("RL agent yield", `${fmt(rl.yield_pct, 1)}%`, `${dThr >= 0 ? "+" : ""}${fmt(dThr, 1)} pts vs sensor rule`, dThr >= -0.05 ? "good" : (dThr > -2 ? "neutral" : "bad")) +
      kpi("Sensor rule water used", `${fmt(thr.water)} mm`, `${fmt(thr.events)} irrigations vs ${fmt(rl.events)} for RL`, "neutral") +
      kpi("Season rainfall", `${fmt(rainTotal)} mm`, `${crop.season_length}-day season`, "neutral");

    const days = e.rain.map((_, i) => i);
    const shown = D.policies.filter(p => state.show.has(p));
    const series = [];
    shown.forEach(p => series.push({
      name: SHORT[p], type: "line", xAxisIndex: 0, yAxisIndex: 0, data: P[p].soil, showSymbol: false,
      lineStyle: { width: 2.2, color: COLORS[p] }, itemStyle: { color: COLORS[p] }, smooth: 0.15,
      emphasis: { focus: "series" },
      markLine: p === shown[0] ? { silent: true, symbol: "none", lineStyle: { color: "#C0392B", type: "dashed", width: 1.2 },
        label: { formatter: "stress line", color: "#C0392B", fontSize: 11, position: "insideStartTop" },
        data: [{ yAxis: crop.stress_line_pct }] } : undefined,
    }));
    shown.forEach(p => series.push({
      name: SHORT[p], type: "bar", xAxisIndex: 1, yAxisIndex: 1, data: P[p].irr.map(v => v || null),
      itemStyle: { color: COLORS[p], borderRadius: [3, 3, 0, 0] }, barMaxWidth: 6, barGap: "10%",
    }));
    series.push({ name: "Rain", type: "bar", xAxisIndex: 2, yAxisIndex: 2, data: e.rain,
      itemStyle: { color: "#86B6EF", borderRadius: [2, 2, 0, 0] }, barMaxWidth: 5 });

    const c = chart("chart-season");
    c.setOption({
      animationDuration: duration, animationEasing: "cubicOut", animationDurationUpdate: fast ? 500 : duration,
      color: shown.map(p => COLORS[p]),
      legend: { top: 0, right: 0, textStyle: { color: INK2 }, icon: "roundRect", itemWidth: 12, itemHeight: 8,
        data: [...shown.map(p => SHORT[p]), "Rain"] },
      axisPointer: { link: [{ xAxisIndex: "all" }], lineStyle: { color: "#9AA59D" } },
      tooltip: { trigger: "axis", backgroundColor: "#fff", borderColor: GRID, textStyle: { color: INK, fontSize: 12 },
        formatter: ps => {
          if (!ps.length) return "";
          const day = ps[0].dataIndex, date = new Date(sow.getTime() + day * 864e5);
          const lines = [`<b>Day ${day}</b> · ${date.toLocaleDateString("en-GB", { day: "numeric", month: "short" })}`];
          shown.forEach(p => lines.push(`<span class="swatch" style="background:${COLORS[p]}"></span>${SHORT[p]}: soil ${P[p].soil[day]}%` +
            (P[p].irr[day] ? `, <b>irrigate ${P[p].irr[day]} mm</b>` : "")));
          lines.push(`<span class="swatch" style="background:#86B6EF"></span>Rain ${fmt(e.rain[day], 1)} mm · ET0 ${fmt(e.et0[day], 1)} mm`);
          return lines.join("<br>");
        } },
      grid: [{ left: 62, right: 16, top: 34, height: "50%" }, { left: 62, right: 16, top: "64%", height: "13%" },
             { left: 62, right: 16, top: "81%", height: "11%" }],
      xAxis: [0, 1, 2].map(i => ({ type: "category", gridIndex: i, data: days, boundaryGap: true, ...axisStyle,
        axisLabel: { ...axisStyle.axisLabel, show: i === 2, interval: 9 }, splitLine: { show: false },
        name: i === 2 ? "days after sowing" : "", nameLocation: "middle", nameGap: 26 })),
      yAxis: [
        { gridIndex: 0, min: 0, max: 100, name: "soil water %", nameLocation: "middle", nameRotate: 90, nameGap: 40, ...axisStyle },
        { gridIndex: 1, min: 0, max: 60, name: "irrigation", nameLocation: "middle", nameRotate: 90, nameGap: 40, splitNumber: 2, ...axisStyle },
        { gridIndex: 2, min: 0, name: "rain", nameLocation: "middle", nameRotate: 90, nameGap: 40, splitNumber: 2, ...axisStyle },
      ],
      dataZoom: [{ type: "inside", xAxisIndex: [0, 1, 2] }],
      series,
    }, { notMerge: true });

    const rows = D.policies.map(p => {
      const v = P[p];
      return `<tr class="${p === RL ? "hl" : ""}"><td><span class="swatch" style="background:${COLORS[p]}"></span>${p}</td>
        <td class="num">${fmt(v.water)}</td><td class="num">${v.events}</td><td class="num">${fmt(v.wasted, 1)}</td>
        <td class="num">${v.stress_days}</td><td class="num">${fmt(v.yield_pct, 1)}%</td><td class="num">${fmt(v.wp, 2)}</td></tr>`;
    }).join("");
    $("#season-table").innerHTML = `<thead><tr><th>Policy</th><th class="num">Water (mm)</th><th class="num">Irrigations</th>
      <th class="num">Wasted (mm)</th><th class="num">Stress days</th><th class="num">Yield</th><th class="num">kg / m³</th></tr></thead><tbody>${rows}</tbody>`;
  }

  /* ------------------------------------------------------------------ agent */
  let agentBuilt = false;
  function renderAgent() {
    if (!D) return;
    if (!agentBuilt) {
      agentBuilt = true;
      seg($("#rb-crop"), Object.keys(D.crops), state.rbCrop, c => { state.rbCrop = c; renderAgent(); });
      seg($("#rb-stage"), [0, 1, 2, 3], state.rbStage, s => { state.rbStage = s; renderAgent(); }, s => D.stages[s]);
      seg($("#rb-et0"), [0, 1, 2], state.rbEt0, s => { state.rbEt0 = s; renderAgent(); }, s => ["low", "medium", "high"][s]);
    }
    const grid = D.rulebook[state.rbCrop][state.rbStage][state.rbEt0];
    const rb = $("#rulebook");
    if (!rb.children.length) {
      let html = `<div class="h"></div>` + D.rain_labels.map(l => `<div class="h">rain ${l}</div>`).join("");
      for (let r = 9; r >= 0; r--) {
        html += `<div class="r">soil ${r * 10}–${r * 10 + 10}% dry</div>`;
        for (let k = 0; k < 3; k++) html += `<div class="c" data-r="${r}" data-k="${k}"></div>`;
      }
      rb.innerHTML = html;
    }
    $$(".c", rb).forEach(cell => {
      const v = grid[+cell.dataset.r][+cell.dataset.k];
      cell.textContent = v === null ? "rare" : `${v} mm`;
      cell.style.backgroundColor = v === null ? "#F0EFEA" : RB_COLORS[v];
      cell.style.color = v === null ? "#9A9A93" : (v >= 30 ? "#fff" : INK);
    });

    const t = D.training[state.rbCrop];
    chart("chart-training").setOption({
      animationDuration: 900, grid: { left: 56, right: 28, top: 16, bottom: 36 },
      tooltip: { trigger: "axis", valueFormatter: v => fmt(v) },
      xAxis: { type: "value", name: "training seasons", nameLocation: "middle", nameGap: 24, ...axisStyle, splitLine: { show: false } },
      yAxis: { type: "value", name: "season reward", ...axisStyle, scale: true },
      series: [{ type: "line", showSymbol: false, data: t.episode.map((e, i) => [e, t.reward[i]]),
        lineStyle: { color: COLORS[RL], width: 2 }, areaStyle: { color: "rgba(42,120,214,.08)" }, smooth: true }],
    }, { notMerge: true });
  }

  /* ------------------------------------------------------------------ results */
  let resultsBuilt = false;
  function renderResults() {
    if (!D || resultsBuilt) { Object.values(charts).forEach(c => c && c.resize()); return; }
    resultsBuilt = true;
    const S = D.summary.by_policy;
    $("#n-test").textContent = D.summary.n_test_seasons;
    const names = S.map(r => SHORT[r.policy]);
    const bar = (id, key, title, unit, digits) => chart(id).setOption({
      title: { text: title, left: 0, textStyle: { color: INK, fontSize: 13, fontWeight: 600 } },
      grid: { left: 110, right: 48, top: 36, bottom: 16 }, animationDuration: 1100,
      tooltip: { trigger: "item", valueFormatter: v => fmt(v, digits) + unit },
      xAxis: { type: "value", ...axisStyle, axisLabel: { show: false }, splitLine: { show: false } },
      yAxis: { type: "category", data: names, inverse: true, ...axisStyle },
      series: [{ type: "bar", barWidth: 22, data: S.map(r => ({ value: r[key], itemStyle: { color: COLORS[r.policy], borderRadius: [0, 6, 6, 0] } })),
        label: { show: true, position: "right", color: INK, formatter: p => fmt(p.value, digits) + unit } }],
    });
    bar("chart-water", "avg_irrigation_mm", "Irrigation water per season", " mm", 0);
    bar("chart-yield", "avg_relative_yield_pct", "Yield (% of maximum)", "%", 1);

    $("#summary-table").innerHTML = `<thead><tr><th>Policy</th><th class="num">Water (mm)</th><th class="num">Irrigations</th>
      <th class="num">Wasted (mm)</th><th class="num">Stress days</th><th class="num">Yield</th><th class="num">kg / m³</th></tr></thead><tbody>` +
      S.map(r => `<tr class="${r.policy === RL ? "hl" : ""}"><td><span class="swatch" style="background:${COLORS[r.policy]}"></span>${r.policy}</td>
        <td class="num">${fmt(r.avg_irrigation_mm)}</td><td class="num">${fmt(r.avg_irrigation_events, 1)}</td>
        <td class="num">${fmt(r.avg_wasted_mm, 1)}</td><td class="num">${fmt(r.avg_stress_days, 1)}</td>
        <td class="num">${fmt(r.avg_relative_yield_pct, 1)}%</td><td class="num">${fmt(r.water_productivity_kg_per_m3, 2)}</td></tr>`).join("") + "</tbody>";

    const cities = Object.keys(D.cities), pols = [RL, THR, FIX];
    chart("chart-city").setOption({
      animationDuration: 1100, legend: { top: 0, right: 0, textStyle: { color: INK2 }, icon: "roundRect" },
      tooltip: { trigger: "axis", valueFormatter: v => fmt(v) + " mm" },
      grid: { left: 52, right: 12, top: 36, bottom: 28 },
      xAxis: { type: "category", data: cities, ...axisStyle, splitLine: { show: false } },
      yAxis: { type: "value", name: "mm / season", ...axisStyle },
      series: pols.map(p => ({ name: SHORT[p], type: "bar", barGap: "12%", barMaxWidth: 46,
        itemStyle: { color: COLORS[p], borderRadius: [5, 5, 0, 0] },
        data: cities.map(c => D.summary.by_city.find(r => r.city === c && r.policy === p).water) })),
    });
  }
  /* ------------------------------------------------------------------ boot */
  async function boot() {
    D = await fetch("web/data/irrigation.json").then(r => r.json());
    const rl = D.summary.by_policy.find(r => r.policy === RL), fx = D.summary.by_policy.find(r => r.policy === FIX);
    $("#ov-save").textContent = `${fmt(100 * (1 - rl.avg_irrigation_mm / fx.avg_irrigation_mm))}%`;
    $("#ov-yield").textContent = `${fmt(rl.avg_relative_yield_pct, 1)}%`;
    const th = D.summary.by_policy.find(r => r.policy === THR);
    $("#ov-thr").textContent = `${fmt(100 * (1 - rl.avg_irrigation_mm / th.avg_irrigation_mm), 1)}%`;
    $("#ov-n").textContent = D.summary.n_test_seasons;
    state.season = Object.keys(D.crops[state.crop].seasons)[0];
    buildControls();
    selectTab((location.hash || "#overview").slice(1));
    if (document.fonts) document.fonts.ready.then(moveInk);
  }
  boot().catch(err => { console.error(err); document.body.insertAdjacentHTML("afterbegin",
    `<p style="padding:12px;background:#FCE9E4;margin:0">Could not load data: ${err.message}. If you opened the file directly, serve the folder with <code>python -m http.server</code>.</p>`); });
})();
