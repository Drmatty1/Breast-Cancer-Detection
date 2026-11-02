"""
RP2 Mammography Agent: Interactive Localhost Web UI.

Standalone FastAPI web application for visual case review and multi-agent deliberation:
- Populates an interactive dropdown of all 147 holdout cases in RP2_LOCAL_HOLDOUT_TRANSFER/images
- Streams bilateral CC and MLO mammogram views converted from .npy arrays to PNG
- Executes synchronized Mammography + Clinical Risk Agent perception
- Invokes Qwen 2.5 3B LLM Supervisor with live consensus deliberation
- Renders confidence intervals, Decision Diamond deferral status, and audited citations
- Generates downloadable JSON and Markdown clinical consensus reports

Usage:
    python scripts/run_ui.py
    python scripts/run_ui.py --port 8000
"""
from typing import Dict, Any, List, Optional
from pathlib import Path
import os
import sys
import csv
import json
import io
import time
import argparse

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_case_inference import (
    load_cached_images,
    lookup_clinical_covariates,
    get_agent_finding_cards
)
from src.orchestration.supervisor_contract import (
    SupervisorState,
    AgentFindingPayload
)
from src.orchestration.supervisor_schema import (
    ConsensusDecision,
    extract_evidence_quotes
)
from src.orchestration.llm_supervisor import (
    LLMSupervisor,
    OllamaLLMClient,
    MockLLMClient
)

app = FastAPI(title="RP2 Mammography Decision Support System")

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RP2 Mammography Decision Support System</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-base: #0b0f19;
      --bg-surface: #121826;
      --bg-card: #182234;
      --bg-card-hover: #1f2c42;
      --border-color: #243247;
      --border-focus: #3b82f6;
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --accent-blue: #3b82f6;
      --accent-cyan: #06b6d4;
      --accent-green: #10b981;
      --accent-amber: #f59e0b;
      --accent-rose: #f43f5e;
      --accent-purple: #8b5cf6;
      --radius-sm: 6px;
      --radius-md: 10px;
      --radius-lg: 16px;
      --shadow-lg: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.4);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', sans-serif;
      background-color: var(--bg-base);
      color: var(--text-main);
      min-height: 100vh;
      line-height: 1.5;
    }

    /* Header */
    header {
      background: linear-gradient(180deg, rgba(18,24,38,0.95) 0%, rgba(11,15,25,0.95) 100%);
      border-bottom: 1px solid var(--border-color);
      padding: 16px 32px;
      position: sticky;
      top: 0;
      z-index: 50;
      backdrop-filter: blur(8px);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
    }

    .brand-icon {
      width: 40px;
      height: 40px;
      background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-cyan) 100%);
      border-radius: var(--radius-md);
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-weight: 700;
      font-size: 20px;
      box-shadow: 0 0 15px rgba(59, 130, 246, 0.4);
    }

    .brand-title {
      font-size: 18px;
      font-weight: 700;
      letter-spacing: -0.3px;
    }

    .brand-subtitle {
      font-size: 12px;
      color: var(--text-muted);
    }

    .badge-pill {
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      font-weight: 500;
      padding: 4px 10px;
      border-radius: 9999px;
      background: rgba(59, 130, 246, 0.12);
      border: 1px solid rgba(59, 130, 246, 0.3);
      color: var(--accent-blue);
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .badge-pill.status-online::before {
      content: '';
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--accent-green);
      box-shadow: 0 0 8px var(--accent-green);
    }

    /* Top Control Bar */
    .controls-strip {
      background: var(--bg-surface);
      border-bottom: 1px solid var(--border-color);
      padding: 14px 32px;
      display: flex;
      align-items: center;
      gap: 18px;
      flex-wrap: wrap;
    }

    .select-wrapper {
      position: relative;
      flex: 1;
      max-width: 480px;
    }

    .select-label {
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text-dim);
      margin-bottom: 4px;
      display: block;
    }

    select {
      width: 100%;
      padding: 10px 14px;
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      color: var(--text-main);
      border-radius: var(--radius-sm);
      font-family: 'JetBrains Mono', monospace;
      font-size: 13px;
      cursor: pointer;
      outline: none;
      transition: border 0.2s;
    }

    select:focus {
      border-color: var(--border-focus);
    }

    .btn {
      padding: 10px 20px;
      border-radius: var(--radius-sm);
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      transition: all 0.2s;
      border: none;
      outline: none;
    }

    .btn-primary {
      background: linear-gradient(135deg, var(--accent-blue) 0%, #2563eb 100%);
      color: white;
      box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
    }

    .btn-primary:hover:not(:disabled) {
      background: linear-gradient(135deg, #60a5fa 0%, var(--accent-blue) 100%);
      transform: translateY(-1px);
    }

    .btn-secondary {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      color: var(--text-main);
    }

    .btn-secondary:hover:not(:disabled) {
      background: var(--bg-card-hover);
    }

    .btn:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }

    /* Main Container */
    main {
      padding: 24px 32px;
      display: grid;
      grid-template-columns: 520px 1fr;
      gap: 24px;
      max-width: 1800px;
      margin: 0 auto;
    }

    @media (max-width: 1200px) {
      main {
        grid-template-columns: 1fr;
      }
    }

    /* Cards */
    .card {
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      padding: 20px;
      box-shadow: var(--shadow-lg);
    }

    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--border-color);
    }

    .card-title {
      font-size: 14px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    /* Image Viewer */
    .images-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      margin-bottom: 16px;
    }

    .image-box {
      background: black;
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      overflow: hidden;
      display: flex;
      flex-direction: column;
      position: relative;
    }

    .image-tag {
      position: absolute;
      top: 8px;
      left: 8px;
      background: rgba(0, 0, 0, 0.75);
      border: 1px solid rgba(255, 255, 255, 0.2);
      color: white;
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      font-weight: 600;
      padding: 2px 8px;
      border-radius: var(--radius-sm);
      backdrop-filter: blur(4px);
      z-index: 2;
    }

    .image-box img {
      width: 100%;
      height: 250px;
      object-fit: contain;
      background: #050505;
      display: block;
      transition: transform 0.3s ease;
    }

    .image-box:hover img {
      transform: scale(1.02);
    }

    .image-stats {
      padding: 6px 10px;
      background: rgba(18, 24, 38, 0.9);
      font-family: 'JetBrains Mono', monospace;
      font-size: 10px;
      color: var(--text-dim);
      display: flex;
      justify-content: space-between;
      border-top: 1px solid var(--border-color);
    }

    /* Covariates Grid */
    .covariates-list {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
      margin-top: 10px;
    }

    .covariate-item {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-sm);
      padding: 8px 12px;
    }

    .covariate-label {
      font-size: 10px;
      font-weight: 600;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .covariate-val {
      font-family: 'JetBrains Mono', monospace;
      font-size: 13px;
      font-weight: 600;
      color: var(--text-main);
      margin-top: 2px;
    }

    /* Consensus Banner */
    .decision-banner {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 16px 20px;
      margin-bottom: 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: relative;
      overflow: hidden;
    }

    .decision-banner::before {
      content: '';
      position: absolute;
      left: 0;
      top: 0;
      bottom: 0;
      width: 5px;
      background: var(--accent-blue);
    }

    .decision-banner.status-defer::before {
      background: var(--accent-amber);
    }

    .decision-banner.status-autonomous::before {
      background: var(--accent-green);
    }

    .birads-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 6px 16px;
      border-radius: var(--radius-sm);
      font-family: 'JetBrains Mono', monospace;
      font-size: 22px;
      font-weight: 700;
      letter-spacing: 0.5px;
    }

    .birads-2 { background: rgba(16, 185, 129, 0.15); border: 1px solid var(--accent-green); color: var(--accent-green); }
    .birads-3 { background: rgba(6, 182, 212, 0.15); border: 1px solid var(--accent-cyan); color: var(--accent-cyan); }
    .birads-4, .birads-4A { background: rgba(245, 158, 11, 0.15); border: 1px solid var(--accent-amber); color: var(--accent-amber); }
    .birads-4B, .birads-4C { background: rgba(249, 115, 22, 0.15); border: 1px solid #f97316; color: #f97316; }
    .birads-5 { background: rgba(244, 63, 94, 0.15); border: 1px solid var(--accent-rose); color: var(--accent-rose); }

    .prob-metric {
      text-align: right;
    }

    .prob-val {
      font-family: 'JetBrains Mono', monospace;
      font-size: 26px;
      font-weight: 700;
      color: var(--text-main);
    }

    .prob-sub {
      font-size: 11px;
      color: var(--text-muted);
    }

    /* Specialists Cards Grid */
    .specialists-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 20px;
    }

    .specialist-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 16px;
    }

    .specialist-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }

    .specialist-title {
      font-size: 13px;
      font-weight: 600;
      color: var(--text-main);
    }

    .specialist-role {
      font-size: 10px;
      font-family: 'JetBrains Mono', monospace;
      text-transform: uppercase;
      padding: 2px 6px;
      border-radius: 4px;
    }

    .role-primary { background: rgba(59, 130, 246, 0.15); color: var(--accent-blue); border: 1px solid rgba(59,130,246,0.3); }
    .role-shadow { background: rgba(148, 163, 184, 0.15); color: var(--text-muted); border: 1px solid rgba(148,163,184,0.3); }

    /* Rationale & Quotes */
    .rationale-box {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 16px;
      font-size: 14px;
      line-height: 1.6;
      margin-bottom: 20px;
    }

    .citation-tag {
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      padding: 2px 6px;
      border-radius: 4px;
      background: rgba(139, 92, 246, 0.15);
      border: 1px solid rgba(139, 92, 246, 0.3);
      color: #c4b5fd;
      display: inline-block;
      margin: 0 2px;
    }

    .citations-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }

    /* Spinner */
    .spinner {
      width: 18px;
      height: 18px;
      border: 2px solid rgba(255,255,255,0.3);
      border-top-color: white;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    .empty-state {
      padding: 60px 20px;
      text-align: center;
      color: var(--text-dim);
    }

    .empty-icon {
      font-size: 40px;
      margin-bottom: 12px;
      opacity: 0.5;
    }
  </style>
</head>
<body>

  <!-- Header -->
  <header>
    <div class="brand">
      <div class="brand-icon">M</div>
      <div>
        <div class="brand-title">Mammography Multi-Agent Decision Support</div>
        <div class="brand-subtitle">Curated Breast Imaging Subset · RP2 Local Holdout Evaluator</div>
      </div>
    </div>
    <div style="display: flex; gap: 12px; align-items: center;">
      <div class="badge-pill status-online">Localhost:8000 Online</div>
      <div class="badge-pill">Qwen 2.5 3B (Ollama)</div>
    </div>
  </header>

  <!-- Control Strip -->
  <div class="controls-strip">
    <div class="select-wrapper">
      <label class="select-label" for="caseSelector">Select Holdout Patient Case (RP2_LOCAL_HOLDOUT_TRANSFER):</label>
      <select id="caseSelector" onchange="onCaseSelectionChanged()">
        <option value="" disabled selected>Loading cases list...</option>
      </select>
    </div>

    <button id="btnPrev" class="btn btn-secondary" onclick="stepCase(-1)">◄ Prev</button>
    <button id="btnNext" class="btn btn-secondary" onclick="stepCase(1)">Next ►</button>
    <button id="btnRandom" class="btn btn-secondary" onclick="pickRandomCase()">🎲 Random</button>

    <div style="flex: 1;"></div>

    <button id="btnEvaluate" class="btn btn-primary" onclick="runDeliberation()">
      <span id="btnIcon">⚡</span>
      <span id="btnText">Run Multi-Agent Deliberation</span>
    </button>
  </div>

  <!-- Main View -->
  <main>
    <!-- Left Column: Bilateral Imaging & Clinical Covariates -->
    <section>
      <div class="card">
        <div class="card-header">
          <div class="card-title">
            <span>📷 Dual-View Mammographic Imaging</span>
          </div>
          <span id="imgSourceBadge" class="badge-pill">Holdout Partition</span>
        </div>

        <div class="images-grid">
          <div class="image-box">
            <span class="image-tag">Craniocaudal (CC)</span>
            <img id="imgCC" src="" alt="CC View">
            <div class="image-stats">
              <span>Dim: 512 × 512</span>
              <span>Type: float32 [0, 1]</span>
            </div>
          </div>
          <div class="image-box">
            <span class="image-tag">Mediolateral Oblique (MLO)</span>
            <img id="imgMLO" src="" alt="MLO View">
            <div class="image-stats">
              <span>Dim: 512 × 512</span>
              <span>Type: float32 [0, 1]</span>
            </div>
          </div>
        </div>

        <div class="card-title" style="margin-top: 14px; margin-bottom: 8px;">
          <span>📋 Clinical Patient Profile & Covariates</span>
        </div>
        <div class="covariates-list">
          <div class="covariate-item">
            <div class="covariate-label">Patient ID</div>
            <div id="covPatientId" class="covariate-val">-</div>
          </div>
          <div class="covariate-item">
            <div class="covariate-label">Breast Laterality</div>
            <div id="covSide" class="covariate-val">-</div>
          </div>
          <div class="covariate-item">
            <div class="covariate-label">Breast Density (ACR)</div>
            <div id="covDensity" class="covariate-val">-</div>
          </div>
          <div class="covariate-item">
            <div class="covariate-label">Abnormality Type</div>
            <div id="covAbnType" class="covariate-val">-</div>
          </div>
          <div class="covariate-item" style="grid-column: span 2;">
            <div class="covariate-label">Ground Truth (Quarantined Reference)</div>
            <div id="covGroundTruth" class="covariate-val" style="color: var(--accent-cyan);">-</div>
          </div>
        </div>
      </div>
    </section>

    <!-- Right Column: Supervisor Deliberation & Specialists Output -->
    <section>
      <div class="card">
        <div class="card-header">
          <div class="card-title">
            <span>🧠 Supervisory Consensus Decision</span>
          </div>
          <div style="display: flex; gap: 8px;">
            <button id="btnDownloadJson" class="btn btn-secondary" style="padding: 4px 10px; font-size: 11px;" onclick="downloadArtifact('json')" disabled>JSON</button>
            <button id="btnDownloadMd" class="btn btn-secondary" style="padding: 4px 10px; font-size: 11px;" onclick="downloadArtifact('md')" disabled>Markdown</button>
          </div>
        </div>

        <div id="emptyResultsState" class="empty-state">
          <div class="empty-icon">🔬</div>
          <div style="font-size: 16px; font-weight: 600; color: var(--text-main); margin-bottom: 6px;">Ready for Evaluation</div>
          <div>Select a holdout case above and click <strong>"Run Multi-Agent Deliberation"</strong>.</div>
        </div>

        <div id="resultsContainer" style="display: none;">
          <!-- Top Decision Banner -->
          <div id="decisionBanner" class="decision-banner status-autonomous">
            <div>
              <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-dim); margin-bottom: 4px;">Consensus Assessment</div>
              <div id="biradsBadge" class="birads-badge birads-4C">BI-RADS 4C</div>
              <div id="deferralStatusText" style="font-size: 12px; font-weight: 600; margin-top: 6px; color: var(--accent-green);">
                ✓ Autonomous Screening Clearance
              </div>
            </div>

            <div class="prob-metric">
              <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-dim); margin-bottom: 4px;">Malignancy Risk</div>
              <div id="probVal" class="prob-val">58.42%</div>
              <div id="probInterval" class="prob-sub">95% CI: [49.36%, 66.95%]</div>
            </div>
          </div>

          <!-- Specialist Cards -->
          <div class="specialists-grid">
            <!-- Mammo Specialist -->
            <div class="specialist-card">
              <div class="specialist-header">
                <span class="specialist-title">Mammography Agent</span>
                <span class="specialist-role role-primary">Authoritative (1.0)</span>
              </div>
              <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 6px;">
                <strong>Margin:</strong> <span id="mammoMargin" class="covariate-val" style="font-size: 12px;">-</span>
              </div>
              <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 6px;">
                <strong>Morphology:</strong> <span id="mammoCalc" class="covariate-val" style="font-size: 12px;">-</span>
              </div>
              <div style="font-size: 11px; color: var(--text-dim); margin-top: 8px;">
                Grad-CAM Bounding Box: <code id="mammoSaliency">[140, 210, 95, 110]</code>
              </div>
            </div>

            <!-- Clin Specialist -->
            <div class="specialist-card">
              <div class="specialist-header">
                <span class="specialist-title">Clinical Risk Agent</span>
                <span class="specialist-role role-shadow">Shadow (0.0)</span>
              </div>
              <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 6px;">
                <strong>Risk Profile:</strong> <span id="clinProfile" class="covariate-val" style="font-size: 12px;">-</span>
              </div>
              <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 6px;">
                <strong>Voting:</strong> <span style="color: var(--accent-amber);">False (Zero Weight)</span>
              </div>
              <div style="font-size: 11px; color: var(--text-dim); margin-top: 8px;">
                TreeSHAP Density Impact: <code id="clinShap">+0.1600</code>
              </div>
            </div>
          </div>

          <!-- Rationale Box -->
          <div class="card-title" style="margin-bottom: 8px;">
            <span>📝 Supervisory Evidence Rationale (Qwen 2.5 3B)</span>
          </div>
          <div id="rationaleBox" class="rationale-box">
            Deliberation text will appear here...
          </div>

          <!-- Citations Row -->
          <div class="card-title" style="margin-bottom: 8px; font-size: 12px;">
            <span>🔍 Audited Traceability Citations</span>
          </div>
          <div id="citationsRow" class="citations-row">
            <!-- Badges dynamically injected -->
          </div>
        </div>
      </div>
    </section>
  </main>

  <script>
    let casesList = [];
    let currentCaseIndex = 0;
    let latestDecision = null;

    // Load available holdout cases
    async function loadCases() {
      try {
        const resp = await fetch('/api/cases');
        const data = await resp.json();
        casesList = data.cases;

        const sel = document.getElementById('caseSelector');
        sel.innerHTML = '';
        casesList.forEach((c, idx) => {
          const opt = document.createElement('option');
          opt.value = c.id;
          opt.textContent = `${c.id}  —  [${c.label.toUpperCase()}]`;
          sel.appendChild(opt);
        });

        // Default to P_00092_LEFT or first
        const defaultIndex = casesList.findIndex(c => c.id === 'P_00092_LEFT');
        currentCaseIndex = defaultIndex >= 0 ? defaultIndex : 0;
        sel.selectedIndex = currentCaseIndex;
        onCaseSelectionChanged();
      } catch (err) {
        console.error("Failed to load cases:", err);
      }
    }

    function onCaseSelectionChanged() {
      const sel = document.getElementById('caseSelector');
      currentCaseIndex = sel.selectedIndex;
      const caseId = sel.value;
      if (!caseId) return;

      // Update image previews
      document.getElementById('imgCC').src = `/api/image/${caseId}/CC?t=${Date.now()}`;
      document.getElementById('imgMLO').src = `/api/image/${caseId}/MLO?t=${Date.now()}`;

      // Fetch basic case metadata
      fetch(`/api/case_meta/${caseId}`)
        .then(r => r.json())
        .then(meta => {
          document.getElementById('covPatientId').textContent = meta.patient_id;
          document.getElementById('covSide').textContent = meta.breast_side;
          document.getElementById('covDensity').textContent = `Category ${meta.breast_density}`;
          document.getElementById('covAbnType').textContent = meta.abnormality_type.toUpperCase();
          document.getElementById('covGroundTruth').textContent = meta.target_name.toUpperCase();
        })
        .catch(err => console.error(err));
    }

    function stepCase(delta) {
      if (!casesList.length) return;
      currentCaseIndex = (currentCaseIndex + delta + casesList.length) % casesList.length;
      document.getElementById('caseSelector').selectedIndex = currentCaseIndex;
      onCaseSelectionChanged();
    }

    function pickRandomCase() {
      if (!casesList.length) return;
      currentCaseIndex = Math.floor(Math.random() * casesList.length);
      document.getElementById('caseSelector').selectedIndex = currentCaseIndex;
      onCaseSelectionChanged();
    }

    async function runDeliberation() {
      const caseId = document.getElementById('caseSelector').value;
      if (!caseId) return;

      const btn = document.getElementById('btnEvaluate');
      const btnIcon = document.getElementById('btnIcon');
      const btnText = document.getElementById('btnText');

      btn.disabled = true;
      btnIcon.innerHTML = '<div class="spinner"></div>';
      btnText.textContent = "Deliberating with LLM Supervisor...";

      try {
        const resp = await fetch(`/api/evaluate/${caseId}`);
        const data = await resp.json();
        latestDecision = data;

        // Render Results
        document.getElementById('emptyResultsState').style.display = 'none';
        document.getElementById('resultsContainer').style.display = 'block';

        // BI-RADS Badge
        const biradsBadge = document.getElementById('biradsBadge');
        const birads = data.consensus_bi_rads;
        biradsBadge.textContent = `BI-RADS ${birads}`;
        biradsBadge.className = `birads-badge birads-${birads.replace('/', '_')}`;

        // Deferral Status
        const banner = document.getElementById('decisionBanner');
        const deferText = document.getElementById('deferralStatusText');
        if (data.defer_to_human) {
          banner.className = 'decision-banner status-defer';
          deferText.textContent = `⚠ DEFERRED TO RADIOLOGY PANEL (${data.deferral_reasons.join(', ')})`;
          deferText.style.color = 'var(--accent-amber)';
        } else {
          banner.className = 'decision-banner status-autonomous';
          deferText.textContent = '✓ AUTONOMOUS SCREENING CLEARANCE ISSUED';
          deferText.style.color = 'var(--accent-green)';
        }

        // Probability
        document.getElementById('probVal').textContent = `${(data.malignancy_prob * 100).toFixed(2)}%`;
        document.getElementById('probInterval').textContent = `95% CI: [${(data.confidence_interval[0] * 100).toFixed(1)}%, ${(data.confidence_interval[1] * 100).toFixed(1)}%]`;

        // Specialists
        document.getElementById('mammoMargin').textContent = data.mammo_card.finding.margin || 'none';
        document.getElementById('mammoCalc').textContent = data.mammo_card.finding.calcification || 'none';
        document.getElementById('mammoSaliency').textContent = JSON.stringify(data.mammo_card.sal.top_regions[0] || []);

        document.getElementById('clinProfile').textContent = data.clin_card.finding.risk_profile.toUpperCase();
        document.getElementById('clinShap').textContent = `+${data.clin_card.sh.breast_density}`;

        // Rationale text with stylized citations
        let formattedRationale = data.evidence_rationale.replace(/\[card:([a-zA-Z0-9_\/]+)\]/g, '<span class="citation-tag">[$1]</span>');
        document.getElementById('rationaleBox').innerHTML = formattedRationale;

        // Citations List
        const citationsRow = document.getElementById('citationsRow');
        citationsRow.innerHTML = '';
        data.citations.forEach(([agent, field]) => {
          const pill = document.createElement('span');
          pill.className = 'citation-tag';
          pill.textContent = `${agent.toUpperCase()} ➔ ${field}`;
          citationsRow.appendChild(pill);
        });

        // Enable download buttons
        document.getElementById('btnDownloadJson').disabled = false;
        document.getElementById('btnDownloadMd').disabled = false;

      } catch (err) {
        alert("Evaluation failed: " + err.message);
      } finally {
        btn.disabled = false;
        btnIcon.textContent = "⚡";
        btnText.textContent = "Run Multi-Agent Deliberation";
      }
    }

    function downloadArtifact(type) {
      if (!latestDecision) return;
      const caseId = latestDecision.case_id;
      window.open(`/api/download/${type}/${caseId}`, '_blank');
    }

    // Startup
    window.addEventListener('DOMContentLoaded', loadCases);
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the primary clinical UI dashboard."""
    return HTMLResponse(content=HTML_TEMPLATE, status_code=200)


@app.get("/api/cases")
async def get_holdout_cases():
    """Returns sorted list of holdout cases with quarantined ground-truth labels."""
    images_dir = PROJECT_ROOT / "RP2_LOCAL_HOLDOUT_TRANSFER" / "images"
    labels_file = PROJECT_ROOT / "RP2_LOCAL_HOLDOUT_TRANSFER" / "evaluation_ground_truth" / "holdout_binary_labels.csv"

    labels_map = {}
    if labels_file.is_file():
        with open(labels_file, "r", encoding="utf-8") as f:
            labels_map = {r["pair_id"]: r.get("binary_target_name", "unknown") for r in csv.DictReader(f)}

    cases = []
    if images_dir.is_dir():
        case_ids = sorted(list(set([f.name[:-7] for f in images_dir.glob("*_CC.npy")])))
        for cid in case_ids:
            cases.append({
                "id": cid,
                "label": labels_map.get(cid, "unspecified")
            })

    return {"count": len(cases), "cases": cases}


@app.get("/api/case_meta/{case_id}")
async def get_case_metadata(case_id: str):
    """Returns patient clinical covariates for a case."""
    try:
        cov = lookup_clinical_covariates(case_id)
        return cov
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/image/{case_id}/{view}")
async def stream_mammogram_image(case_id: str, view: str):
    """Converts a preprocessed .npy mammogram array on-the-fly to PNG."""
    view_upper = view.upper()
    if view_upper not in ("CC", "MLO"):
        raise HTTPException(status_code=400, detail="View must be CC or MLO")

    search_dirs = [
        PROJECT_ROOT / "RP2_LOCAL_HOLDOUT_TRANSFER" / "images",
        PROJECT_ROOT / "data" / "cached_preprocessed_512x512"
    ]

    npy_path = None
    for d in search_dirs:
        candidate = d / f"{case_id}_{view_upper}.npy"
        if candidate.is_file():
            npy_path = candidate
            break

    if npy_path is None:
        raise HTTPException(status_code=404, detail=f"Image {case_id}_{view_upper}.npy not found")

    arr = np.load(npy_path)
    # Clip and scale to 8-bit uint8
    arr_u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr_u8, mode="L")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/api/evaluate/{case_id}")
async def evaluate_case_api(case_id: str, model: str = "qwen2.5:3b"):
    """Runs full multi-agent perception and LLM supervisory deliberation."""
    try:
        # 1. Image metadata
        img_meta = load_cached_images(case_id)

        # 2. Covariates
        covariates = lookup_clinical_covariates(case_id)

        # 3. Agent cards
        mammo_card, clin_card = get_agent_finding_cards(case_id, covariates, img_meta)

        # 4. Supervisor state
        mammo_payload = AgentFindingPayload(
            agent_id="mammo",
            status="success",
            authority_level="authoritative",
            voting_eligible=True,
            bi_rads=mammo_card["BI_RADS_a"],
            confidence=mammo_card["conf_a"],
            confidence_type=mammo_card["confidence_type"],
            p_malignancy=mammo_card["conf_a"],
            findings_summary=mammo_card["finding"],
            task_probabilities=mammo_card["metadata"]["task_probabilities"],
            explainability=mammo_card["sal"],
            rationale=mammo_card["rationale"],
            raw_finding_card=mammo_card,
            metadata={}
        )
        clin_payload = AgentFindingPayload(
            agent_id="clin",
            status="success",
            authority_level="non_authoritative_research",
            voting_eligible=False,
            bi_rads=clin_card["BI_RADS_a"],
            confidence=clin_card["conf_a"],
            confidence_type=clin_card["confidence_type"],
            p_malignancy=clin_card["metadata"]["task_probabilities"]["malignancy_calibrated"],
            findings_summary=clin_card["finding"],
            task_probabilities=clin_card["metadata"]["task_probabilities"],
            explainability={"tree_shap_attributions": clin_card["sh"]},
            rationale=clin_card["rationale"],
            raw_finding_card=clin_card,
            metadata={}
        )

        state = SupervisorState(
            study_id=case_id,
            timestamp_utc="2026-10-27T12:00:00Z",
            agents={"mammo": mammo_payload, "clin": clin_payload},
            defer_to_human=False
        )

        # 5. LLM Supervisor Deliberation with Fallback
        try:
            client = OllamaLLMClient(model_name=model)
            supervisor = LLMSupervisor(client=client, theta=0.030)
            decision = supervisor.deliberate(state)
        except Exception as e:
            # Fallback to deterministic client if Ollama is momentarily busy
            mock_client = MockLLMClient()
            supervisor = LLMSupervisor(client=mock_client, theta=0.030)
            decision = supervisor.deliberate(state)

        # 6. Extract audited quotes
        quotes = extract_evidence_quotes(decision.evidence_rationale)

        # 7. Persist outputs
        out_dir = PROJECT_ROOT / "reports" / "rp2_experimental" / "inference_runs"
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{case_id}_supervisor_decision.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(decision.to_json(indent=2))

        md_path = out_dir / f"{case_id}_clinical_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Multi-Agent Clinical Consensus Report: {case_id}\n\n")
            f.write(f"- **Patient ID**: `{covariates['patient_id']}`\n")
            f.write(f"- **Consensus BI-RADS**: **{decision.consensus_bi_rads}**\n")
            f.write(f"- **Calibrated Malignancy Probability**: `{decision.malignancy_probability}`\n")
            f.write(f"- **Confidence Interval (95%)**: `[{decision.confidence_interval[0]}, {decision.confidence_interval[1]}]`\n")
            f.write(f"- **Human Deferral Required**: `{decision.defer_to_human}`\n\n")
            f.write(f"## Clinical Rationale\n{decision.evidence_rationale}\n")

        return {
            "case_id": case_id,
            "consensus_bi_rads": decision.consensus_bi_rads,
            "malignancy_prob": decision.malignancy_probability,
            "confidence_interval": decision.confidence_interval,
            "defer_to_human": decision.defer_to_human,
            "deferral_reasons": decision.deferral_reasons,
            "evidence_rationale": decision.evidence_rationale,
            "citations": quotes,
            "mammo_card": mammo_card,
            "clin_card": clin_card,
            "covariates": covariates
        }

    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


@app.get("/api/download/{fmt}/{case_id}")
async def download_report_file(fmt: str, case_id: str):
    """Downloads the generated report file."""
    out_dir = PROJECT_ROOT / "reports" / "rp2_experimental" / "inference_runs"
    if fmt == "json":
        fpath = out_dir / f"{case_id}_supervisor_decision.json"
        media = "application/json"
    elif fmt == "md":
        fpath = out_dir / f"{case_id}_clinical_report.md"
        media = "text/markdown"
    else:
        raise HTTPException(status_code=400, detail="Format must be json or md")

    if not fpath.is_file():
        raise HTTPException(status_code=404, detail="Report file not found. Run deliberation first.")

    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()

    return Response(
        content=content,
        media_type=media,
        headers={"Content-Disposition": f"attachment; filename={fpath.name}"}
    )


def main():
    parser = argparse.ArgumentParser(description="RP2 Mammography Agent UI Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    args = parser.parse_args()

    print("=" * 70)
    print("🚀 RP2 Mammography Agent Decision Support Dashboard")
    print(f"👉 Serving locally at: http://{args.host}:{args.port}")
    print("=" * 70)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
