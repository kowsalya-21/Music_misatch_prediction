/* Label Audit Frontend Controller (Vanilla JS, no framework, no build step) */

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("auditForm");
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("audioFile");
  const fileInfo = document.getElementById("fileInfo");
  const titleInput = document.getElementById("titleInput");
  const artistInput = document.getElementById("artistInput");
  const submitBtn = document.getElementById("submitBtn");
  const statusBanner = document.getElementById("statusBanner");
  const resultsSection = document.getElementById("resultsSection");
  const tabUpload = document.getElementById("tabUpload");
  const tabSearch = document.getElementById("tabSearch");
  const searchNotice = document.getElementById("searchNotice");

  let currentMode = "upload"; // "upload" or "search"

  function setMode(mode) {
    currentMode = mode;
    if (mode === "upload") {
      tabUpload.classList.add("active");
      tabUpload.style.border = "1.5px solid #2B2A27";
      tabUpload.style.background = "#2B2A27";
      tabUpload.style.color = "#ffffff";

      tabSearch.classList.remove("active");
      tabSearch.style.border = "1.5px solid #D1CCBF";
      tabSearch.style.background = "transparent";
      tabSearch.style.color = "#57544C";

      dropzone.style.display = "block";
      if (searchNotice) searchNotice.style.display = "none";
      submitBtn.textContent = "Run Label Audit";
    } else {
      tabSearch.classList.add("active");
      tabSearch.style.border = "1.5px solid #2B2A27";
      tabSearch.style.background = "#2B2A27";
      tabSearch.style.color = "#ffffff";

      tabUpload.classList.remove("active");
      tabUpload.style.border = "1.5px solid #D1CCBF";
      tabUpload.style.background = "transparent";
      tabUpload.style.color = "#57544C";

      dropzone.style.display = "none";
      if (searchNotice) searchNotice.style.display = "block";
      submitBtn.textContent = "Fetch Online & Run Label Audit";
    }
  }

  // Mode switching
  if (tabUpload && tabSearch) {
    tabUpload.addEventListener("click", () => setMode("upload"));
    tabSearch.addEventListener("click", () => setMode("search"));
    setMode("upload"); // Initial styling
  }

  // Drag and drop handlers
  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      fileInput.files = files;
      showFileInfo(files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      showFileInfo(fileInput.files[0]);
    }
  });

  function showFileInfo(file) {
    const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
    fileInfo.style.display = "block";
    fileInfo.textContent = `Selected: ${file.name} (${sizeMb} MB)`;
  }

  function setStatus(type, message) {
    statusBanner.className = `status-banner ${type}`;
    statusBanner.textContent = message;
    statusBanner.style.display = "block";
  }

  function clearStatus() {
    statusBanner.className = "status-banner";
    statusBanner.textContent = "";
    statusBanner.style.display = "none";
  }

  // Form submission
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    resultsSection.style.display = "none";

    const formData = new FormData(form);

    const hasFile = fileInput.files && fileInput.files.length > 0;
    const titleVal = titleInput ? titleInput.value.trim() : "";
    const isSearch = currentMode === "search" || (!hasFile && titleVal.length > 0);

    if (!hasFile && !titleVal) {
      setStatus("error", "Please select an audio file to analyze, OR type a Song Title to search online.");
      return;
    }

    if (!isSearch) {
      // Audio File Upload Mode
      submitBtn.disabled = true;
      setStatus("analyzing", "Uploading audio and running empirical mood models on CPU...");

      try {
        const response = await fetch("/api/analyze", {
          method: "POST",
          body: formData
        });
        await handleApiResponse(response);
      } catch (err) {
        setStatus("error", "Network connection error. Could not connect to Label Audit API.");
      } finally {
        submitBtn.disabled = false;
      }
    } else {
      // Online Search mode
      submitBtn.disabled = true;
      setStatus("analyzing", `Fetching studio preview and lyrics for "${titleVal}" online...`);

      try {
        const response = await fetch("/api/fetch_and_analyze", {
          method: "POST",
          body: formData
        });
        await handleApiResponse(response);
      } catch (err) {
        setStatus("error", "Network connection error. Could not connect to Label Audit API.");
      } finally {
        submitBtn.disabled = false;
      }
    }
  });

  async function handleApiResponse(response) {
    if (response.status === 200) {
      const data = await response.json();
      clearStatus();
      renderResults(data);
    } else if (response.status === 404) {
      const err = await response.json().catch(() => ({}));
      setStatus("error", `Song not found: ${err.detail || "Could not find audio preview for this track online."}`);
    } else if (response.status === 413) {
      setStatus("error", "File too large: Audio file exceeds maximum limit of 50 MB.");
    } else if (response.status === 415) {
      setStatus("error", "Unsupported format: Please upload a valid MP3, WAV, OGG, FLAC, or M4A audio file.");
    } else if (response.status === 429) {
      setStatus("error", "Rate limit exceeded (60 requests per minute). Please wait a moment before trying again.");
    } else if (response.status === 503) {
      setStatus("error", "The analysis model is not connected to this deployment yet.");
    } else {
      const errData = await response.json().catch(() => ({}));
      setStatus("error", `Analysis failed (${response.status}): ${errData.detail || "Server error"}`);
    }
  }

  function renderResults(data) {
    resultsSection.style.display = "block";

    // Track Artwork / Title Banner
    const trackCard = document.getElementById("trackInfoCard");
    const trackArt = document.getElementById("trackArtwork");
    const trackTitleDisp = document.getElementById("trackTitleDisplay");
    const trackArtistDisp = document.getElementById("trackArtistDisplay");

    if (trackCard && (data.artwork_url || data.track_title)) {
      trackCard.style.display = "flex";
      if (data.artwork_url) {
        trackArt.src = data.artwork_url;
        trackArt.style.display = "block";
      } else {
        trackArt.style.display = "none";
      }
      trackTitleDisp.textContent = data.track_title || "Analyzed Track";
      trackArtistDisp.textContent = data.track_artist || "";
    } else if (trackCard) {
      trackCard.style.display = "none";
    }


    // 1. Acoustic Details
    document.getElementById("resValence").textContent = data.audio.valence.toFixed(3);
    document.getElementById("resArousal").textContent = data.audio.arousal.toFixed(3);
    document.getElementById("resDuration").textContent = data.duration_analyzed_s.toFixed(1);
    document.getElementById("resTopTags").textContent = data.audio.top_tags.join(", ");
    document.getElementById("canvasDurationEnd").textContent = `${data.duration_analyzed_s.toFixed(0)}s`;

    // 2. Probability Bars
    const probContainer = document.getElementById("probBars");
    probContainer.innerHTML = "";
    const quadNames = ["happy", "angry", "sad", "calm"];

    quadNames.forEach((q) => {
      const prob = data.audio.quadrant_probs[q] || 0.0;
      const pct = (prob * 100).toFixed(1);

      const barHtml = `
        <div class="prob-bar-container">
          <div class="prob-header">
            <span style="text-transform: capitalize; font-weight: 500;">${q}</span>
            <span class="mono-num">${pct}%</span>
          </div>
          <div class="prob-track">
            <div class="prob-fill" style="width: ${pct}%;"></div>
          </div>
        </div>
      `;
      probContainer.insertAdjacentHTML("beforeend", barHtml);
    });

    // 3. Circumplex SVG Plot
    renderCircumplexPlane(data);

    // 4. Mel Spectrogram Canvas
    renderMelCanvas(data.mel);

    // 5. Mismatch Section
    const badgeWrapper = document.getElementById("mismatchBadgeWrapper");
    const scoreLine = document.getElementById("mismatchScoreLine");
    const reasonLine = document.getElementById("mismatchReason");
    const disclaimerLine = document.getElementById("mismatchDisclaimer");
    const actualLabelEl = document.getElementById("actualLabelDisplay");
    const predictedLabelEl = document.getElementById("predictedLabelDisplay");
    const auditStatusTag = document.getElementById("auditStatusTag");

    badgeWrapper.innerHTML = "";
    disclaimerLine.textContent = data.mismatch.disclaimer;

    // Actual vs Predicted Status
    const rawActual = data.label && data.label.raw_label ? data.label.raw_label : "(None provided)";
    const canonActual = data.label && data.label.quadrant ? data.label.quadrant : "unknown";
    const predMood = data.audio && data.audio.primary_quadrant ? data.audio.primary_quadrant : "-";

    if (actualLabelEl) {
      if (data.label && data.label.is_known) {
        actualLabelEl.textContent = `${rawActual} (Mapped to: ${canonActual})`;
      } else {
        actualLabelEl.textContent = rawActual;
      }
    }
    if (predictedLabelEl) {
      predictedLabelEl.textContent = `${predMood} (Acoustic Model)`;
    }

    if (!data.label || !data.label.is_known) {
      badgeWrapper.innerHTML = `<span class="mismatch-badge badge-neutral">No Valid Label</span>`;
      scoreLine.textContent = "Mismatch score: null (label unknown or not provided)";
      reasonLine.textContent = data.label.note || "No human label provided for comparison.";
      if (auditStatusTag) {
        auditStatusTag.textContent = "UNAUDITED (NO LABEL)";
        auditStatusTag.style.color = "#57544C";
      }
    } else if (data.mismatch.flagged) {
      badgeWrapper.innerHTML = `<span class="mismatch-badge badge-mismatch">Flagged Mismatch</span>`;
      scoreLine.innerHTML = `Score: <span class="mono-num">${data.mismatch.score.toFixed(4)}</span> (Threshold: <span class="mono-num">${data.mismatch.threshold.toFixed(4)}</span>)`;
      reasonLine.textContent = data.mismatch.reason;
      if (auditStatusTag) {
        auditStatusTag.textContent = "MISMATCH DETECTED";
        auditStatusTag.style.color = "#C0392B";
      }
    } else {
      badgeWrapper.innerHTML = `<span class="mismatch-badge badge-match">Confirmed Match</span>`;
      scoreLine.innerHTML = `Score: <span class="mono-num">${data.mismatch.score.toFixed(4)}</span> (Threshold: <span class="mono-num">${data.mismatch.threshold.toFixed(4)}</span>)`;
      reasonLine.textContent = data.mismatch.reason;
      if (auditStatusTag) {
        auditStatusTag.textContent = "VERIFIED MATCH";
        auditStatusTag.style.color = "#27AE60";
      }
    }

    // 6. Lyrics Section
    const lyrStatus = document.getElementById("lyricsStatusLine");
    const lyrCoords = document.getElementById("lyricsCoordsLine");
    const lyrNote = document.getElementById("lyricsNote");

    if (data.lyrics.status === "success") {
      lyrStatus.innerHTML = `Lyrical Mood: <strong style="text-transform: capitalize;">${data.lyrics.primary_quadrant}</strong> (Source: ${data.lyrics.source || "online"})`;
      lyrCoords.innerHTML = `Coordinates: Valence = <span class="mono-num">${data.lyrics.valence.toFixed(3)}</span>, Arousal = <span class="mono-num">${data.lyrics.arousal.toFixed(3)}</span>`;
      lyrNote.textContent = data.lyrics.note || "";
    } else if (data.lyrics.status === "language_not_validated") {
      lyrStatus.textContent = "Lyrical Sentiment: Language Not Validated";
      lyrCoords.textContent = "Model inference skipped.";
      lyrNote.textContent = data.lyrics.note;
    } else {
      lyrStatus.textContent = "Lyrical Sentiment: Lyrics Not Found";
      lyrCoords.textContent = "Coordinates: null";
      lyrNote.textContent = "Verification confidence is adjusted accordingly when lyrics are unavailable.";
    }

    // 7. Verification Agent Section
    const agentCard = document.getElementById("agentCard");
    const verdictBox = document.getElementById("agentVerdictBox");
    const traceList = document.getElementById("agentTraceList");

    if (data.agent) {
      agentCard.style.display = "block";
      const confPct = (data.agent.confidence * 100).toFixed(1);
      const verdictTitle = data.agent.verdict.replace("_", " ").toUpperCase();

      verdictBox.innerHTML = `
        <div style="margin-bottom: 8px;">
          <strong>Verdict:</strong> <span class="mono-num" style="font-weight: 600;">${verdictTitle}</span>
          (Confidence: <span class="mono-num">${confPct}%</span>)
        </div>
        <p style="font-size: 0.95rem; line-height: 1.45; color: var(--text-main);">${data.agent.explanation}</p>
      `;

      traceList.innerHTML = "";
      data.agent.trace.forEach((t) => {
        const itemHtml = `
          <li class="trace-item">
            <div>
              <span class="trace-tool">[Step ${t.step}: ${t.tool}]</span>
              <span class="trace-time">${t.duration_ms.toFixed(1)}ms</span>
            </div>
            <div class="trace-summary">${t.output_summary}</div>
          </li>
        `;
        traceList.insertAdjacentHTML("beforeend", itemHtml);
      });
    } else {
      agentCard.style.display = "none";
    }

    // Smooth scroll to results
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderCircumplexPlane(data) {
    const markersGroup = document.getElementById("svgMarkers");
    markersGroup.innerHTML = "";

    // Mapping [-1, 1] to SVG coordinates (origin at center: 160, 160; radius 140)
    function toSvgCoords(v, a) {
      const x = 160 + v * 130;
      const y = 160 - a * 130; // invert y for SVG
      return { x, y };
    }

    const readout = document.getElementById("distanceReadout");
    let readoutText = "";

    // 1. Audio Marker (Red Circle)
    const audioPt = toSvgCoords(data.audio.valence, data.audio.arousal);
    markersGroup.insertAdjacentHTML("beforeend", `
      <circle cx="${audioPt.x}" cy="${audioPt.y}" r="6.5" fill="#C8401B" stroke="#FFFFFF" stroke-width="1.5" />
      <text x="${audioPt.x + 8}" y="${audioPt.y - 6}" font-size="11" font-weight="600" fill="#C8401B" font-family="var(--font-mono)">Audio</text>
    `);

    // 2. Human Label Marker (Near-black Square)
    if (data.label && data.label.is_known && data.label.valence !== null) {
      const labelPt = toSvgCoords(data.label.valence, data.label.arousal);
      markersGroup.insertAdjacentHTML("beforeend", `
        <rect x="${labelPt.x - 6}" y="${labelPt.y - 6}" width="12" height="12" fill="#15140F" stroke="#FFFFFF" stroke-width="1.5" />
        <text x="${labelPt.x + 9}" y="${labelPt.y + 4}" font-size="11" font-weight="600" fill="#15140F" font-family="var(--font-mono)">Label (${data.label.quadrant})</text>
        <line x1="${audioPt.x}" y1="${audioPt.y}" x2="${labelPt.x}" y2="${labelPt.y}" stroke="#C8401B" stroke-dasharray="3,3" stroke-width="1.2" />
      `);

      if (data.mismatch && data.mismatch.distance_audio_label !== null) {
        readoutText += `Euclidean Distance (Audio to Label): ${data.mismatch.distance_audio_label.toFixed(4)}. `;
      }
    }

    // 3. Lyrics Marker (Dark Gray Diamond)
    if (data.lyrics && data.lyrics.status === "success" && data.lyrics.valence !== null) {
      const lyrPt = toSvgCoords(data.lyrics.valence, data.lyrics.arousal);
      const diamondPoints = `${lyrPt.x},${lyrPt.y - 7} ${lyrPt.x + 7},${lyrPt.y} ${lyrPt.x},${lyrPt.y + 7} ${lyrPt.x - 7},${lyrPt.y}`;
      markersGroup.insertAdjacentHTML("beforeend", `
        <polygon points="${diamondPoints}" fill="#57544C" stroke="#FFFFFF" stroke-width="1.5" />
        <text x="${lyrPt.x + 9}" y="${lyrPt.y + 12}" font-size="11" font-weight="600" fill="#57544C" font-family="var(--font-mono)">Lyrics</text>
      `);

      if (data.mismatch && data.mismatch.distance_audio_lyrics !== null) {
        readoutText += `Distance (Audio to Lyrics): ${data.mismatch.distance_audio_lyrics.toFixed(4)}.`;
      }
    }

    readout.textContent = readoutText;
  }

  function renderMelCanvas(melData) {
    const canvas = document.getElementById("melCanvas");
    const ctx = canvas.getContext("2d");

    const binaryStr = atob(melData.uint8_data);
    const len = binaryStr.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = binaryStr.charCodeAt(i);
    }

    const nMels = melData.n_mels;
    const nFrames = melData.n_frames;

    canvas.width = nFrames;
    canvas.height = nMels;

    const imgData = ctx.createImageData(nFrames, nMels);
    const data = imgData.data;

    // Convert mel values to a clear high-contrast spectrogram display
    for (let m = 0; m < nMels; m++) {
      // Invert row index so low frequencies are at bottom
      const row = nMels - 1 - m;
      for (let f = 0; f < nFrames; f++) {
        const val = bytes[m * nFrames + f]; // 0 to 255
        const pxIdx = (row * nFrames + f) * 4;

        // Custom high-contrast colormap from warm off-white to deep black
        data[pxIdx + 0] = Math.min(255, val * 1.2);      // R
        data[pxIdx + 1] = Math.min(255, val * 0.9);      // G
        data[pxIdx + 2] = Math.min(255, val * 0.6);      // B
        data[pxIdx + 3] = 255;                           // Alpha
      }
    }

    ctx.putImageData(imgData, 0, 0);
  }
});
