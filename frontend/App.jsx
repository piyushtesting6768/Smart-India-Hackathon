import React, { useState, useRef, useEffect, useCallback } from "react";

/* =========================================================================
   BORDER CONTROL SCREENING SYSTEM — MHA / SSB
   SIH 2026 · Problem Statement 26188 · AI-Based Fake Identity & Document
   Screening System

   Design tokens
   - Display face:  "Oswald"        (condensed, tactical, stenciled feel)
   - Body face:     "IBM Plex Sans" (disciplined, government-form register)
   - Data/mono:     "IBM Plex Mono" (every readout, hash, score, timestamp)
   - Palette:       slate / navy darks, crisp white, emerald / amber / red
                     reserved exclusively for GENUINE / SUSPICIOUS / FAKE
   - Signature:     the Audit Log rendered as a literal linked ledger —
                     each entry shows its own hash physically overlapping
                     the next block's "previous hash" field, so the chain
                     is something you can see break, not just read about.
   ========================================================================= */

const FONT_IMPORT_STYLE = `
  @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
  .font-display { font-family: 'Oswald', sans-serif; letter-spacing: 0.02em; }
  .font-body { font-family: 'IBM Plex Sans', sans-serif; }
  .font-mono { font-family: 'IBM Plex Mono', monospace; }
`;

const CHECKPOINTS = [
  "Checkpoint Alpha - Land Border",
  "Checkpoint Bravo - Land Border",
  "Checkpoint Delta - Rail Terminal",
  "Checkpoint Echo - Airport Transit",
];

const DOC_TYPES = [
  { id: "passport", label: "Passport" },
  { id: "visa", label: "Visa" },
  { id: "national_id", label: "National ID" },
  { id: "driving_license", label: "Driving License" },
  { id: "permit", label: "Permit Document" },
];

const PROCESSING_STEPS = [
  "Extracting OCR text fields (EasyOCR + Tesseract)...",
  "Validating checksums, QR codes and formats...",
  "Analyzing ELA / FFT frequency spectrum for GenAI artifacts...",
  "Executing deep face-embedding cross-match...",
  "Running rPPG pulse signal and blink-tracking liveness check...",
  "Fusing module scores and generating verdict...",
];

function hexChars(n) {
  const chars = "0123456789abcdef";
  let out = "";
  for (let i = 0; i < n; i++) out += chars[Math.floor(Math.random() * 16)];
  return out;
}

function fakeSha256() {
  return hexChars(64);
}

function truncateHash(h) {
  if (!h) return "—";
  return `${h.slice(0, 10)}…${h.slice(-8)}`;
}

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

function verdictFromScore(score) {
  if (score >= 80) return "GENUINE";
  if (score >= 50) return "SUSPICIOUS";
  return "FAKE";
}

const VERDICT_STYLES = {
  GENUINE: {
    bg: "bg-emerald-950",
    border: "border-emerald-700",
    text: "text-emerald-400",
    pill: "bg-emerald-600 text-emerald-50",
    bar: "bg-emerald-500",
  },
  SUSPICIOUS: {
    bg: "bg-amber-950",
    border: "border-amber-700",
    text: "text-amber-400",
    pill: "bg-amber-500 text-amber-950",
    bar: "bg-amber-500",
  },
  FAKE: {
    bg: "bg-red-950",
    border: "border-red-700",
    text: "text-red-400",
    pill: "bg-red-600 text-red-50",
    bar: "bg-red-500",
  },
};

function StatusBadge({ pass, labelPass = "PASS", labelFail = "FAIL" }) {
  return (
    <span
      className={
        "px-2 py-0.5 rounded font-mono text-[11px] font-semibold tracking-wide border " +
        (pass
          ? "bg-emerald-950 text-emerald-400 border-emerald-700"
          : "bg-red-950 text-red-400 border-red-700")
      }
    >
      {pass ? labelPass : labelFail}
    </span>
  );
}

function SectionCard({ title, eyebrow, children, right }) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-sm">
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800 bg-slate-950/60">
        <div>
          {eyebrow && (
            <div className="font-mono text-[10px] tracking-[0.2em] text-slate-500 uppercase">
              {eyebrow}
            </div>
          )}
          <h3 className="font-display text-slate-100 text-base tracking-wide uppercase">
            {title}
          </h3>
        </div>
        {right}
      </div>
      <div className="p-4">{children}</div>
    </div>
  );
}

/* -------------------------------------------------------------------------
   Document Upload & Preview
   ------------------------------------------------------------------------- */
function DocumentUploadCard({ docType, setDocType, file, setFile }) {
  const [dragOver, setDragOver] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [cropInspect, setCropInspect] = useState(false);
  const [lensPos, setLensPos] = useState({ x: 0, y: 0, visible: false });
  const imgWrapRef = useRef(null);
  const fileInputRef = useRef(null);

  const acceptTypes = ".jpg,.jpeg,.png,.tiff,.tif,.pdf";

  const handleFiles = (fileList) => {
    const f = fileList && fileList[0];
    if (!f) return;
    const isPdf = f.type === "application/pdf";
    const reader = new FileReader();
    reader.onload = (e) => {
      setFile({
        name: f.name,
        type: f.type,
        isPdf,
        dataUrl: isPdf ? null : e.target.result,
        sizeKb: Math.round(f.size / 1024),
      });
      setZoom(1);
    };
    reader.readAsDataURL(f);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    handleFiles(e.dataTransfer.files);
  };

  const onMouseMoveImg = (e) => {
    if (!cropInspect || !imgWrapRef.current) return;
    const rect = imgWrapRef.current.getBoundingClientRect();
    const x = clamp(((e.clientX - rect.left) / rect.width) * 100, 0, 100);
    const y = clamp(((e.clientY - rect.top) / rect.height) * 100, 0, 100);
    setLensPos({ x, y, visible: true });
  };

  return (
    <SectionCard
      eyebrow="Module 1–3 Input"
      title="Document Upload & Selection"
      right={
        <span className="font-mono text-[11px] text-slate-500">
          {file ? `${file.sizeKb} KB` : "No file loaded"}
        </span>
      }
    >
      <div className="space-y-4">
        <div>
          <label className="block font-mono text-[11px] tracking-wide text-slate-500 uppercase mb-2">
            Document Type
          </label>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {DOC_TYPES.map((d) => (
              <button
                key={d.id}
                onClick={() => setDocType(d.id)}
                className={
                  "font-mono text-xs px-3 py-2 border rounded-sm text-left transition-colors " +
                  (docType === d.id
                    ? "bg-slate-100 text-slate-950 border-slate-100"
                    : "bg-slate-950 text-slate-300 border-slate-700 hover:border-slate-500")
                }
              >
                {d.label}
              </button>
            ))}
          </div>
        </div>

        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClick={() => fileInputRef.current?.click()}
          className={
            "border-2 border-dashed rounded-sm p-6 text-center cursor-pointer transition-colors " +
            (dragOver
              ? "border-slate-300 bg-slate-800"
              : "border-slate-700 bg-slate-950 hover:border-slate-600")
          }
        >
          <input
            ref={fileInputRef}
            type="file"
            accept={acceptTypes}
            className="hidden"
            onChange={(e) => handleFiles(e.target.files)}
          />
          <p className="font-body text-sm text-slate-400">
            Drag &amp; drop a scan, or click to browse
          </p>
          <p className="font-mono text-[11px] text-slate-600 mt-1">
            JPG · PNG · TIFF · PDF
          </p>
        </div>

        {file && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-mono text-[11px] text-slate-500 truncate max-w-[60%]">
                {file.name}
              </span>
              <div className="flex items-center gap-1">
                <button
                  onClick={() => setZoom((z) => clamp(z - 0.25, 1, 3))}
                  className="w-7 h-7 font-mono text-slate-300 border border-slate-700 rounded-sm hover:border-slate-500"
                >
                  −
                </button>
                <span className="font-mono text-[11px] text-slate-500 w-10 text-center">
                  {Math.round(zoom * 100)}%
                </span>
                <button
                  onClick={() => setZoom((z) => clamp(z + 0.25, 1, 3))}
                  className="w-7 h-7 font-mono text-slate-300 border border-slate-700 rounded-sm hover:border-slate-500"
                >
                  +
                </button>
                <button
                  onClick={() => setCropInspect((c) => !c)}
                  className={
                    "ml-2 font-mono text-[11px] px-2 py-1 border rounded-sm " +
                    (cropInspect
                      ? "bg-slate-100 text-slate-950 border-slate-100"
                      : "border-slate-700 text-slate-300 hover:border-slate-500")
                  }
                >
                  Crop Inspect
                </button>
              </div>
            </div>

            <div
              ref={imgWrapRef}
              onMouseMove={onMouseMoveImg}
              onMouseLeave={() => setLensPos((p) => ({ ...p, visible: false }))}
              className="relative overflow-auto bg-slate-950 border border-slate-800 rounded-sm h-64"
            >
              {file.isPdf ? (
                <div className="flex items-center justify-center h-full font-mono text-xs text-slate-500">
                  PDF loaded — page 1 rendered for OCR pipeline
                </div>
              ) : (
                <img
                  src={file.dataUrl}
                  alt="Uploaded document"
                  style={{ transform: `scale(${zoom})`, transformOrigin: "top left" }}
                  className="max-w-none"
                />
              )}

              {cropInspect && lensPos.visible && !file.isPdf && (
                <div
                  className="pointer-events-none absolute w-32 h-32 rounded-full border-2 border-slate-100 shadow-lg"
                  style={{
                    left: `calc(${lensPos.x}% - 64px)`,
                    top: `calc(${lensPos.y}% - 64px)`,
                    backgroundImage: `url(${file.dataUrl})`,
                    backgroundSize: "400% 400%",
                    backgroundPosition: `${lensPos.x}% ${lensPos.y}%`,
                  }}
                />
              )}
            </div>
          </div>
        )}
      </div>
    </SectionCard>
  );
}

/* -------------------------------------------------------------------------
   Live Camera & Liveness Capture
   ------------------------------------------------------------------------- */
function LiveCameraCard({ capturedPhoto, setCapturedPhoto }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const [cameraOn, setCameraOn] = useState(false);
  const [recording, setRecording] = useState(false);
  const [recordSecondsLeft, setRecordSecondsLeft] = useState(0);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraOn(true);
    } catch (err) {
      alert(
        "Camera access denied or unavailable. Grant camera permission to run liveness capture."
      );
    }
  };

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setCameraOn(false);
    setRecording(false);
  }, []);

  const resetFeed = () => {
    stopCamera();
    setCapturedPhoto(null);
  };

  const captureLivenessSequence = () => {
    if (!videoRef.current || !canvasRef.current) return;
    setRecording(true);
    setRecordSecondsLeft(5);

    const canvas = canvasRef.current;
    canvas.width = videoRef.current.videoWidth || 640;
    canvas.height = videoRef.current.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    const snapshot = canvas.toDataURL("image/png");

    let secondsLeft = 5;
    const interval = setInterval(() => {
      secondsLeft -= 1;
      setRecordSecondsLeft(secondsLeft);
      if (secondsLeft <= 0) {
        clearInterval(interval);
        setRecording(false);
        setCapturedPhoto(snapshot);
      }
    }, 1000);
  };

  useEffect(() => {
    return () => stopCamera();
  }, [stopCamera]);

  return (
    <SectionCard
      eyebrow="Module 4–5 Input"
      title="Live Camera & Liveness Capture"
      right={
        capturedPhoto && (
          <span className="font-mono text-[11px] text-emerald-400">
            LIVENESS SAMPLE CAPTURED
          </span>
        )
      }
    >
      <div className="space-y-3">
        <div className="relative bg-black border border-slate-800 rounded-sm h-64 flex items-center justify-center overflow-hidden">
          {!cameraOn && !capturedPhoto && (
            <p className="font-mono text-xs text-slate-600">
              Camera feed inactive
            </p>
          )}

          <video
            ref={videoRef}
            muted
            playsInline
            className={
              "w-full h-full object-cover " + (cameraOn ? "block" : "hidden")
            }
          />

          {capturedPhoto && !cameraOn && (
            <img
              src={capturedPhoto}
              alt="Captured liveness frame"
              className="w-full h-full object-cover"
            />
          )}

          {cameraOn && (
            <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
              <div className="w-40 h-52 border-2 border-dashed border-slate-100/70 rounded-[50%]" />
              <div className="absolute top-6 left-6 w-6 h-6 border-t-2 border-l-2 border-emerald-400" />
              <div className="absolute top-6 right-6 w-6 h-6 border-t-2 border-r-2 border-emerald-400" />
              <div className="absolute bottom-6 left-6 w-6 h-6 border-b-2 border-l-2 border-emerald-400" />
              <div className="absolute bottom-6 right-6 w-6 h-6 border-b-2 border-r-2 border-emerald-400" />
            </div>
          )}

          {recording && (
            <div className="absolute top-2 right-2 bg-red-600 text-white font-mono text-[11px] px-2 py-1 rounded-sm">
              REC {recordSecondsLeft}s
            </div>
          )}
        </div>

        <canvas ref={canvasRef} className="hidden" />

        <div className="flex flex-wrap gap-2">
          {!cameraOn ? (
            <button
              onClick={startCamera}
              className="font-mono text-xs px-3 py-2 bg-slate-100 text-slate-950 rounded-sm hover:bg-white"
            >
              Start Camera
            </button>
          ) : (
            <button
              onClick={captureLivenessSequence}
              disabled={recording}
              className="font-mono text-xs px-3 py-2 bg-emerald-600 text-emerald-50 rounded-sm hover:bg-emerald-500 disabled:opacity-50"
            >
              Capture Live Photo / 5s Liveness Video
            </button>
          )}
          <button
            onClick={resetFeed}
            className="font-mono text-xs px-3 py-2 border border-slate-700 text-slate-300 rounded-sm hover:border-slate-500"
          >
            Reset Feed
          </button>
        </div>
      </div>
    </SectionCard>
  );
}

/* -------------------------------------------------------------------------
   Processing Panel
   ------------------------------------------------------------------------- */
function ProcessingPanel({ processing, currentStep }) {
  return (
    <div className="mt-4 space-y-1.5">
      {PROCESSING_STEPS.map((step, i) => {
        const done = i < currentStep || !processing;
        const active = processing && i === currentStep;
        return (
          <div key={step} className="flex items-center gap-2">
            <span
              className={
                "w-4 h-4 flex items-center justify-center rounded-full border font-mono text-[9px] shrink-0 " +
                (processing && i < currentStep
                  ? "bg-emerald-600 border-emerald-600 text-emerald-50"
                  : active
                  ? "border-slate-100 text-slate-100 animate-pulse"
                  : "border-slate-700 text-slate-700")
              }
            >
              {processing && i < currentStep ? "✓" : i + 1}
            </span>
            <span
              className={
                "font-mono text-xs " +
                (processing && i <= currentStep
                  ? "text-slate-200"
                  : "text-slate-600")
              }
            >
              {step}
            </span>
          </div>
        );
      })}
    </div>
  );
}

/* -------------------------------------------------------------------------
   Results Dashboard
   ------------------------------------------------------------------------- */
function OcrCard({ docType, ocr }) {
  return (
    <SectionCard eyebrow="Module 1" title="OCR Extraction">
      <dl className="grid grid-cols-2 gap-x-3 gap-y-2">
        {Object.entries(ocr).map(([k, v]) => (
          <React.Fragment key={k}>
            <dt className="font-mono text-[10px] uppercase tracking-wide text-slate-500">
              {k}
            </dt>
            <dd className="font-mono text-xs text-slate-200 text-right truncate">
              {v}
            </dd>
          </React.Fragment>
        ))}
      </dl>
    </SectionCard>
  );
}

function ValidationCard({ validation }) {
  return (
    <SectionCard eyebrow="Module 2" title="Document Validation">
      <div className="space-y-2">
        {Object.entries(validation).map(([k, v]) => (
          <div key={k} className="flex items-center justify-between">
            <span className="font-mono text-xs text-slate-400">{k}</span>
            <StatusBadge pass={v} />
          </div>
        ))}
      </div>
    </SectionCard>
  );
}

function TamperingCard({ tampering }) {
  return (
    <SectionCard eyebrow="Module 3 · Core Innovation" title="Tampering Detection">
      <div className="space-y-3">
        <MetricBar label="ELA Anomaly Score" value={tampering.ela} suffix="%" invert />
        <MetricBar
          label="FFT Frequency Grid Spike (GenAI)"
          value={tampering.fft}
          suffix="%"
          invert
        />
        <MetricBar
          label="CNN Forgery Probability"
          value={tampering.cnn}
          suffix="%"
          invert
        />
      </div>
    </SectionCard>
  );
}

function FaceCard({ face }) {
  return (
    <SectionCard eyebrow="Module 4" title="Face Verification">
      <div className="space-y-3">
        <MetricBar label="Similarity Match" value={face.similarity} suffix="%" />
        <div className="flex items-center justify-between">
          <span className="font-mono text-xs text-slate-400">
            Embedding Distance
          </span>
          <span className="font-mono text-xs text-slate-200">
            {face.distance.toFixed(3)}
          </span>
        </div>
      </div>
    </SectionCard>
  );
}

function LivenessCard({ liveness }) {
  return (
    <SectionCard eyebrow="Module 5" title="Liveness & Deepfake Detection">
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="font-mono text-xs text-slate-400">Pulse Signal (rPPG)</span>
          <StatusBadge pass={liveness.pulseDetected} labelPass="DETECTED" labelFail="ABSENT" />
        </div>
        <div className="flex items-center justify-between">
          <span className="font-mono text-xs text-slate-400">Estimated BPM</span>
          <span className="font-mono text-xs text-slate-200">{liveness.bpm} BPM</span>
        </div>
        <MetricBar label="Blink Verification Score" value={liveness.blinkScore} suffix="%" />
      </div>
    </SectionCard>
  );
}

function MetricBar({ label, value, suffix = "", invert = false }) {
  const good = invert ? value < 35 : value > 65;
  const warn = invert ? value >= 35 && value < 60 : value >= 40 && value <= 65;
  const color = good ? "bg-emerald-500" : warn ? "bg-amber-500" : "bg-red-500";
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="font-mono text-xs text-slate-400">{label}</span>
        <span className="font-mono text-xs text-slate-200">
          {value}
          {suffix}
        </span>
      </div>
      <div className="h-1.5 bg-slate-800 rounded-sm overflow-hidden">
        <div className={`h-full ${color}`} style={{ width: `${clamp(value, 0, 100)}%` }} />
      </div>
    </div>
  );
}

function VerdictBanner({ verdict, score }) {
  const s = VERDICT_STYLES[verdict];
  return (
    <div className={`border rounded-sm ${s.bg} ${s.border} p-5`}>
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <div className="font-mono text-[10px] tracking-[0.2em] text-slate-400 uppercase mb-1">
            Unified Risk Verdict
          </div>
          <div className={`font-display text-3xl tracking-wide ${s.text}`}>
            {verdict}
          </div>
        </div>
        <div className="text-right">
          <div className="font-mono text-[10px] tracking-[0.2em] text-slate-400 uppercase mb-1">
            Risk Score
          </div>
          <div className="font-mono text-3xl text-slate-100">{score}/100</div>
        </div>
      </div>
      <div className="mt-4 h-2 bg-slate-800 rounded-sm overflow-hidden">
        <div className={`h-full ${s.bar}`} style={{ width: `${score}%` }} />
      </div>
      <div className="mt-2 font-mono text-[11px] text-slate-500">
        Score = 0.15×OCR + 0.20×Validation + 0.30×Tampering + 0.15×FaceMatch + 0.20×Liveness
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------
   Audit Log — the ledger (signature element)
   ------------------------------------------------------------------------- */
function AuditLedger({ log }) {
  return (
    <SectionCard eyebrow="Module 6" title="Audit Trail — Hash-Chained Log">
      {log.length === 0 ? (
        <p className="font-mono text-xs text-slate-600">
          No entries yet. Run a screening to append the first block.
        </p>
      ) : (
        <div className="space-y-0">
          {log.map((entry, i) => {
            const s = VERDICT_STYLES[entry.verdict];
            const isLast = i === 0;
            return (
              <div key={entry.logId} className="relative pl-6">
                {i !== log.length - 1 && (
                  <div className="absolute left-[9px] top-6 bottom-0 w-px bg-slate-700" />
                )}
                <div
                  className={
                    "absolute left-0 top-1.5 w-[19px] h-[19px] rounded-full border-2 flex items-center justify-center " +
                    s.border
                  }
                >
                  <div className={`w-2 h-2 rounded-full ${s.bar}`} />
                </div>
                <div className="border border-slate-800 rounded-sm p-3 mb-3 bg-slate-950">
                  <div className="flex items-center justify-between flex-wrap gap-2 mb-2">
                    <span className="font-mono text-xs text-slate-300">
                      {entry.logId}
                    </span>
                    <span
                      className={`px-2 py-0.5 rounded font-mono text-[10px] font-semibold ${s.pill}`}
                    >
                      {entry.verdict}
                    </span>
                    <span className="font-mono text-[11px] text-slate-500">
                      {entry.timestamp}
                    </span>
                  </div>
                  <dl className="grid sm:grid-cols-2 gap-x-4 gap-y-1">
                    <dt className="font-mono text-[10px] text-slate-500 uppercase">
                      Document Hash
                    </dt>
                    <dd className="font-mono text-[11px] text-slate-400">
                      {truncateHash(entry.docHash)}
                    </dd>
                    <dt className="font-mono text-[10px] text-slate-500 uppercase">
                      Previous Hash
                    </dt>
                    <dd className="font-mono text-[11px] text-slate-500">
                      {isLast ? "GENESIS" : truncateHash(entry.prevHash)}
                    </dd>
                    <dt className="font-mono text-[10px] text-slate-500 uppercase">
                      Current Hash
                    </dt>
                    <dd className="font-mono text-[11px] text-emerald-400">
                      {truncateHash(entry.currHash)}
                    </dd>
                  </dl>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </SectionCard>
  );
}

/* -------------------------------------------------------------------------
   Mock analysis generation
   ------------------------------------------------------------------------- */
function mockOcrFields(docType) {
  const base = {
    passport: {
      Name: "RAVI KUMAR SHARMA",
      "Passport No.": "P" + hexChars(0).padStart(0, "") + "K" + Math.floor(1000000 + Math.random() * 8999999),
      Nationality: "IND",
      DOB: "14-MAR-1991",
      "Expiry Date": "22-NOV-2031",
      Gender: "M",
    },
    visa: {
      "Visa No.": "V" + Math.floor(100000 + Math.random() * 899999),
      "Visa Type": "Tourist - T2",
      "Entry Validation": "Single Entry",
      "Stay Duration": "30 Days",
    },
    national_id: {
      Name: "ANITA VERMA",
      "ID Number": Math.floor(100000000000 + Math.random() * 899999999999),
      DOB: "02-JUL-1988",
      Address: "Sector 21, New Delhi",
    },
    driving_license: {
      Name: "MOHD. FAISAL",
      "DL Number": "DL" + Math.floor(1000000000000 + Math.random() * 8999999999999),
      "Valid Till": "05-JAN-2029",
      Class: "LMV, MCWG",
    },
    permit: {
      "Permit No.": "PRM-" + Math.floor(10000 + Math.random() * 89999),
      "Issuing Authority": "SSB Border Wing",
      "Valid For": "Restricted Area Entry",
      "Valid Till": "30-SEP-2026",
    },
  };
  return base[docType] || base.passport;
}

function generateMockResults(docType, prevHash) {
  const ocrConfidence = Math.round(80 + Math.random() * 19);
  const validation = {
    "Checksum Verification": Math.random() > 0.15,
    "QR Code Match": Math.random() > 0.2,
    "Expiry Date Valid": Math.random() > 0.1,
    "Field Format (Regex)": Math.random() > 0.1,
  };
  const validationScore = Math.round(
    (Object.values(validation).filter(Boolean).length / 4) * 100
  );

  const tampering = {
    ela: Math.round(Math.random() * 55),
    fft: Math.round(Math.random() * 55),
    cnn: Math.round(Math.random() * 55),
  };
  const tamperingScore = Math.round(
    100 - (tampering.ela + tampering.fft + tampering.cnn) / 3
  );

  const face = {
    similarity: Math.round(70 + Math.random() * 29),
    distance: Math.random() * 0.6,
  };
  const faceScore = face.similarity;

  const liveness = {
    pulseDetected: Math.random() > 0.15,
    bpm: Math.round(62 + Math.random() * 34),
    blinkScore: Math.round(65 + Math.random() * 34),
  };
  const livenessScore = Math.round(
    (liveness.pulseDetected ? 55 : 15) + liveness.blinkScore * 0.45
  );

  const riskScore = Math.round(
    0.15 * ocrConfidence +
      0.2 * validationScore +
      0.3 * tamperingScore +
      0.15 * faceScore +
      0.2 * clamp(livenessScore, 0, 100)
  );

  const verdict = verdictFromScore(riskScore);

  const docHash = fakeSha256();
  const currHash = fakeSha256();
  const now = new Date();
  const timestamp = now.toISOString().replace("T", " ").slice(0, 19) + " UTC";

  return {
    ocr: { ...mockOcrFields(docType), "OCR Confidence": ocrConfidence + "%" },
    validation,
    tampering,
    face,
    liveness,
    riskScore,
    verdict,
    auditEntry: {
      logId: "LOG-" + hexChars(6).toUpperCase(),
      timestamp,
      docHash,
      prevHash,
      currHash,
      verdict,
    },
  };
}

/* -------------------------------------------------------------------------
   App
   ------------------------------------------------------------------------- */
export default function App() {
  const [checkpoint, setCheckpoint] = useState(CHECKPOINTS[0]);
  const [officerId] = useState("OFC-SSB-" + hexChars(4).toUpperCase());
  const [docType, setDocType] = useState("passport");
  const [file, setFile] = useState(null);
  const [capturedPhoto, setCapturedPhoto] = useState(null);

  const [processing, setProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [results, setResults] = useState(null);
  const [auditLog, setAuditLog] = useState([]);

  const readyToRun = Boolean(file && capturedPhoto) && !processing;

  const runAnalysis = () => {
    if (!readyToRun) return;
    setResults(null);
    setProcessing(true);
    setCurrentStep(0);

    let step = 0;
    const stepTimer = setInterval(() => {
      step += 1;
      setCurrentStep(step);
      if (step >= PROCESSING_STEPS.length) {
        clearInterval(stepTimer);
        setTimeout(() => {
          const prevHash = auditLog[0]?.currHash || null;
          const r = generateMockResults(docType, prevHash);
          setResults(r);
          setAuditLog((log) => [r.auditEntry, ...log]);
          setProcessing(false);
        }, 400);
      }
    }, 550);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-body">
      <style>{FONT_IMPORT_STYLE}</style>

      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-950 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-4 flex flex-wrap items-center justify-between gap-4">
          <div>
            <div className="font-mono text-[10px] tracking-[0.25em] text-slate-500 uppercase">
              Ministry of Home Affairs · Sashastra Seema Bal
            </div>
            <h1 className="font-display text-xl sm:text-2xl uppercase tracking-wide text-slate-100">
              Border Control Screening System — MHA / SSB
            </h1>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <select
              value={checkpoint}
              onChange={(e) => setCheckpoint(e.target.value)}
              className="font-mono text-xs bg-slate-900 border border-slate-700 text-slate-200 rounded-sm px-2 py-2"
            >
              {CHECKPOINTS.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>

            <div className="font-mono text-xs border border-slate-700 rounded-sm px-3 py-2 text-slate-300">
              Officer&nbsp;
              <span className="text-slate-100">{officerId}</span>
            </div>

            <div className="flex items-center gap-2 font-mono text-xs border border-emerald-800 bg-emerald-950 text-emerald-400 rounded-sm px-3 py-2">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              System Active / AI Modules Ready
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-6 space-y-6">
        <div className="grid lg:grid-cols-2 gap-6">
          <DocumentUploadCard
            docType={docType}
            setDocType={setDocType}
            file={file}
            setFile={setFile}
          />
          <LiveCameraCard
            capturedPhoto={capturedPhoto}
            setCapturedPhoto={setCapturedPhoto}
          />
        </div>

        {/* Action & processing */}
        <SectionCard eyebrow="Action" title="Screening Execution">
          <div className="flex flex-col sm:flex-row sm:items-center gap-4">
            <button
              onClick={runAnalysis}
              disabled={!readyToRun}
              className={
                "font-display tracking-wide uppercase text-sm px-6 py-3 rounded-sm transition-colors " +
                (readyToRun
                  ? "bg-slate-100 text-slate-950 hover:bg-white"
                  : "bg-slate-800 text-slate-600 cursor-not-allowed")
              }
            >
              Run Border Screening Analysis
            </button>
            {!readyToRun && !processing && (
              <span className="font-mono text-[11px] text-slate-500">
                Load a document and capture a live photo to enable analysis.
              </span>
            )}
          </div>

          {processing && (
            <ProcessingPanel processing={processing} currentStep={currentStep} />
          )}
        </SectionCard>

        {/* Results dashboard */}
        {results && (
          <div className="space-y-6">
            <VerdictBanner verdict={results.verdict} score={results.riskScore} />

            <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-6">
              <OcrCard docType={docType} ocr={results.ocr} />
              <ValidationCard validation={results.validation} />
              <TamperingCard tampering={results.tampering} />
              <FaceCard face={results.face} />
              <LivenessCard liveness={results.liveness} />
            </div>

            <AuditLedger log={auditLog} />
          </div>
        )}
      </main>

      <footer className="border-t border-slate-800 py-4 text-center">
        <p className="font-mono text-[10px] text-slate-600">
          Prototype interface only · All module outputs are simulated pending
          backend integration with the FastAPI orchestrator
        </p>
      </footer>
    </div>
  );
}
