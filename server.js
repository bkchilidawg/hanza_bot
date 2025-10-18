// server.js
// GPT-5-nano via /v1/responses (string input), robust extraction, smart join +
// punctuation normalization, auto-continue, and env-tunable token caps.

import express from "express";
import dotenv from "dotenv";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import PDFDocument from "pdfkit";
import compression from "compression";
import helmet from "helmet";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const app   = express();
const PORT  = process.env.PORT || 3000;

/* ---------------- Model normalize ---------------- */
function normalizeModelId(id) {
  const s = String(id || "").trim();
  if (/^gpt[-_]?5([-_]?nano)?$/i.test(s)) return "gpt-5-nano";
  if (/^gpt[-_]?5/i.test(s)) return s.replace(/_/g, "-").toLowerCase();
  return s || "gpt-5-nano";
}
const MODEL = normalizeModelId(process.env.OPENAI_MODEL || "gpt-5-nano");
const DEMO  = String(process.env.DEMO_MODE || "").toLowerCase() === "true";
const DEBUG = !!process.env.DEBUG;
const HAS_KEY = !!process.env.OPENAI_API_KEY;
const IS_GPT5 = /^gpt-5/i.test(MODEL);

/* ---------------- Paths ---------------- */
const PUBLIC_DIR  = path.join(__dirname, "public");
const DATA_DIR    = path.join(__dirname, "data");
const TRANSCRIPTS = path.join(DATA_DIR, "transcripts");
fs.mkdirSync(DATA_DIR, { recursive: true });
fs.mkdirSync(TRANSCRIPTS, { recursive: true });

/* ---------------- Middleware ---------------- */
app.disable("x-powered-by");
app.use(helmet({ contentSecurityPolicy: false }));
app.use(compression());
app.use(express.json({ limit: "2mb" }));
app.use(express.static(PUBLIC_DIR));
app.use("/transcripts", express.static(TRANSCRIPTS));

/* ---------------- Utils ---------------- */
function styleInstruction(tone = "balanced", length = "standard") {
  const toneMap = {
    balanced:      "Use a balanced, approachable professional tone.",
    formal:        "Use a formal, executive-ready tone with precise language.",
    casual:        "Use a casual, conversational tone with plain language.",
    bold:          "Use a bold, persuasive tone with clear calls to action.",
    friendly:      "Use a warm, friendly tone that feels encouraging.",
    authoritative: "Use an authoritative, research-backed tone with confident statements.",
    storytelling:  "Use a storytelling tone with narrative hooks and vivid examples."
  };
  const lengthMap = {
    concise:  "Target ~600 words. Be tight and focused.",
    standard: "Target ~900 words with clear sections.",
    detailed: "Target 1200–1500 words with rich detail and examples.",
    ultra:    "Target 2000+ words with exhaustive coverage."
  };
  return `${toneMap[tone] || toneMap.balanced} ${lengthMap[length] || lengthMap.standard}`;
}
function sseHeaders() {
  return {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache, no-transform",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no"
  };
}
const escapeSSE = (s="") => String(s).replace(/\u0000/g, "");
const delay = (ms) => new Promise(r => setTimeout(r, ms));
function chunkForSSE(text = "", size = 200) {
  const parts = [];
  for (let i=0;i<text.length;i+=size) parts.push(text.slice(i, i+size));
  return parts;
}

// Path safety
function safeBasename(name="") {
  const raw = String(name);
  if (raw.includes("/") || raw.includes("\\")) throw new Error("Invalid filename");
  return raw.replace(/[^\w.\-]+/g, "_").slice(0, 120);
}
function safePathJoin(root, name) {
  const base = safeBasename(name);
  const p = path.join(root, base);
  const rel = path.relative(root, p);
  if (rel.startsWith("..") || path.isAbsolute(rel)) throw new Error("Unsafe path");
  return p;
}

// Extract user's real ask from meta blocks
function extractUserRequest(raw = "") {
  const s = String(raw || "");
  const marker = s.match(/USER REQUEST:\s*([\s\S]*)$/i);
  if (marker && marker[1]) return marker[1].trim();
  let cleaned = s
    .replace(/VOICE\s*&\s*STYLE[\s\S]*?(?:^-{3,}\s*$|\n---\s*\n)/gmi, "")
    .replace(/CONTEXT\s*EXCERPTS:[\s\S]*?(?:^-{3,}\s*$|\n---\s*\n)/gmi, "")
    .replace(/^INSTRUCTIONS:[\s\S]*?(?:^-{3,}\s*$|\n---\s*\n)/gmi, "");
  cleaned = cleaned.trim();
  return cleaned || s.trim();
}

/* ---------------- Token budgets (x10-ready) ---------------- */
const MAX_TOKEN_CAP = Math.max(128, Number(process.env.RESP_MAX_TOKENS_CAP || 3500));
const TOKEN_MULT    = Math.min(10, Math.max(1, Number(process.env.TOKEN_MULTIPLIER || 1)));
const AUTOCONTINUE_ROUNDS = Math.min(20, Math.max(1, Number(process.env.AUTOCONTINUE_ROUNDS || 6)));

function tokensForLength(length = "standard") {
  switch (String(length)) {
    case "concise":   return 900;
    case "standard":  return 1300;
    case "detailed":  return 2000;
    case "ultra":     return 3200;
    default:          return 1500;
  }
}
function clampTokens(n) { return Math.max(64, Math.min(MAX_TOKEN_CAP, Number.isFinite(n) ? n : 1500)); }
function applyMultiplier(n) { return clampTokens(Math.round(n * TOKEN_MULT)); }

/* ---------------- Prompt builders ---------------- */
function buildMessages(userPrompt, tone, length) {
  const style = styleInstruction(tone, length);
  const system =
    "You are a focused blog-writing assistant. " +
    "Write clearly in Markdown with # Title and ## Sections, bullets, numbered steps, and practical advice. " +
    "Avoid emojis unless asked.";
  return [
    { role: "system", content: `${system}\n\nStylistic guidance: ${style}` },
    { role: "user",   content: userPrompt }
  ];
}
function messagesToPlainString(messages = []) {
  return messages.map(m => {
    const role = m.role.toUpperCase();
    const content = typeof m.content === "string" ? m.content : JSON.stringify(m.content);
    return `${role}:\n${content}`;
  }).join("\n\n");
}
const lastUserText = (messages=[]) =>
  (messages.findLast?.(m => m.role === "user")?.content) ||
  [...messages].reverse().find(m => m.role === "user")?.content || "";

/* ---------------- OpenAI (Responses) ---------------- */
function formatOpenAIError(status, bodyText) {
  try {
    const j = JSON.parse(bodyText);
    const msg = j?.error?.message || j?.message || bodyText;
    return `OpenAI error ${status}: ${msg}`;
  } catch {
    return `OpenAI error ${status}: ${bodyText || "(no body)"}`;
  }
}
async function fetchJSON(url, body, { signal } = {}) {
  if (!HAS_KEY) throw new Error("OPENAI_API_KEY is missing");
  const resp = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body),
    signal
  });
  const text = await resp.text().catch(()=> "");
  if (!resp.ok) throw new Error(formatOpenAIError(resp.status, text));
  try { return JSON.parse(text); } catch { return {}; }
}

/* ---- Smart text assembly & normalization ---- */
function smartJoin(parts = []) {
  let out = "";
  for (const raw of parts) {
    const s = String(raw ?? "");
    if (!s) continue;
    if (!out) { out = s; continue; }

    const prev = out[out.length - 1];
    const next = s[0];

    const needSpace =
      (/\w/.test(prev) && /\w/.test(next)) ||
      (/[)\]]/.test(prev) && /\w/.test(next)) ||
      (/[.,;:!?]/.test(prev) && !/[\s.,;:!?)\]]/.test(next));

    out += (needSpace ? " " : "") + s;
  }
  return out.replace(/\s+([.,;:!?])/g, "$1");
}
function normalizeOutputText(s = "") {
  if (!s) return s;
  s = s.replace(/\r\n?/g, "\n");
  s = s.replace(/([A-Za-z0-9])\.([A-Za-z])/g, "$1. $2");
  s = s.replace(/([,;:!?])(?=[A-Za-z0-9(])/g, "$1 ");
  s = s.replace(/\s+([.,;:!?])/g, "$1");
  s = s.replace(/\be\. ?g\./gi, "e.g.");
  s = s.replace(/\bi\. ?e\./gi, "i.e.");
  s = s.replace(/\b([A-Z])\. ?([A-Z])\./g, "$1.$2.");
  s = s.replace(/[ \t]{2,}/g, " ").replace(/\. \./g, ". ");
  return s;
}

/* Robust extractor for Responses variants */
function extractResponseText(data) {
  if (typeof data?.output_text === "string" && data.output_text) return normalizeOutputText(data.output_text);
  if (Array.isArray(data?.output_text) && data.output_text.length) return normalizeOutputText(data.output_text.join(""));

  const chunks = [];
  const push = (s) => { if (s && typeof s === "string") chunks.push(s); };

  const visit = (node, role = null) => {
    if (node == null) return;
    if (typeof node === "string") { push(node); return; }
    if (Array.isArray(node)) { for (const n of node) visit(n, role); return; }
    if (typeof node !== "object") return;

    const nextRole = typeof node.role === "string" ? node.role : role;
    const ty = node.type;

    for (const k of ["output_text", "text", "content", "value", "message"]) {
      if (typeof node[k] === "string") {
        if (nextRole === "assistant" || k === "output_text" || ty === "output_text" || ty === "output_text_delta" || ty === "text" || ty === "refusal") {
          push(node[k]);
        }
      }
    }

    if (node.delta) {
      if (typeof node.delta === "string") push(node.delta);
      else visit(node.delta, nextRole);
    }

    for (const k of ["output", "content", "message", "messages", "choices", "arguments", "items", "parts", "data"]) {
      if (node[k]) visit(node[k], nextRole);
    }
  };

  visit(data, null);

  const out = normalizeOutputText(smartJoin(chunks)).trim();
  if (!out && DEBUG) {
    try { console.log("ℹ️ Empty extract; raw head:", JSON.stringify(data).slice(0, 400)); } catch {}
  }
  return out;
}

function getIncompleteReason(data) {
  if (!data || typeof data !== "object") return null;
  if (data.status === "incomplete") return data?.incomplete_details?.reason || "unknown";
  return null;
}

// Single call with string input
async function callOnce(messages, tokens, { signal } = {}) {
  const inputStr = messagesToPlainString(messages);
  const body = { model: MODEL, input: inputStr, max_output_tokens: clampTokens(tokens) };
  const data = await fetchJSON("https://api.openai.com/v1/responses", body, { signal });
  const text = extractResponseText(data);
  const reason = getIncompleteReason(data);
  return { text, incompleteReason: reason, raw: data };
}

// Auto-continue on max_output_tokens
async function completeWithAutoContinue(messages, tokens, { rounds = AUTOCONTINUE_ROUNDS } = {}) {
  let acc = "";
  let baseMessages = messages;
  let lastReason = null;

  for (let i = 0; i < Math.max(1, rounds); i++) {
    const { text, incompleteReason } = await callOnce(baseMessages, tokens);
    if (text) acc += (acc ? "\n" : "") + text;
    lastReason = incompleteReason;

    if (incompleteReason !== "max_output_tokens") break;

    baseMessages = [
      ...messages,
      { role: "assistant", content: text || "" },
      { role: "user", content: "Continue from where you stopped. Keep the same structure and style." }
    ];
    tokens = clampTokens(Math.round(tokens * 1.25));
  }

  return { text: acc, incompleteReason: lastReason };
}

/* ---------------- Health + config ---------------- */
app.get("/health", (_req, res) => res.json({
  ok: true,
  port: String(PORT),
  model: MODEL,
  key: HAS_KEY ? "present" : "missing",
  api_mode: IS_GPT5 ? "responses" : "chat",
  caps: {
    RESP_MAX_TOKENS_CAP: MAX_TOKEN_CAP,
    TOKEN_MULTIPLIER: TOKEN_MULT,
    AUTOCONTINUE_ROUNDS
  }
}));
app.get("/config", (_req, res) => res.json({
  model: MODEL, demo: DEMO, api_mode: IS_GPT5 ? "responses" : "chat_completions"
}));

/* ---------------- Verify endpoint ---------------- */
app.get("/api/test-gpt5", async (_req, res) => {
  try {
    if (!HAS_KEY) return res.status(400).json({ ok:false, error:"OPENAI_API_KEY missing" });
    if (!IS_GPT5) return res.status(400).json({ ok:false, error:`Model '${MODEL}' is not GPT-5.*` });

    const data = await fetchJSON("https://api.openai.com/v1/responses", {
      model: MODEL,
      input: "Say: hello from gpt-5-nano",
      max_output_tokens: 64
    });

    const output = extractResponseText(data) || "";
    res.json({ ok:true, output, raw: data });
  } catch (e) {
    res.status(500).json({ ok:false, error: String(e.message || e) });
  }
});

/* ---------------- /api/raw: dump raw Responses JSON for debugging ---------------- */
app.post("/api/raw", async (req, res) => {
  try {
    const {
      message = "",
      tone = "balanced",
      length = "standard",
      rounds = 1,
      max_output_tokens
    } = req.body || {};
    if (!message) return res.status(400).json({ ok:false, error:"Missing 'message'." });

    if (DEMO) {
      return res.json({
        ok: true,
        note: "DEMO_MODE=true: not calling OpenAI",
        sampleRequest: { model: MODEL, message, tone, length, rounds, max_output_tokens }
      });
    }
    if (!HAS_KEY) return res.status(400).json({ ok:false, error:"OPENAI_API_KEY missing" });
    if (!IS_GPT5) return res.status(400).json({ ok:false, error:`OPENAI_MODEL='${MODEL}' is not a GPT-5 model` });

    const userText = extractUserRequest(message);
    const baseMsgs = buildMessages(userText, tone, length);

    let tokens = clampTokens(max_output_tokens ?? applyMultiplier(tokensForLength(length)));
    const responses = [];
    let accText = "";
    let incompleteReason = null;

    let msgs = baseMsgs;
    const ROUNDS = Math.min(20, Math.max(1, Number(rounds || 1)));

    for (let i = 0; i < ROUNDS; i++) {
      const body = { model: MODEL, input: messagesToPlainString(msgs), max_output_tokens: tokens };
      const data = await fetchJSON("https://api.openai.com/v1/responses", body);
      responses.push(data);

      const piece = extractResponseText(data) || "";
      if (piece) accText += (accText ? "\n" : "") + piece;

      incompleteReason = getIncompleteReason(data);
      if (incompleteReason !== "max_output_tokens") break;

      // prepare next round (auto-continue style)
      msgs = [
        ...baseMsgs,
        { role: "assistant", content: piece },
        { role: "user", content: "Continue from where you stopped. Keep the same structure and style." }
      ];
      tokens = clampTokens(Math.round(tokens * 1.25));
    }

    res.json({
      ok: true,
      model: MODEL,
      roundsUsed: responses.length,
      tokenBudgetStart: clampTokens(max_output_tokens ?? applyMultiplier(tokensForLength(length))),
      tokenBudgetFinal: tokens,
      incompleteReason: incompleteReason || null,
      sample_text: accText.slice(0, 2000),
      responses
    });
  } catch (e) {
    if (DEBUG) console.error("api/raw error:", e);
    res.status(500).json({ ok:false, error: String(e.message || e) });
  }
});

/* GET convenience (single-round) --------------------------------------------- */
app.get("/api/raw", async (req, res) => {
  try {
    const message = String(req.query.message || "");
    const tone    = String(req.query.tone || "balanced");
    const length  = String(req.query.length || "standard");
    if (!message) return res.status(400).json({ ok:false, error:"Missing 'message'." });

    if (DEMO) {
      return res.json({ ok:true, note:"DEMO_MODE=true", sampleRequest:{ model: MODEL, message, tone, length } });
    }
    if (!HAS_KEY) return res.status(400).json({ ok:false, error:"OPENAI_API_KEY missing" });
    if (!IS_GPT5) return res.status(400).json({ ok:false, error:`OPENAI_MODEL='${MODEL}' is not a GPT-5 model` });

    const data = await fetchJSON("https://api.openai.com/v1/responses", {
      model: MODEL,
      input: messagesToPlainString(buildMessages(extractUserRequest(message), tone, length)),
      max_output_tokens: clampTokens(applyMultiplier(tokensForLength(length)))
    });

    res.json({
      ok: true,
      sample_text: extractResponseText(data),
      incompleteReason: getIncompleteReason(data),
      response: data
    });
  } catch (e) {
    if (DEBUG) console.error("api/raw GET error:", e);
    res.status(500).json({ ok:false, error: String(e.message || e) });
  }
});


/* ---------------- /api/chat ---------------- */
app.post("/api/chat", async (req, res) => {
  try {
    const { message, tone = "balanced", length = "standard" } = req.body || {};
    if (!message || typeof message !== "string") return res.status(400).json({ error: "Missing 'message'." });

    if (DEMO) {
      const title = `Quick draft: ${message.slice(0, 60)}`;
      const demo = `# ${title}\n\n- Demo mode.\n- Set DEMO_MODE=false to use the model.`;
      return res.json({ reply: demo });
    }
    if (!HAS_KEY) return res.json({ reply: "⚠️ OPENAI_API_KEY missing on server." });
    if (!IS_GPT5) return res.json({ reply: `⚠️ OPENAI_MODEL='${MODEL}' is not a GPT-5 model. Set OPENAI_MODEL=gpt-5-nano in .env.` });

    const userText = extractUserRequest(message);
    const messages = buildMessages(userText, tone, length);

    const tokenBudget = applyMultiplier(tokensForLength(length));
    const { text, incompleteReason } = await completeWithAutoContinue(messages, tokenBudget, { rounds: AUTOCONTINUE_ROUNDS });

    const reply = text || "⚠️ No content returned by GPT-5.";
    const meta = incompleteReason ? `\n\n> _Note: model stopped early (${incompleteReason}); auto-continued ${AUTOCONTINUE_ROUNDS}×._` : "";
    res.json({ reply: reply + (text ? "" : meta) });
  } catch (err) {
    if (DEBUG) console.error("Server error:", err);
    res.json({ reply: `⚠️ ${String(err.message || err)}` });
  }
});

/* ---------------- /api/stream (pseudo-stream) ---------------- */
async function handleStream(message, tone, length, _req, res) {
  res.writeHead(200, sseHeaders());
  if (!message) { res.write(`data: Missing 'message'.\n\n`); res.write(`data: [DONE]\n\n`); return res.end(); }

  if (DEMO) {
    const text = `# Demo stream\n\nYou sent: ${message}`;
    for (const ch of chunkForSSE(text, 180)) { res.write(`data: ${escapeSSE(ch)}\n\n`); await delay(8); }
    res.write(`data: [DONE]\n\n`); return res.end();
  }
  if (!HAS_KEY) { res.write(`data: ⚠️ OPENAI_API_KEY missing.\n\n`); res.write(`data: [DONE]\n\n`); return res.end(); }
  if (!IS_GPT5) { res.write(`data: ⚠️ Set OPENAI_MODEL=gpt-5-nano.\n\n`); res.write(`data: [DONE]\n\n`); return res.end(); }

  const userText = extractUserRequest(message);
  const messages = buildMessages(userText, tone, length);

  try {
    const tokenBudget = applyMultiplier(tokensForLength(length));
    const { text, incompleteReason } = await completeWithAutoContinue(messages, tokenBudget, { rounds: AUTOCONTINUE_ROUNDS });
    const out = text && String(text).trim();
    if (!out) {
      res.write(`data: ⚠️ No content returned by GPT-5.\n\n`);
      res.write(`data: [DONE]\n\n`);
      return res.end();
    }
    for (const part of chunkForSSE(out, 200)) res.write(`data: ${escapeSSE(part)}\n\n`);
    if (incompleteReason) res.write(`data: \n\n> _Auto-continued due to ${incompleteReason}._\n\n`);
    res.write(`data: [DONE]\n\n`);
    res.end();
  } catch (e) {
    if (DEBUG) console.error("OpenAI stream error:", e);
    res.write(`data: ⚠️ ${String(e.message || e)}\n\n`);
    res.write(`data: [DONE]\n\n`);
    res.end();
  }
}

app.get("/api/stream", async (req, res) => {
  try {
    const message = String(req.query.message || "");
    const tone    = String(req.query.tone || "balanced");
    const length  = String(req.query.length || "standard");
    await handleStream(message, tone, length, req, res);
  } catch (err) {
    if (DEBUG) console.error("Stream server error (GET):", err);
    try { res.write(`data: ⚠️ Server error while streaming.\n\ndata: [DONE]\n\n`); res.end(); } catch {}
  }
});
app.post("/api/stream", async (req, res) => {
  try {
    const { message = "", tone = "balanced", length = "standard" } = req.body || {};
    await handleStream(String(message), String(tone), String(length), req, res);
  } catch (err) {
    if (DEBUG) console.error("Stream server error (POST):", err);
    try { res.writeHead(200, sseHeaders()); res.write(`data: ⚠️ Server error while streaming.\n\ndata: [DONE]\n\n`); res.end(); } catch {}
  }
});

/* ---------------- Save/list/delete transcripts ---------------- */
app.post("/api/save", async (req, res) => {
  try {
    const { transcript = [], format = "pdf", title = "hanza_session" } = req.body || {};
    if (!Array.isArray(transcript) || !transcript.length) return res.json({ ok:false, error:"empty transcript" });
    if (transcript.length > 2000) return res.status(413).json({ ok:false, error:"transcript too large" });

    const safe = s => (s || "").replace(/[^\w\- ]+/g, "").trim().replace(/\s+/g, "_").slice(0, 60);
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    const base  = `${safe(title) || "hanza_session"}__${stamp}`;

    const jsonName = `${base}.json`;
    const jsonPath = safePathJoin(TRANSCRIPTS, jsonName);
    const jsonDoc = { title, savedAt: new Date().toISOString(), transcript };
    fs.writeFileSync(jsonPath, JSON.stringify(jsonDoc, null, 2), "utf8");

    if (format === "md") {
      const mdName = `${base}.md`;
      const mdPath = safePathJoin(TRANSCRIPTS, mdName);
      const md = transcript.map(t => `${t.role === "user" ? "You" : "Hanza"}:\n${t.text}\n`).join("\n---\n\n");
      fs.writeFileSync(mdPath, md, "utf8");
      return res.json({ ok:true, file: mdName, url: `/transcripts/${encodeURIComponent(mdName)}`, json: `/transcripts/${encodeURIComponent(jsonName)}` });
    }

    const pdfName = `${base}.pdf`;
    const pdfPath = safePathJoin(TRANSCRIPTS, pdfName);
    const doc = new PDFDocument({ margin: 50 });
    const stream = fs.createWriteStream(pdfPath);
    doc.pipe(stream);

    doc.fontSize(18).text(title, { underline: false });
    doc.moveDown(0.5);
    transcript.forEach(({ role, text }) => {
      doc.fontSize(12).fillColor("#555").text(role === "user" ? "You" : "Hanza");
      doc.moveDown(0.1);
      doc.fontSize(12).fillColor("#000").text(String(text || ""), { paragraphGap: 8 });
      doc.moveDown(0.6);
    });

    doc.end();
    stream.on("finish", () => res.json({ ok:true, file: pdfName, url: `/transcripts/${encodeURIComponent(pdfName)}`, json: `/transcripts/${encodeURIComponent(jsonName)}` }));
    stream.on("error", (e) => res.status(500).json({ ok:false, error: String(e) }));
  } catch (e) {
    res.status(500).json({ ok:false, error: String(e) });
  }
});
app.get("/api/transcripts", async (_req, res) => {
  try {
    const files = fs.readdirSync(TRANSCRIPTS).filter(f => f.endsWith(".pdf") || f.endsWith(".md"));
    const list = files.map(f => {
      const st = fs.statSync(path.join(TRANSCRIPTS, f));
      const baseNoExt = f.replace(/\.(pdf|md)$/i, "");
      const jsonName = `${baseNoExt}.json`;
      const jsonExists = fs.existsSync(path.join(TRANSCRIPTS, jsonName));
      return { name: f, size: st.size, mtime: st.mtime, url: `/transcripts/${encodeURIComponent(f)}`, jsonUrl: jsonExists ? `/transcripts/${encodeURIComponent(jsonName)}` : null };
    }).sort((a,b)=> b.mtime - a.mtime);
    res.json({ ok:true, files: list });
  } catch (e) {
    res.status(500).json({ ok:false, error: String(e) });
  }
});
app.delete("/api/transcripts/:file", (req, res) => {
  try {
    const raw = req.params.file;
    const file = safeBasename(raw);
    const target = safePathJoin(TRANSCRIPTS, file);
    if (!fs.existsSync(target)) return res.status(404).json({ ok:false, error:"Not found" });
    fs.unlinkSync(target);
    const baseNoExt = file.replace(/\.(pdf|md)$/i, "");
    const jsonPath = safePathJoin(TRANSCRIPTS, `${baseNoExt}.json`);
    if (fs.existsSync(jsonPath)) fs.unlinkSync(jsonPath);
    res.json({ ok:true });
  } catch (e) {
    res.status(500).json({ ok:false, error: String(e) });
  }
});

app.listen(PORT, () => {
  console.log(`✅ Server running at http://localhost:${PORT}`);
  console.log("Using model:", MODEL, "| DEMO_MODE:", DEMO, "| api_mode:", IS_GPT5 ? "responses" : "chat");
  if (!HAS_KEY) console.log("⚠️ OPENAI_API_KEY is missing.");
});
