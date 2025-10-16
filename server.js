import express from "express";
import dotenv from "dotenv";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import PDFDocument from "pdfkit";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const app   = express();
const PORT  = process.env.PORT || 3000;
const MODEL = process.env.OPENAI_MODEL || "gpt-4o-mini";
const DEMO  = String(process.env.DEMO_MODE || "").toLowerCase() === "true";

const EMBED_MODEL = process.env.OPENAI_EMBED_MODEL || "text-embedding-3-small";

// ---- Paths
const PUBLIC_DIR   = path.join(__dirname, "public");
const DATA_DIR     = path.join(__dirname, "data");
const SRC_DIR      = path.join(DATA_DIR, "sources");
const INDEX_FILE   = path.join(DATA_DIR, "index.json");
const STYLE_FILE   = path.join(PUBLIC_DIR, "assets", "style_guide.md");
const TRANSCRIPTS  = path.join(DATA_DIR, "transcripts");

// Ensure dirs
fs.mkdirSync(DATA_DIR, { recursive: true });
fs.mkdirSync(SRC_DIR,  { recursive: true });
fs.mkdirSync(TRANSCRIPTS, { recursive: true });

app.use(express.json({ limit: "2mb" }));
app.use(express.static(PUBLIC_DIR));
app.use("/transcripts", express.static(TRANSCRIPTS)); // serve PDFs + JSON

// ---- Load style guide once (with a safe cap)
let STYLE_GUIDE = "";
try {
  STYLE_GUIDE = fs.readFileSync(STYLE_FILE, "utf8").slice(0, 8000);
  console.log("✅ Loaded style guide.");
} catch { console.warn("ℹ️ No style_guide.md found. (Optional)"); }

// ---- Load RAG index (if available)
let RAG_INDEX = null;
try {
  RAG_INDEX = JSON.parse(fs.readFileSync(INDEX_FILE, "utf8"));
  console.log(`✅ Loaded RAG index: ${RAG_INDEX.count} chunk(s).`);
} catch {
  console.warn("ℹ️ No data/index.json found. Retrieval disabled until you build the index.");
}

// ===== Utilities =====
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
    ultra:    "Target 2000+ words with exhaustive coverage and case-style depth."
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
function escapeSSE(s = "") { return s.replace(/\u0000/g, ""); }
function delay(ms) { return new Promise(r => setTimeout(r, ms)); }
function capitalize(s) { return (s || "").charAt(0).toUpperCase() + (s || "").slice(1); }

// ===== DEMO draft for offline mode =====
function demoDraft(topic, tone, length) {
  const style = styleInstruction(tone, length);
  const title = `A Practical Guide to ${capitalize(topic)}`;
  return [
    `# ${title}`,
    ``,
    `**Style:** ${style}`,
    ``,
    `**Overview**`,
    `In this post, we unpack ${topic} with a focus on practical moves you can apply this week.`,
    ``,
    `## Why it matters`,
    `- Clear business impact`,
    `- Common traps to avoid`,
    `- What “good” looks like in practice`,
    ``,
    `## 3 actionable steps`,
    `1) Define the outcome: Write a one-sentence success statement.`,
    `2) Pick one metric: Choose a single KPI that proves you’re moving.`,
    `3) Ship a small test: Validate with a no-regret experiment in 7 days.`,
    ``,
    `## Quick template`,
    `- Audience:`,
    `- Problem:`,
    `- Desired outcome:`,
    `- First experiment:`,
    ``,
    `**Bottom line:** Start small, measure, iterate.`
  ].join("\n");
}
async function streamDemoDraft(res, topic, tone, length) {
  const text = demoDraft(topic, tone, length);
  for (const ch of text) { res.write(`data: ${escapeSSE(ch)}\n\n`); await delay(5); }
  res.write(`data: [DONE]\n\n`); res.end();
}

// ====== Retrieval helpers ======
function cosineSim(a = [], b = []) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i], y = b[i];
    dot += x * y;
    na  += x * x;
    nb  += y * y;
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}

async function embedText(text) {
  const r = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ input: text, model: EMBED_MODEL })
  });
  if (!r.ok) {
    const msg = await r.text().catch(()=> "");
    throw new Error(`Embeddings API error ${r.status}: ${msg}`);
  }
  const data = await r.json();
  return data.data[0].embedding;
}

/** Retrieve topK similar chunks from index. */
async function retrieveSimilar(query, { topK = 4, maxCharsTotal = 3000 } = {}) {
  if (!RAG_INDEX?.records?.length) return [];

  const qVec = await embedText(query);
  const scored = RAG_INDEX.records.map(r => ({
    ...r,
    score: cosineSim(qVec, r.embedding)
  })).sort((a,b) => b.score - a.score);

  const results = [];
  let used = 0;
  for (const rec of scored) {
    if (results.length >= topK) break;
    const remaining = Math.max(0, maxCharsTotal - used);
    if (remaining < 200) break;
    const snippet = rec.content.slice(0, remaining);
    results.push({
      title: rec.title,
      file: rec.file,
      content: snippet,
      score: rec.score
    });
    used += snippet.length;
  }
  return results;
}

function buildMessages(userPrompt, tone, length, retrieved = []) {
  const style = styleInstruction(tone, length);

  const baseSystem =
    "You are a blog writing assistant trained in Hanza Stephens' voice and style. " +
    "Write as a calm, strategic operator: practical, structured, and insightful. " +
    "Favor clear section headings, bullets, and concrete frameworks over fluff.";

  const guide = STYLE_GUIDE
    ? `\n\n### VOICE GUIDE (Highest priority)\n${STYLE_GUIDE}\n\nFollow this guide strictly.`
    : "";

  const formattingRules =
    "Format with clean markdown: use H1/H2/H3 headings, bullets, numbered steps, and code fences for templates where helpful. " +
    "Avoid emojis unless the user asks. Keep advice concrete and de-jargonized.";

  let refs = "";
  if (retrieved?.length) {
    const blocks = retrieved.map((r, i) =>
      `— Source ${i+1}: ${r.title}\n${r.content.trim()}`
    ).join("\n\n");
    refs = `\n\n### REFERENCE EXCERPTS (Use for tone & examples; do not cite verbatim)\n${blocks}`;
  }

  return [
    { role: "system", content: `${baseSystem}${guide}\n\n${formattingRules}${refs}` },
    { role: "system", content: `Stylistic guidance: ${style}` },
    { role: "user",   content: userPrompt }
  ];
}

// ====== Health + config ======
app.get("/health", (_req, res) => res.json({ ok: true, port: PORT }));
app.get("/config", (_req, res) => res.json({
  model: MODEL, demo: DEMO,
  rag: Boolean(RAG_INDEX?.records?.length),
  embed_model: EMBED_MODEL
}));

// Optional: reload style and index without restarting
app.post("/admin/reload-style", (_req, res) => {
  try {
    STYLE_GUIDE = fs.readFileSync(STYLE_FILE, "utf8").slice(0, 8000);
    res.json({ ok: true, bytes: STYLE_GUIDE.length });
  } catch (e) {
    res.status(500).json({ ok: false, error: String(e) });
  }
});
app.post("/admin/reload-index", (_req, res) => {
  try {
    RAG_INDEX = JSON.parse(fs.readFileSync(INDEX_FILE, "utf8"));
    res.json({ ok: true, count: RAG_INDEX.count });
  } catch (e) {
    res.status(500).json({ ok: false, error: String(e) });
  }
});

// ====== Non-stream (fallback) ======
app.post("/api/chat", async (req, res) => {
  try {
    const { message, tone = "balanced", length = "standard" } = req.body || {};
    if (!message) return res.status(400).json({ error: "Missing 'message'." });

    if (DEMO) return res.json({ reply: demoDraft(message, tone, length) });

    // Skip server retrieval if client already stuffed context
    const clientStuffed = /CONTEXT EXCERPTS:/i.test(message);
    const retrieved = clientStuffed ? [] : await retrieveSimilar(message, { topK: 4, maxCharsTotal: 3000 });

    const resp = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: MODEL,
        messages: buildMessages(message, tone, length, retrieved),
        temperature: 0.7,
        max_tokens: 1500
      })
    });

    if (!resp.ok) {
      const txt = await resp.text().catch(() => "");
      console.error("OpenAI error:", resp.status, txt);
      if (resp.status === 404) return res.json({ reply: `⚠️ Model "${MODEL}" unavailable. Try OPENAI_MODEL=gpt-4o-mini.` });
      if (resp.status === 429) return res.json({ reply: "⚠️ No remaining credit. Add billing or set DEMO_MODE=true." });
      return res.json({ reply: "⚠️ The AI service returned an error. Please try again." });
    }

    const data = await resp.json();
    const reply = data?.choices?.[0]?.message?.content?.trim() || "⚠️ No content returned.";
    res.json({ reply });
  } catch (err) {
    console.error("Server error:", err);
    res.json({ reply: "⚠️ Server error. Check logs." });
  }
});

// ====== Stream (SSE) ======
app.get("/api/stream", async (req, res) => {
  try {
    const message = String(req.query.message || "");
    const tone    = String(req.query.tone || "balanced");
    const length  = String(req.query.length || "standard");

    res.writeHead(200, sseHeaders());
    if (!message) {
      res.write(`data: Missing 'message'.\n\n`);
      res.write(`data: [DONE]\n\n`);
      return res.end();
    }

    if (DEMO) return streamDemoDraft(res, message, tone, length);

    const clientStuffed = /CONTEXT EXCERPTS:/i.test(message);
    const retrieved = clientStuffed ? [] : await retrieveSimilar(message, { topK: 4, maxCharsTotal: 3000 });

    // --- NEW: Abort upstream call if the client disconnects
    const ac = new AbortController();
    const { signal } = ac;
    req.on("close", () => {
      try { ac.abort(); } catch {}
      try {
        res.write(`data: [DONE]\n\n`);
        res.end();
      } catch {}
    });

    const resp = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: MODEL,
        messages: buildMessages(message, tone, length, retrieved),
        temperature: 0.7,
        max_tokens: 1700,
        stream: true
      }),
      signal
    });

    if (!resp.ok || !resp.body) {
      const txt = await resp.text().catch(() => "");
      console.error("OpenAI stream error:", resp.status, txt);
      if (resp.status === 404) res.write(`data: ⚠️ Model "${MODEL}" unavailable.\n\n`);
      else if (resp.status === 429) res.write(`data: ⚠️ No project credit. Add billing or enable DEMO_MODE.\n\n`);
      else res.write(`data: ⚠️ Failed to start stream.\n\n`);
      res.write(`data: [DONE]\n\n`);
      return res.end();
    }

    const reader  = resp.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let leftover  = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      const lines = (leftover + chunk).split(/\r?\n/);
      leftover = lines.pop() || "";

      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") {
          res.write(`data: [DONE]\n\n`);
          res.end();
          return;
        }
        try {
          const json = JSON.parse(data);
          const token = json.choices?.[0]?.delta?.content || "";
          // --- NEW: simple, robust token write
          if (token) res.write(`data: ${escapeSSE(token)}\n\n`);
        } catch {
          // ignore keep-alives / non-JSON lines
        }
      }
    }

    res.write(`data: [DONE]\n\n`);
    res.end();
  } catch (err) {
    console.error("Stream server error:", err);
    try {
      res.write(`data: ⚠️ Server error while streaming.\n\n`);
      res.write(`data: [DONE]\n\n`);
      res.end();
    } catch {}
  }
});

// ====== Save transcript (PDF or MD) + JSON for Jump Back In ======
app.post("/api/save", async (req, res) => {
  try {
    const { transcript = [], format = "pdf", title = "hanza_session" } = req.body || {};
    if (!Array.isArray(transcript) || !transcript.length) return res.json({ ok:false, error:"empty transcript" });

    const safe = s => (s || "").replace(/[^\w\- ]+/g, "").trim().replace(/\s+/g, "_").slice(0, 60);
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    const base  = `${safe(title) || "hanza_session"}__${stamp}`;

    // Always write JSON (for Jump Back In)
    const jsonName = `${base}.json`;
    const jsonPath = path.join(TRANSCRIPTS, jsonName);
    const jsonDoc = {
      title,
      savedAt: new Date().toISOString(),
      transcript // [{role, text}]
    };
    fs.writeFileSync(jsonPath, JSON.stringify(jsonDoc, null, 2), "utf8");

    // If MD requested, write it; otherwise default to PDF
    if (format === "md") {
      const mdName = `${base}.md`;
      const mdPath = path.join(TRANSCRIPTS, mdName);
      const md = transcript.map(t => `${t.role === "user" ? "You" : "Hanza"}:\n${t.text}\n`).join("\n---\n\n");
      fs.writeFileSync(mdPath, md, "utf8");
      return res.json({
        ok:true,
        file: mdName,
        url: `/transcripts/${encodeURIComponent(mdName)}`,
        json: `/transcripts/${encodeURIComponent(jsonName)}`
      });
    }

    // PDF
    const pdfName = `${base}.pdf`;
    const pdfPath = path.join(TRANSCRIPTS, pdfName);
    const doc = new PDFDocument({ margin: 50 });
    const stream = fs.createWriteStream(pdfPath);
    doc.pipe(stream);

    doc.fontSize(18).text(title, { underline: false });
    doc.moveDown(0.5);
    transcript.forEach(({ role, text }) => {
      doc.fontSize(12).fillColor("#555").text(role === "user" ? "You" : "Hanza", { continued:false });
      doc.moveDown(0.1);
      doc.fontSize(12).fillColor("#000").text(text);
      doc.moveDown(0.6);
    });

    doc.end();
    stream.on("finish", () => {
      res.json({
        ok:true,
        file: pdfName,
        url: `/transcripts/${encodeURIComponent(pdfName)}`,
        json: `/transcripts/${encodeURIComponent(jsonName)}`
      });
    });
    stream.on("error", (e) => res.status(500).json({ ok:false, error: String(e) }));
  } catch (e) {
    res.status(500).json({ ok:false, error: String(e) });
  }
});

// ====== List transcripts (include jsonUrl) ======
app.get("/api/transcripts", async (_req, res) => {
  try {
    const files = fs.readdirSync(TRANSCRIPTS).filter(f => f.endsWith(".pdf") || f.endsWith(".md"));
    const list = files.map(f => {
      const st = fs.statSync(path.join(TRANSCRIPTS, f));
      const baseNoExt = f.replace(/\.(pdf|md)$/i, "");
      const jsonName = `${baseNoExt}.json`;
      const jsonExists = fs.existsSync(path.join(TRANSCRIPTS, jsonName));
      return {
        name: f,
        size: st.size,
        mtime: st.mtime,
        url: `/transcripts/${encodeURIComponent(f)}`,
        jsonUrl: jsonExists ? `/transcripts/${encodeURIComponent(jsonName)}` : null
      };
    }).sort((a,b)=> b.mtime - a.mtime);
    res.json({ ok:true, files: list });
  } catch (e) {
    res.status(500).json({ ok:false, error: String(e) });
  }
});

// ====== Delete transcript (also remove matching JSON) ======
app.delete("/api/transcripts/:file", (req, res) => {
  try {
    const file = req.params.file;
    const target = path.join(TRANSCRIPTS, file);
    if (!fs.existsSync(target)) return res.status(404).json({ ok:false, error:"Not found" });
    fs.unlinkSync(target);

    // Remove matching JSON if it exists
    const baseNoExt = file.replace(/\.(pdf|md)$/i, "");
    const jsonPath = path.join(TRANSCRIPTS, `${baseNoExt}.json`);
    if (fs.existsSync(jsonPath)) fs.unlinkSync(jsonPath);

    res.json({ ok:true });
  } catch (e) {
    res.status(500).json({ ok:false, error: String(e) });
  }
});

// ---- BEGIN: sources endpoint (clean) ----
app.get('/api/sources', async (_req, res) => {
  try {
    const files = (await fs.promises.readdir(SRC_DIR))
      .filter(f => /\.(txt|md)$/i.test(f))
      .sort();

    const items = await Promise.all(
      files.map(async (name) => {
        const text = await fs.promises.readFile(path.join(SRC_DIR, name), 'utf8');
        return { name, text };
      })
    );

    res.json({ ok: true, items });
  } catch (err) {
    console.error('sources error:', err);
    res.json({ ok: false, items: [] });
  }
});
// ---- END: sources endpoint ----

// ====== start ======
app.listen(PORT, () => {
  console.log(`✅ Server running at http://localhost:${PORT}`);
  console.log("Using model:", MODEL, "| DEMO_MODE:", DEMO, "| RAG:", Boolean(RAG_INDEX?.records?.length));
});
