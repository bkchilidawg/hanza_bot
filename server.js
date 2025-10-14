import express from "express";
import dotenv from "dotenv";

dotenv.config();

const app   = express();
const PORT  = process.env.PORT || 3000;
const MODEL = process.env.OPENAI_MODEL || "gpt-4o-mini";
const DEMO  = String(process.env.DEMO_MODE || "").toLowerCase() === "true";

app.use(express.json());
app.use(express.static("public"));

// ---------- helpers ----------
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

function buildMessages(userPrompt, tone, length) {
  const style = styleInstruction(tone, length);
  return [
    {
      role: "system",
      content:
        "You are a blog writing assistant trained in Hanza Stephens' voice and style. " +
        "Write as a calm, strategic operator: practical, structured, and insightful. " +
        "Favor clear section headings, bullets, and concrete frameworks over fluff."
    },
    { role: "system", content: `Stylistic guidance: ${style}` },
    { role: "user", content: userPrompt }
  ];
}

function sseHeaders() {
  return {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache, no-transform",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no"
  };
}

function escapeSSE(s = "") {
  // keep it simple: remove null chars that can break streams
  return s.replace(/\u0000/g, "");
}

function delay(ms) { return new Promise(r => setTimeout(r, ms)); }
function capitalize(s) { return (s || "").charAt(0).toUpperCase() + (s || "").slice(1); }

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
  for (const ch of text) {
    res.write(`data: ${escapeSSE(ch)}\n\n`);
    await delay(5);
  }
  res.write(`data: [DONE]\n\n`);
  res.end();
}

// ---------- diagnostics ----------
app.get("/health", (_req, res) => res.json({ ok: true, port: PORT }));
app.get("/config", (_req, res) => res.json({ model: MODEL, demo: DEMO }));

// ---------- non-stream endpoint (used when "Stream" toggle is off) ----------
app.post("/api/chat", async (req, res) => {
  try {
    const { message, tone = "balanced", length = "standard" } = req.body || {};
    if (!message) return res.status(400).json({ error: "Missing 'message'." });

    if (DEMO) {
      return res.json({ reply: demoDraft(message, tone, length) });
    }

    const resp = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: MODEL,
        messages: buildMessages(message, tone, length),
        temperature: 0.75,
        max_tokens: 1200
      })
    });

    if (!resp.ok) {
      const txt = await resp.text().catch(() => "");
      console.error("OpenAI error:", resp.status, txt);

      if (resp.status === 404 || /model.*not.*found/i.test(txt)) {
        return res.status(200).json({
          reply: `⚠️ The selected model is not available to this key. Current model: "${MODEL}". Try OPENAI_MODEL=gpt-4o-mini in your .env, then restart.`
        });
      }
      if (resp.status === 429 || /insufficient_quota|quota/i.test(txt)) {
        return res.status(200).json({
          reply: "⚠️ Your OpenAI project has no remaining credit. Add billing or set DEMO_MODE=true to test."
        });
      }
      return res.status(200).json({ reply: "⚠️ The AI service returned an error. Please try again." });
    }

    const data = await resp.json();
    const reply = data?.choices?.[0]?.message?.content?.trim() || "⚠️ No content returned from the AI.";
    res.json({ reply });
  } catch (err) {
    console.error("Server error:", err);
    res.status(200).json({ reply: "⚠️ Server error. Check the console for details." });
  }
});

// ---------- stream endpoint (SSE) for the UI "Stream" mode ----------
app.get("/api/stream", async (req, res) => {
  try {
    const message = req.query.message || "";
    const tone    = req.query.tone || "balanced";
    const length  = req.query.length || "standard";
    res.writeHead(200, sseHeaders());

    if (!message) {
      res.write(`data: Missing 'message'.\n\n`);
      res.write(`data: [DONE]\n\n`);
      return res.end();
    }

    if (DEMO) {
      await streamDemoDraft(res, message, tone, length);
      return;
    }

    const resp = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: MODEL,
        messages: buildMessages(message, tone, length),
        temperature: 0.75,
        max_tokens: 1500,
        stream: true
      })
    });

    if (!resp.ok || !resp.body) {
      const txt = await resp.text().catch(() => "");
      console.error("OpenAI stream error:", resp.status, txt);

      if (resp.status === 404 || /model.*not.*found/i.test(txt)) {
        res.write(`data: ⚠️ Model unavailable: "${MODEL}". Try OPENAI_MODEL=gpt-4o-mini in .env.\n\n`);
      } else if (resp.status === 429 || /insufficient_quota|quota/i.test(txt)) {
        res.write(`data: ⚠️ No remaining credit. Add billing or set DEMO_MODE=true.\n\n`);
      } else {
        res.write(`data: ⚠️ Failed to start stream.\n\n`);
      }
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
          if (token) res.write(`data: ${escapeSSE(token)}\n\n`);
        } catch {
          // ignore keep-alives/heartbeat lines
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

// ---------- start ----------
app.listen(PORT, () => {
  console.log(`✅ Server running at http://localhost:${PORT}`);
  console.log("Using model:", MODEL, "| DEMO_MODE:", DEMO);
});
