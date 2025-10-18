// FILE: scripts/verify-gpt5.mjs
// Verifies gpt-5-nano via /v1/responses (no 'modalities')

import dotenv from "dotenv";
dotenv.config();

const API_KEY = process.env.OPENAI_API_KEY;
if (!API_KEY) {
  console.error("❌ Missing OPENAI_API_KEY in .env");
  process.exit(1);
}

function normalizeModelId(id) {
  const s = String(id || "").trim();
  if (/^gpt[-_]?5([-_]?nano)?$/i.test(s)) return "gpt-5-nano";
  if (/^gpt[-_]?5/i.test(s)) return s.replace(/_/g, "-").toLowerCase();
  return "gpt-5-nano";
}
const MODEL = normalizeModelId(process.env.OPENAI_MODEL);

const body = {
  model: MODEL,
  input: [
    { role: "user", content: [{ type: "input_text", text: "Say: hello from gpt-5-nano" }] }
  ],
  max_output_tokens: 64
};

try {
  const resp = await fetch("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: { Authorization: `Bearer ${API_KEY}`, "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  const text = await resp.text();
  if (!resp.ok) {
    console.error("❌ Responses error:", resp.status, text);
    process.exit(2);
  }
  const json = JSON.parse(text);
  const out =
    json.output_text ||
    (Array.isArray(json.output)
      ? json.output.flatMap(p => p.content || []).map(c => c.text || "").join("")
      : "");
  console.log("✅ GPT-5 reachable. Model:", MODEL);
  console.log("Output:", out || "[no output_text field]");
} catch (e) {
  console.error("❌ Network/parse error:", e.message);
  process.exit(3);
}
