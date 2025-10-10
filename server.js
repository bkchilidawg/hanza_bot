import express from "express";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// ---- choose model from env or default to one broadly available
const MODEL = process.env.OPENAI_MODEL || "gpt-4o-mini";

app.use(express.json());
app.use(express.static("public"));

// Health + config endpoints for quick debugging
app.get("/health", (_req, res) => res.json({ ok: true, port: PORT }));
app.get("/config", (_req, res) => res.json({ model: MODEL }));

app.post("/api/chat", async (req, res) => {
  try {
    const { message } = req.body || {};
    if (!message) return res.status(400).json({ error: "Missing 'message'." });

    const resp = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: MODEL,
        messages: [
          {
            role: "system",
            content:
              "You are a blog writing assistant trained in Hanza Stephens' voice and style. Be structured, insightful, and practical."
          },
          { role: "user", content: message }
        ],
        temperature: 0.75,
        max_tokens: 1000
      })
    });

    if (!resp.ok) {
      const txt = await resp.text();
      console.error("OpenAI error:", resp.status, txt);

      if (resp.status === 404 || /model.*not.*found/i.test(txt)) {
        return res.status(400).json({
          error:
            `The selected model is not available to your key. Current model: "${MODEL}". ` +
            `Try setting OPENAI_MODEL=gpt-4o-mini in your .env and restart the server.`
        });
      }
      return res.status(502).json({ error: "OpenAI request failed." });
    }

    const data = await resp.json();
    const reply =
      data?.choices?.[0]?.message?.content?.trim() ||
      "⚠️ No content returned from the AI.";
    res.json({ reply });
  } catch (err) {
    console.error("Server error:", err);
    res.status(500).json({ error: "Server error." });
  }
});

app.listen(PORT, () => {
  console.log(`✅ Server running at http://localhost:${PORT}`);
  console.log("Using model:", MODEL);
});
