// scripts/build_index.mjs
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);
const ROOT       = path.join(__dirname, "..");
const SRC_DIR    = path.join(ROOT, "data", "sources");
const OUT_FILE   = path.join(ROOT, "data", "index.json");

// Config
const EMBED_MODEL = process.env.OPENAI_EMBED_MODEL || "text-embedding-3-small";
const MAX_CHUNK_CHARS = 1100;         // ~700–800 tokens-ish
const CHUNK_OVERLAP   = 120;

// Ensure dirs
fs.mkdirSync(path.join(ROOT, "data"), { recursive: true });

if (!process.env.OPENAI_API_KEY) {
  console.error("❌ OPENAI_API_KEY missing. Set it in your .env when running: node --env-file=.env scripts/build_index.mjs");
  process.exit(1);
}

// Utilities
function chunkText(txt, max = MAX_CHUNK_CHARS, overlap = CHUNK_OVERLAP) {
  const clean = txt.replace(/\r/g, "").trim();
  if (clean.length <= max) return [clean];

  const chunks = [];
  let i = 0;
  while (i < clean.length) {
    const slice = clean.slice(i, i + max);
    chunks.push(slice);
    i += max - overlap;
  }
  return chunks;
}

async function embedBatch(texts) {
  const body = {
    input: texts,
    model: EMBED_MODEL
  };
  const r = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body)
  });
  if (!r.ok) {
    const msg = await r.text().catch(()=> "");
    throw new Error(`Embeddings API error ${r.status}: ${msg}`);
  }
  const data = await r.json();
  return data.data.map(d => d.embedding);
}

// Main
async function main() {
  const files = fs.readdirSync(SRC_DIR).filter(f => f.toLowerCase().endsWith(".txt"));
  if (!files.length) {
    console.error(`❌ No .txt files found in ${SRC_DIR}. Add your posts first.`);
    process.exit(1);
  }

  console.log(`📚 Building index from ${files.length} source file(s)…`);

  const records = [];
  for (const file of files) {
    const full = path.join(SRC_DIR, file);
    const text = fs.readFileSync(full, "utf8");
    const title = file.replace(/\.txt$/i, "");
    const chunks = chunkText(text);
    console.log(` - ${file}: ${chunks.length} chunk(s)`);

    // Embed in batches (keep it simple; all at once is fine for small sets)
    const embeddings = await embedBatch(chunks);

    chunks.forEach((content, i) => {
      records.push({
        id: `${file}#${i}`,
        file,
        title,
        content,
        embedding: embeddings[i],
        chars: content.length
      });
    });
  }

  fs.writeFileSync(OUT_FILE, JSON.stringify({
    model: EMBED_MODEL,
    createdAt: new Date().toISOString(),
    count: records.length,
    records
  }, null, 2));

  console.log(`✅ Wrote ${records.length} chunk(s) → ${OUT_FILE}`);
}

main().catch(e => {
  console.error("❌ Build failed:", e);
  process.exit(1);
});
