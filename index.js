// YouTube Creator AI Platform (extended)
// Run: npm install express openai cosine-similarity uuid youtube-transcript node-cache @google/generative-ai
// Env: YT_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY (optional for LLM/embeddings),
//      ELEVEN_API_KEY (tts), HUGGINGFACE_API_KEY (tts), KOKORO_URL (optional HTTP tts endpoint)
// Start: node index.js

require("dotenv").config();
const path = require("path");
const express = require("express");
const { OpenAI } = require("openai");
const cosine = require("cosine-similarity");
const { v4: uuid } = require("uuid");
// youtube-transcript is ESM-only; load lazily via dynamic import inside fetchTranscript
const NodeCache = require("node-cache");
const { GoogleGenerativeAI } = require("@google/generative-ai");

const app = express();
app.use(express.json({ limit: "2mb" }));
app.use(express.static("public"));

const openai = process.env.OPENAI_API_KEY ? new OpenAI({ apiKey: process.env.OPENAI_API_KEY }) : null;
const gemini = process.env.GEMINI_API_KEY ? new GoogleGenerativeAI(process.env.GEMINI_API_KEY) : null;

const channels = new Map();
const videos = new Map();
const clusters = new Map();
const styles = new Map();
const bulkImageQueue = [];
const cache = new NodeCache({ stdTTL: 300 });

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const norm = (arr) => {
  const min = Math.min(...arr), max = Math.max(...arr);
  return arr.map(v => max === min ? 0.5 : (v - min) / (max - min));
};
async function embed(text) {
  if (!openai) throw new Error("Set OPENAI_API_KEY for embeddings");
  const res = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text.slice(0, 1000)
  });
  return res.data[0].embedding;
}

// ---- YouTube Data API ----
async function ytFetch(endpoint, params) {
  const query = new URLSearchParams({ key: process.env.YT_API_KEY, ...params });
  const url = `https://www.googleapis.com/youtube/v3/${endpoint}?${query.toString()}`;
  const hit = cache.get(url);
  if (hit) return hit;
  const r = await fetch(url);
  if (!r.ok) throw new Error(`YouTube API ${endpoint} failed: ${r.status}`);
  const data = await r.json();
  cache.set(url, data);
  return data;
}
async function fetchChannel(channelId) {
  const data = await ytFetch("channels", { part: "snippet,statistics", id: channelId });
  const c = data.items?.[0];
  if (!c) throw new Error("Channel not found");
  const subs = Number(c.statistics.subscriberCount || 0);
  const medianViews = Number(c.statistics.viewCount || 1) / Math.max(Number(c.statistics.videoCount || 1), 1);
  const chObj = { id: channelId, handle: c.snippet.customUrl || c.snippet.title, subs, medianViews, rpmEst: 5 };
  channels.set(channelId, chObj);
  return chObj;
}
async function fetchRecentVideos(channelId, days = 7, max = 25) {
  const pubAfter = new Date(Date.now() - days * 86400 * 1000).toISOString();
  const search = await ytFetch("search", { part: "id", channelId, publishedAfter: pubAfter, order: "date", maxResults: max, type: "video" });
  const ids = search.items.map(i => i.id.videoId).join(",");
  if (!ids) return [];
  const vd = await ytFetch("videos", { part: "snippet,statistics,contentDetails", id: ids });
  const now = Date.now();
  return vd.items.map(v => {
    const views = Number(v.statistics.viewCount || 0);
    const publishedAt = v.snippet.publishedAt;
    const ageHours = (now - new Date(publishedAt).getTime()) / 3.6e6;
    const vph = ageHours > 0 ? views / ageHours : views;
    return {
      channelId,
      ytId: v.id,
      title: v.snippet.title,
      description: v.snippet.description,
      views,
      vph,
      subs: channels.get(channelId)?.subs || 0,
      publishedAt
    };
  });
}

// Simpler trending/outlier fetch using mostPopular feed
async function fetchTrendingOutliers(regionCode = "US", maxResults = 25) {
  if (!process.env.YT_API_KEY) throw new Error("YT_API_KEY not set");
  const popular = await ytFetch("videos", {
    chart: "mostPopular",
    part: "id,snippet,statistics",
    maxResults,
    regionCode
  });
  const vids = popular.items || [];
  const channelIds = [...new Set(vids.map(v => v.snippet.channelId))];
  const channelsResp = channelIds.length
    ? await ytFetch("channels", { part: "snippet,statistics", id: channelIds.join(",") })
    : { items: [] };
  const channelMap = new Map(
    (channelsResp.items || []).map(c => [
      c.id,
      {
        title: c.snippet.title,
        subs: Number(c.statistics.subscriberCount || 0),
        videoCount: Number(c.statistics.videoCount || 0)
      }
    ])
  );
  const now = Date.now();
  const videoScores = vids.map(v => {
    const views = Number(v.statistics.viewCount || 0);
    const publishedAt = v.snippet.publishedAt;
    const ageHours = Math.max((now - new Date(publishedAt).getTime()) / 3.6e6, 0.01);
    const vph = views / ageHours;
    const ch = channelMap.get(v.snippet.channelId) || { subs: 0, title: v.snippet.channelTitle };
    const smallBoost = clamp((100000 - ch.subs) / 100000, 0, 1) * 0.4;
    const score = vph * (1 + smallBoost);
    return {
      id: v.id,
      title: v.snippet.title,
      channelTitle: ch.title,
      channelId: v.snippet.channelId,
      views,
      vph,
      ageHours,
      publishedAt,
      subs: ch.subs || 0,
      score,
      url: `https://www.youtube.com/watch?v=${v.id}`
    };
  });
  const topVideos = videoScores.sort((a, b) => b.score - a.score);

  const channelAgg = new Map();
  topVideos.forEach(v => {
    const entry = channelAgg.get(v.channelId) || { channelId: v.channelId, channelTitle: v.channelTitle, score: 0, videos: 0 };
    entry.score += v.score;
    entry.videos += 1;
    channelAgg.set(v.channelId, entry);
  });
  const topChannels = Array.from(channelAgg.values()).sort((a, b) => b.score - a.score);

  return { videos: topVideos, channels: topChannels };
}

// Search videos by query, rank by views/hour
async function searchVideos(query, maxResults = 25, regionCode = "US", order = "viewCount", days = 1000000) {
  if (!process.env.YT_API_KEY) throw new Error("YT_API_KEY not set");
  const params = {
    part: "id,snippet",
    q: query,
    type: "video",
    maxResults: Math.min(maxResults, 50),
    order,
    regionCode
  };
  if (days < 100000) {
    const pubAfter = new Date(Date.now() - days * 86400 * 1000).toISOString();
    params.publishedAfter = pubAfter;
  }
  const search = await ytFetch("search", {
    ...params
  });
  const ids = (search.items || []).map(i => i.id.videoId).filter(Boolean);
  if (!ids.length) return [];
  const details = await ytFetch("videos", {
    part: "snippet,statistics",
    id: ids.join(",")
  });
  const now = Date.now();
  return (details.items || []).map(v => {
    const views = Number(v.statistics.viewCount || 0);
    const publishedAt = v.snippet.publishedAt;
    const ageHours = Math.max((now - new Date(publishedAt).getTime()) / 3.6e6, 0.01);
    const vph = views / ageHours;
    const subs = Number(v.statistics?.subscriberCount || 0); // may be empty; keep for ratio checks client-side
    return {
      id: v.id,
      title: v.snippet.title,
      channelTitle: v.snippet.channelTitle,
      channelId: v.snippet.channelId,
      views,
      vph,
      ageHours,
      publishedAt,
      subs,
      url: `https://www.youtube.com/watch?v=${v.id}`
    };
  }).sort((a, b) => b.vph - a.vph);
}

// Category-based outliers with timeframe filter
async function fetchCategoryOutliers({ regionCode = "US", categoryId = "0", days = 7, maxResults = 50 }) {
  if (!process.env.YT_API_KEY) throw new Error("YT_API_KEY not set");
  const popular = await ytFetch("videos", {
    chart: "mostPopular",
    part: "id,snippet,statistics",
    maxResults: Math.min(maxResults, 50),
    regionCode,
    ...(categoryId !== "0" ? { videoCategoryId: categoryId } : {})
  });
  const now = Date.now();
  const items = (popular.items || []).map(v => {
    const views = Number(v.statistics.viewCount || 0);
    const publishedAt = v.snippet.publishedAt;
    const ageHours = Math.max((now - new Date(publishedAt).getTime()) / 3.6e6, 0.01);
    const vph = views / ageHours;
    return {
      id: v.id,
      title: v.snippet.title,
      channelTitle: v.snippet.channelTitle,
      channelId: v.snippet.channelId,
      views,
      vph,
      ageHours,
      publishedAt,
      url: `https://www.youtube.com/watch?v=${v.id}`
    };
  }).filter(v => v.ageHours <= days * 24);
  const scored = items.map(v => {
    const smallBoost = clamp((100000 - (v.subs || 0)) / 100000, 0, 1) * 0.3;
    const score = v.vph * (1 + smallBoost);
    return { ...v, score };
  }).sort((a, b) => b.score - a.score);
  return scored;
}
async function fetchTranscript(ytId, lang = "en") {
  try {
    const { YoutubeTranscript } = await import("youtube-transcript");
    const transcript = await YoutubeTranscript.fetchTranscript(ytId, { lang, country: "US" });
    return transcript.map(t => t.text).join(" ");
  } catch (e) {
    try {
      const alt = await fetch(`https://youtubetranscript.com/?server_vid=${ytId}`);
      if (alt.ok) {
        const txt = await alt.text();
        try {
          const json = JSON.parse(txt);
          if (Array.isArray(json)) {
            return json.map(item => item.text).join(" ");
          }
        } catch {
          // fall through if not JSON
        }
      }
      return "";
    } catch {
      return "";
    }
  }
}

// ---- Clustering ----
async function addVideo(payload) {
  const id = uuid();
  const emb = await embed(`${payload.title}\n${payload.description || ""}`);
  const video = { id, embedding: emb, clusterId: null, ...payload };
  videos.set(id, video);
  return video;
}
function clusterVideos(k = 25) {
  const vids = Array.from(videos.values());
  if (!vids.length) return;
  const centroids = vids.slice(0, Math.min(k, vids.length)).map(v => [...v.embedding]);
  for (let iter = 0; iter < 3; iter++) {
    const buckets = centroids.map(() => []);
    vids.forEach(v => {
      const idx = centroids
        .map(c => cosine(v.embedding, c))
        .reduce((best, val, i, arr) => (val > arr[best] ? i : best), 0);
      buckets[idx].push(v.embedding);
      v.clusterId = idx;
    });
    centroids.forEach((c, i) => {
      const b = buckets[i];
      if (!b.length) return;
      for (let d = 0; d < c.length; d++) c[d] = b.reduce((s, v) => s + v[d], 0) / b.length;
    });
  }
  centroids.forEach((c, i) => {
    const clusterId = i.toString();
    const vidsIn = vids.filter(v => v.clusterId === i);
    const channelIds = new Set(vidsIn.map(v => v.channelId));
    const vphSum = vidsIn.reduce((s, v) => s + (v.vph || 0), 0);
    const growthScore = vphSum / (vidsIn.length || 1);
    const compScore = Math.log(channelIds.size + 1) / ((vidsIn.length || 1) + 1);
    clusters.set(clusterId, {
      id: clusterId,
      label: `Niche ${i}`,
      centroid: c,
      rpmEst: 5 + Math.random() * 8,
      growthScore,
      compScore,
      channelIds: Array.from(channelIds)
    });
  });
}
app.get("/v1/niches", (_req, res) => {
  const list = Array.from(clusters.values());
  const growth = norm(list.map(c => c.growthScore || 0));
  const comp = norm(list.map(c => c.compScore || 0));
  const rpm = norm(list.map(c => c.rpmEst || 0));
  const scored = list.map((c, i) => ({
    ...c,
    nicheScore: 0.45 * growth[i] + 0.25 * (1 - comp[i]) + 0.20 * rpm[i] + 0.10 * (c.channelIds.length / (list.length || 1))
  }));
  res.json(scored.sort((a, b) => b.nicheScore - a.nicheScore));
});

// ---- Outliers ----
function outlierScore(v) {
  const ch = channels.get(v.channelId) || { medianViews: 1, subs: 0 };
  const median = ch.medianViews || 1;
  const viewsRatio = v.views / (median + 1);
  const cluster = clusters.get(v.clusterId);
  const clusterVph = (cluster ? Array.from(videos.values()).filter(x => x.clusterId === v.clusterId).map(x => x.vph || 0) : [v.vph || 0]);
  const vphPct = clusterVph.filter(x => x <= (v.vph || 0)).length / (clusterVph.length || 1);
  const smallBoost = clamp((100000 - (ch.subs || 0)) / 100000, 0, 1) * 0.3;
  const ageHours = (Date.now() - new Date(v.publishedAt).getTime()) / 3.6e6;
  const timeDecay = Math.exp(-ageHours / 72);
  return Math.pow(viewsRatio, 0.4) * (1 + vphPct) * (1 + smallBoost) * timeDecay * 10;
}
app.get("/v1/outliers", (req, res) => {
  const min = Number(req.query.min_score || 5);
  const scored = Array.from(videos.values())
    .map(v => ({ videoId: v.id, ytId: v.ytId, title: v.title, score: outlierScore(v), channelId: v.channelId }))
    .filter(x => x.score >= min)
    .sort((a, b) => b.score - a.score);
  res.json(scored);
});

// ---- Styles ----
app.post("/v1/styles/extract", (req, res) => {
  const { name, sampleTitles = [] } = req.body;
  const pattern = {
    numericLead: sampleTitles.filter(t => /^\d/.test(t)).length / (sampleTitles.length || 1),
    bracketUse: sampleTitles.filter(t => t.includes("[") || t.includes("(")).length / (sampleTitles.length || 1),
    colonSplit: sampleTitles.filter(t => t.includes(":")).length / (sampleTitles.length || 1)
  };
  const id = uuid();
  styles.set(id, { id, name, patterns: pattern });
  res.json(styles.get(id));
});

// ---- Title Generator ----
app.post("/v1/titles/generate", async (req, res) => {
  const { niche, emotion = "curiosity", hook = "curiosity", styleId } = req.body;
  const style = styleId ? styles.get(styleId) : null;
  const prompt = `
Generate 12 YouTube titles (55-68 chars) for niche: ${niche}.
Target emotion: ${emotion}. Hook type: ${hook}.
${style ? `Patterns: numericLead=${style.patterns.numericLead.toFixed(2)}, bracket=${style.patterns.bracketUse.toFixed(2)}` : ""}
Return as JSON array of strings.`;
  let titles;
  if (openai) {
    const resp = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: prompt }],
      temperature: 0.9,
      max_tokens: 400
    });
    titles = JSON.parse(resp.choices[0].message.content);
  } else {
    titles = Array.from({ length: 12 }, (_, i) => `${niche} secret #${i + 1} that boosts ${emotion}`);
  }
  res.json({ titles });
});

// ---- Script Writer ----
app.post("/v1/scripts/generate", async (req, res) => {
  const { niche, format = "explainer", words = 800, lang = "en" } = req.body;
  const prompt = `
Write a ${format} YouTube script about "${niche}" in ${lang}, ${words} words.
Hook in first 5 seconds, pattern interrupt every 15 seconds, add scene markers with timestamps.
Return JSON: {script: string, scenes:[{t_start, t_end, summary, visual_prompt}]}`;
  let script;
  if (openai) {
    const resp = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: prompt }],
      temperature: 0.7,
      max_tokens: 1800
    });
    script = JSON.parse(resp.choices[0].message.content);
  } else {
    script = { script: `Intro hook about ${niche}...`, scenes: [{ t_start: 0, t_end: 10, summary: "Hook", visual_prompt: "dramatic intro" }] };
  }
  res.json(script);
});

// ---- Voice (Eleven / HuggingFace / Kokoro) ----
async function ttsEleven(text, voiceId = "21m00Tcm4TlvDq8ikWAM") {
  const r = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`, {
    method: "POST",
    headers: {
      "xi-api-key": process.env.ELEVEN_API_KEY,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ text, model_id: "eleven_multilingual_v2" })
  });
  if (!r.ok) throw new Error("ElevenLabs TTS failed");
  const buf = Buffer.from(await r.arrayBuffer());
  return `data:audio/mpeg;base64,${buf.toString("base64")}`;
}
async function ttsHuggingFace(text, model = "espnet/kan-bayashi_ljspeech_vits") {
  const r = await fetch(`https://api-inference.huggingface.co/models/${model}`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.HUGGINGFACE_API_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ inputs: text })
  });
  if (!r.ok) throw new Error("HuggingFace TTS failed");
  const buf = Buffer.from(await r.arrayBuffer());
  return `data:audio/wav;base64,${buf.toString("base64")}`;
}
async function ttsKokoro(text) {
  if (!process.env.KOKORO_URL) throw new Error("Set KOKORO_URL for Kokoro TTS");
  const r = await fetch(process.env.KOKORO_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text })
  });
  if (!r.ok) throw new Error("Kokoro TTS failed");
  const buf = Buffer.from(await r.arrayBuffer());
  return `data:audio/wav;base64,${buf.toString("base64")}`;
}
app.post("/v1/voice/synthesize", async (req, res) => {
  const { text, provider = "eleven", voiceId } = req.body;
  try {
    let audio;
    if (provider === "huggingface") audio = await ttsHuggingFace(text);
    else if (provider === "kokoro") audio = await ttsKokoro(text);
    else audio = await ttsEleven(text, voiceId);
    res.json({ audio });
  } catch (e) {
    res.status(400).json({ error: e.message });
  }
});

// ---- Music (stub) ----
app.post("/v1/music/generate", (req, res) => {
  const { mood = "cinematic" } = req.body;
  res.json({ musicUrl: "s3://placeholder/music.mp3", mood, notes: "Integrate MusicGen/Stable Audio here." });
});

// ---- Scenes (stub) ----
app.post("/v1/scenes/build", (req, res) => {
  const { scenes = [] } = req.body;
  const outputs = scenes.map((s, i) => ({ ...s, imageUrl: `s3://placeholder/scene_${i}.png` }));
  res.json({ scenes: outputs });
});

// ---- Thumbnails (stub) ----
app.post("/v1/thumbnails/generate", (req, res) => {
  const { title } = req.body;
  res.json({
    variants: [
      { url: "s3://placeholder/thumb_a.png", notes: "High-contrast, minimal text" },
      { url: "s3://placeholder/thumb_b.png", notes: "Face close-up, warm/cool split" }
    ],
    title
  });
});

// ---- Gemini bulk image queue ----
async function geminiImage(prompt) {
  if (!gemini) throw new Error("Set GEMINI_API_KEY for image generation");
  const model = gemini.getGenerativeModel({ model: "imagen-3.0-generate" });
  const result = await model.generateContent([{ text: prompt }]);
  const img = result.response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
  if (!img) throw new Error("Gemini returned no image data");
  return `data:image/png;base64,${img}`;
}
app.post("/v1/images/queue", (req, res) => {
  const { prompts = [] } = req.body;
  const ids = prompts.map(p => {
    const id = uuid();
    bulkImageQueue.push({ id, prompt: p, status: "queued" });
    return id;
  });
  res.json({ jobIds: ids });
});
app.get("/v1/images/status", (req, res) => {
  const { id } = req.query;
  const job = bulkImageQueue.find(j => j.id === id);
  if (!job) return res.status(404).json({ error: "not found" });
  res.json(job);
});
setInterval(async () => {
  const job = bulkImageQueue.find(j => j.status === "queued");
  if (!job) return;
  job.status = "running";
  try {
    job.result = await geminiImage(job.prompt);
    job.status = "done";
  } catch (e) {
    job.status = "error";
    job.error = e.message;
  }
}, 2000);

// ---- Ingest endpoints ----
app.post("/v1/ingest/channel", async (req, res) => {
  try {
    const { channelId, days = 7 } = req.body;
    const ch = await fetchChannel(channelId);
    const vids = await fetchRecentVideos(channelId, days);
    for (const v of vids) await addVideo(v);
    clusterVideos();
    res.json({ channel: ch, ingested: vids.length });
  } catch (e) {
    res.status(400).json({ error: e.message });
  }
});
app.post("/v1/ingest/video", async (req, res) => {
  try {
    const video = await addVideo(req.body);
    res.json(video);
  } catch (e) {
    res.status(400).json({ error: e.message });
  }
});
app.post("/v1/cluster/rebuild", (_req, res) => {
  clusterVideos();
  res.json({ clusters: clusters.size });
});

// ---- Transcript endpoint ----
app.get("/v1/transcript/:ytId", async (req, res) => {
  const { ytId } = req.params;
  const lang = req.query.lang || "en";
  const transcript = await fetchTranscript(ytId, lang);
  res.json({ ytId, transcript });
});

// ---- Simple trending/outlier API + minimal UI ----
app.get("/api/simple/outliers", async (req, res) => {
  const region = req.query.region || "US";
  const maxResults = Number(req.query.max || 25);
  try {
    const data = await fetchTrendingOutliers(region, maxResults);
    res.json(data);
  } catch (e) {
    res.status(400).json({ error: e.message });
  }
});

app.get("/api/simple/search", async (req, res) => {
  const q = req.query.q || "";
  const maxResults = Number(req.query.max || 25);
  const region = req.query.region || "US";
  const order = req.query.sort === "recent" ? "date" : "viewCount";
  const days = Number(req.query.days || 1000000);
  if (!q.trim()) return res.status(400).json({ error: "q is required" });
  try {
    const data = await searchVideos(q, maxResults, region, order, days);
    res.json({ query: q, region, videos: data });
  } catch (e) {
    res.status(400).json({ error: e.message });
  }
});

app.get("/api/simple/category", async (req, res) => {
  const region = req.query.region || "US";
  const categoryId = req.query.categoryId || "0";
  const days = Number(req.query.days || 7);
  const maxResults = Number(req.query.max || 50);
  try {
    const data = await fetchCategoryOutliers({ regionCode: region, categoryId, days, maxResults });
    res.json({ region, categoryId, days, videos: data });
  } catch (e) {
    res.status(400).json({ error: e.message });
  }
});

app.get("/api/simple/transcript", async (req, res) => {
  const ytId = req.query.id;
  const lang = req.query.lang || "en";
  if (!ytId) return res.status(400).json({ error: "id is required" });
  try {
    const text = await fetchTranscript(ytId, lang);
    if (!text) {
      res.status(404).json({ error: "Transcript not available" });
      return;
    }
    res.setHeader("Content-Type", "text/plain; charset=utf-8");
    res.send(text);
  } catch (e) {
    res.status(400).json({ error: e.message });
  }
});

app.get("/simple", (_req, res) => {
  res.sendFile(path.join(__dirname, "public", "simple.html"));
});

// ---- Server ----
const port = process.env.PORT || 3000;
// When required (e.g., Vercel serverless), just export the Express app.
if (require.main === module) {
  app.listen(port, () => console.log(`Creator AI API running on :${port}`));
}

module.exports = app;
