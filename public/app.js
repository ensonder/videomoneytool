async function post(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
async function get(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

document.getElementById("ingestBtn").onclick = async () => {
  const channelId = document.getElementById("channelId").value.trim();
  const days = Number(document.getElementById("days").value || 7);
  const out = document.getElementById("ingestResult");
  out.textContent = "Running...";
  try {
    const data = await post("/api/ingest/channel", { channelId, days });
    out.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    out.textContent = e.message;
  }
};

document.getElementById("outliersBtn").onclick = async () => {
  const out = document.getElementById("outliersResult");
  out.textContent = "Loading...";
  try {
    const data = await get("/api/outliers");
    out.textContent = JSON.stringify(data.slice(0, 10), null, 2);
  } catch (e) {
    out.textContent = e.message;
  }
};

document.getElementById("nichesBtn").onclick = async () => {
  const out = document.getElementById("nichesResult");
  out.textContent = "Loading...";
  try {
    const data = await get("/api/niches");
    out.textContent = JSON.stringify(data.slice(0, 10), null, 2);
  } catch (e) {
    out.textContent = e.message;
  }
};

document.getElementById("titlesBtn").onclick = async () => {
  const niche = document.getElementById("titleNiche").value || "AI side hustles";
  const out = document.getElementById("titlesResult");
  out.textContent = "Generating...";
  try {
    const data = await post("/api/titles/generate", { niche });
    out.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    out.textContent = e.message;
  }
};
