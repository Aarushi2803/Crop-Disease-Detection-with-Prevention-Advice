// CropDoc frontend: pick a photo -> send to /api/predict -> show the result.

// ---------- Shortcuts to page elements ----------
const $ = (id) => document.getElementById(id);
const el = {
  capture: $("capture"), stage: $("stage"), scanning: $("scanning"),
  result: $("result"), error: $("error"), errorMsg: $("errorMsg"),
  photo: $("photo"), demoBanner: $("demoBanner"),
  camera: $("cameraInput"), gallery: $("galleryInput"),
  crop: $("crop"), disease: $("disease"), chip: $("chip"),
  ringFg: $("ringFg"), confText: $("confText"),
  unsure: $("unsure"), guessTitle: $("guessTitle"), bars: $("bars"),
  advice: $("advice"), cause: $("cause"), symptoms: $("symptoms"),
  treatList: $("treatList"), preventList: $("preventList"),
  disclaimer: $("disclaimer"),
  historyBox: $("historyBox"), historyList: $("historyList"),
};

const RING_LENGTH = 2 * Math.PI * 52;   // length of the gauge circle (radius 52)
const MAX_SIDE = 1024;                  // shrink big phone photos to this size before upload
const HISTORY_KEY = "cropdoc_history";  // where recent scans are saved in the browser
let photoUrl = null;                    // current preview address (so we can free memory)

// ---------- Screen switching ----------
// Only one "screen" is visible at a time: idle, scanning, result or error.
function setView(view) {
  el.capture.hidden = view !== "idle";
  el.stage.hidden = !(view === "scanning" || view === "result");
  el.scanning.hidden = view !== "scanning";
  el.result.hidden = view !== "result";
  el.error.hidden = view !== "error";
  if (view !== "idle") window.scrollTo({ top: 0, behavior: "smooth" });
}

function showError(message) {
  el.errorMsg.textContent = message;
  setView("error");
}

// ---------- Picking a photo ----------
function onPick(event) {
  const file = event.target.files[0];
  event.target.value = "";            // lets the user pick the same photo again later
  if (file) handleFile(file);
}
el.camera.addEventListener("change", onPick);
el.gallery.addEventListener("change", onPick);

// Phone photos are often 5-12 MB. Shrinking makes the upload fast and stays under the 8 MB limit.
// Drawing on a canvas also fixes sideways photos (it reads the phone's rotation info).
async function shrink(file) {
  try {
    const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
    const scale = Math.min(1, MAX_SIDE / Math.max(bmp.width, bmp.height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(bmp.width * scale);
    canvas.height = Math.round(bmp.height * scale);
    canvas.getContext("2d").drawImage(bmp, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise((done) => canvas.toBlob(done, "image/jpeg", 0.9));
    return blob || file;
  } catch (e) {
    return file;                      // if shrinking fails, just send the original
  }
}

// Show the chosen photo on the specimen card
function showPhoto(blob) {
  if (photoUrl) URL.revokeObjectURL(photoUrl);
  photoUrl = URL.createObjectURL(blob);
  el.photo.src = photoUrl;
}

// ---------- Main flow ----------
async function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    return showError("Please choose a photo (JPG or PNG).");
  }
  setView("scanning");
  const blob = await shrink(file);
  showPhoto(blob);

  const form = new FormData();
  form.append("file", blob, "leaf.jpg");   // the backend expects a field named "file"

  try {
    const res = await fetch("/api/predict", { method: "POST", body: form });
    if (!res.ok) {
      let message = "Something went wrong. Please try again.";
      try { message = (await res.json()).detail || message; } catch (e) { /* keep default */ }
      throw new Error(message);
    }
    render(await res.json());
  } catch (err) {
    // A TypeError here usually means the server cannot be reached
    showError(err instanceof TypeError ? "Cannot reach the server. Check your connection." : err.message);
  }
}

// ---------- Showing the result ----------
function fillList(listEl, items) {
  listEl.replaceChildren();           // clear old items
  (items || []).forEach((text) => {
    const li = document.createElement("li");
    li.textContent = text;            // textContent is safe (no HTML injection)
    listEl.appendChild(li);
  });
}

function render(data) {
  const advice = data.advice || {};
  const unsure = data.low_confidence;
  const pct = Math.round(data.top.confidence * 100);
  const state = unsure ? "unsure" : advice.healthy ? "healthy" : "disease";
  el.result.dataset.state = state;    // CSS uses this to pick colours

  // Title area
  el.crop.textContent = unsure ? "Result" : advice.crop || data.top.label;
  el.disease.textContent = unsure
    ? "Not sure, please retake the photo"
    : advice.healthy ? "Looks healthy" : advice.disease || data.top.label;
  el.chip.textContent = unsure ? "Low confidence"
    : advice.healthy ? "Healthy leaf" : "Disease detected";

  // Confidence gauge (animates from empty to the right amount)
  el.confText.textContent = pct + "%";
  el.ringFg.style.strokeDasharray = RING_LENGTH;
  el.ringFg.style.strokeDashoffset = RING_LENGTH;
  setTimeout(() => {
    el.ringFg.style.strokeDashoffset = RING_LENGTH * (1 - data.top.confidence);
  }, 80);

  // Top-3 bars
  el.guessTitle.textContent = unsure ? "Closest guesses" : "Top guesses";
  el.bars.replaceChildren();
  const fills = [];
  data.predictions.forEach((p) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    const name = document.createElement("span");
    name.textContent = p.label;
    const value = document.createElement("span");
    value.textContent = Math.round(p.confidence * 100) + "%";
    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    track.appendChild(fill);
    row.append(name, value, track);
    el.bars.appendChild(row);
    fills.push([fill, p.confidence]);
  });
  setTimeout(() => fills.forEach(([f, c]) => (f.style.width = Math.max(2, c * 100) + "%")), 80);

  // Retake tips only when not sure; advice only when sure
  // (advice for a shaky guess could mislead a farmer, so we hide it)
  el.unsure.hidden = !unsure;
  el.advice.hidden = unsure;
  if (!unsure) {
    el.cause.textContent = advice.cause || "";
    el.symptoms.textContent = advice.symptoms || "";
    fillList(el.treatList, advice.treatment);
    fillList(el.preventList, advice.prevention);
    openTab("why");
  }

  el.disclaimer.textContent = data.disclaimer || "";
  el.disclaimer.hidden = !data.disclaimer;
  el.demoBanner.hidden = !data.demo_mode;

  if (!data.demo_mode) saveHistory(data.top.label, pct, unsure);   // do not save fake results
  setView("result");
}

// ---------- Tabs ----------
function openTab(name) {
  document.querySelectorAll(".tab").forEach((t) => {
    t.classList.toggle("active", t.dataset.tab === name);
  });
  ["why", "treat", "prevent"].forEach((n) => {
    $("tab-" + n).hidden = n !== name;
  });
}
document.querySelectorAll(".tab").forEach((t) => {
  t.addEventListener("click", () => openTab(t.dataset.tab));
});

// ---------- Recent scans (saved only in this browser) ----------
function loadHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; }
  catch (e) { return []; }            // storage can be blocked, so never crash
}

function saveHistory(label, pct, unsure) {
  const list = loadHistory();
  list.unshift({ label: unsure ? "Unclear photo" : label, pct, t: Date.now() });
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(list.slice(0, 5))); }
  catch (e) { /* ignore */ }
  renderHistory();
}

function renderHistory() {
  const list = loadHistory();
  el.historyBox.hidden = list.length === 0;
  el.historyList.replaceChildren();
  list.forEach((item) => {
    const li = document.createElement("li");
    const left = document.createElement("span");
    left.textContent = `${item.label} (${item.pct}%)`;
    const right = document.createElement("span");
    right.textContent = new Date(item.t).toLocaleDateString();
    li.append(left, right);
    el.historyList.appendChild(li);
  });
}

$("clearHistory").addEventListener("click", () => {
  try { localStorage.removeItem(HISTORY_KEY); } catch (e) { /* ignore */ }
  renderHistory();
});

// ---------- Buttons that go back to the start ----------
$("againBtn").addEventListener("click", () => setView("idle"));
$("errorBtn").addEventListener("click", () => setView("idle"));

// ---------- Start ----------
renderHistory();
setView("idle");