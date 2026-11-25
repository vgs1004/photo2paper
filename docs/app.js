import {
  checkToken, commitFiles, findRun, getRun, lastCommitFor, getFileBlob,
  listFolder, slugify, toBase64, utf8Base64,
} from "./github.js";

// ---------------------------------------------------------------- config
const params = new URLSearchParams(location.search);
function detectRepo() {
  if (params.get("repo")) return params.get("repo");
  const m = location.hostname.match(/^([^.]+)\.github\.io$/);
  const first = location.pathname.split("/").filter(Boolean)[0];
  if (m && first) return `${m[1]}/${first}`;
  return "vgs1004/photo2paper";
}
const REPO = detectRepo();
const BRANCH = params.get("branch") || "main";
const KEY_TOKEN = "p2p.token";
const KEY_JOB = "p2p.job";

const $ = (id) => document.getElementById(id);
const store = {
  get(k) { try { return localStorage.getItem(k); } catch (_) { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (_) {} },
  del(k) { try { localStorage.removeItem(k); } catch (_) {} },
};

let token = "";
let photo = null;          // { base64, ext, url }
let busy = false;
const choice = { size: "150", paper: "A4", faces: "150" };

// ---------------------------------------------------------------- setup
for (const id of ["repo-link", "repo-foot"]) $(id).href = `https://github.com/${REPO}`;
$("repo-link").textContent = REPO;
$("repo-name-hint").textContent = REPO.split("/")[1];

function setConnected(ok, msg) {
  $("conn-state").textContent = ok ? "Connected" : "Not connected";
  $("conn-state").classList.toggle("ok", ok);
  $("connect-body").hidden = ok;
  $("conn-err").hidden = !msg;
  $("conn-err").textContent = msg || "";
  if (ok) {
    $("conn-state").innerHTML = 'Connected · <a href="#" id="disconnect">disconnect</a>';
    $("disconnect").onclick = (e) => {
      e.preventDefault();
      token = ""; store.del(KEY_TOKEN); $("token").value = "";
      setConnected(false);
    };
  }
  refreshGo();
}

async function connect(t, silent) {
  t = (t || "").trim();
  if (!t) return;
  $("connect").disabled = true;
  const r = await checkToken(REPO, t);
  $("connect").disabled = false;
  if (r.ok) {
    token = t;
    if ($("remember").checked) store.set(KEY_TOKEN, t);
    setConnected(true);
  } else {
    if (silent) store.del(KEY_TOKEN);
    setConnected(false, r.reason);
  }
}
$("connect").onclick = () => connect($("token").value);
$("token").addEventListener("keydown", (e) => { if (e.key === "Enter") connect($("token").value); });

// segmented controls
function seg(id, key, onChange) {
  const el = $(id);
  el.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    e.preventDefault();
    el.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    choice[key] = b.dataset.v;
    onChange && onChange();
  });
}
seg("size-seg", "size", () => { $("custom-row").hidden = choice.size !== "custom"; hints(); });
seg("paper-seg", "paper", hints);
seg("detail-seg", "faces", hints);
$("custom-size").addEventListener("input", hints);

function sizeMm() {
  if (choice.size === "custom") {
    const cm = parseFloat($("custom-size").value);
    return Math.round(Math.min(100, Math.max(5, isFinite(cm) ? cm : 25)) * 10);
  }
  return parseInt(choice.size, 10);
}

function hints() {
  const mm = sizeMm();
  const big = mm >= 300 && choice.paper !== "A3";
  $("detail-hint").textContent = {
    "80": "Fewer, bigger pieces: quickest to build, chunkier look.",
    "150": "A good balance for most objects.",
    "250": "Smoother shape, but more small pieces to cut and glue.",
  }[choice.faces];
  $("size-hint").textContent = big
    ? `At ${mm / 10} cm the model needs many ${choice.paper} sheets. A3 paper keeps the pieces larger and the sheet count lower.`
    : "";
}

// ---------------------------------------------------------------- photo
const MAX_SIDE = 1280;

async function loadPhoto(file) {
  if (!file || !/^image\/(jpeg|png|webp)$/.test(file.type)) {
    showErr("go-err", "Please choose a JPG, PNG or WEBP image.");
    return;
  }
  showErr("go-err", "");
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.src = url;
  await img.decode();
  // shrink big phone photos: faster upload, the AI works at ~512 px anyway
  const scale = Math.min(1, MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(img.naturalWidth * scale);
  canvas.height = Math.round(img.naturalHeight * scale);
  const ctx = canvas.getContext("2d");
  const keepAlpha = file.type === "image/png";
  if (!keepAlpha) { ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, canvas.width, canvas.height); }
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  const type = keepAlpha ? "image/png" : "image/jpeg";
  const blob = await new Promise((r) => canvas.toBlob(r, type, 0.9));
  photo = { base64: toBase64(await blob.arrayBuffer()), ext: keepAlpha ? "png" : "jpg" };
  $("preview").src = url;
  $("preview").hidden = false;
  $("drop").classList.add("has-img");
  if (!$("name").value) $("name").value = slugify(file.name);
  refreshGo();
}

$("file").addEventListener("change", (e) => loadPhoto(e.target.files[0]));
const drop = $("drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => loadPhoto(e.dataTransfer.files[0]));
$("name").addEventListener("input", refreshGo);

function refreshGo() {
  $("go").disabled = busy || !token || !photo || !$("name").value.trim();
  $("go").textContent = !token ? "Connect to GitHub first" : !photo ? "Choose a photo first" : "Make my papercraft";
}

function showErr(id, msg) {
  $(id).hidden = !msg;
  $(id).textContent = msg || "";
}

// ---------------------------------------------------------------- build
let timer = null;
function startClock(since) {
  clearInterval(timer);
  const tick = () => {
    const s = Math.max(0, Math.round((Date.now() - since) / 1000));
    $("elapsed").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  };
  tick();
  timer = setInterval(tick, 1000);
}

function step(k, state) {
  const order = ["upload", "queue", "build", "done"];
  const idx = order.indexOf(k);
  document.querySelectorAll("#steps li").forEach((li, i) => {
    li.className = i < idx ? "done" : i === idx ? state || "active" : "";
  });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function uniqueSlug(base) {
  const taken = new Set((await listFolder(REPO, "photos", BRANCH, token)).map((f) => f.name.replace(/\.[^.]+$/, "")));
  if (!taken.has(base)) return base;
  for (let i = 2; ; i++) if (!taken.has(`${base}-${i}`)) return `${base}-${i}`;
}

$("go").onclick = async () => {
  if (busy) return;
  busy = true; refreshGo();
  showErr("go-err", ""); showErr("run-err", "");
  $("result").hidden = true;
  $("progress-card").hidden = false;
  $("progress-card").scrollIntoView({ behavior: "smooth", block: "start" });
  const started = Date.now();
  startClock(started);
  step("upload");
  try {
    const slug = await uniqueSlug(slugify($("name").value));
    $("job-name").textContent = `“${slug}”`;
    const settings = { size: sizeMm(), faces: parseInt(choice.faces, 10), paper: choice.paper };
    const sha = await commitFiles({
      repo: REPO, token, branch: BRANCH,
      message: `Add photo ${slug} (${settings.size / 10} cm, ${settings.paper})`,
      files: [
        { path: `photos/${slug}.json`, base64: utf8Base64(JSON.stringify(settings, null, 2) + "\n") },
        { path: `photos/${slug}.${photo.ext}`, base64: photo.base64 },
      ],
    });
    const job = { slug, sha, started };
    store.set(KEY_JOB, JSON.stringify(job));
    await follow(job);
  } catch (e) {
    failWith(e);
  } finally {
    busy = false; refreshGo();
  }
};

async function follow(job) {
  $("progress-card").hidden = false;
  $("job-name").textContent = `“${job.slug}”`;
  startClock(job.started);
  step("queue");
  let run = null;
  for (let i = 0; i < 60 && !run; i++) {        // up to ~5 min to appear
    run = await findRun(REPO, job.sha, token).catch(() => null);
    if (!run) await sleep(5000);
  }
  if (!run) throw new Error("GitHub did not start the build. Check the Actions tab of the repo.");
  $("open-gh").href = run.html_url;
  while (run.status !== "completed") {
    step(run.status === "in_progress" ? "build" : "queue");
    await sleep(6000);
    run = await getRun(REPO, run.id, token);
  }
  if (run.conclusion !== "success") {
    store.del(KEY_JOB);
    throw Object.assign(new Error("The build failed. Most often the free daily AI limit has been used up; try again tomorrow or add an HF_TOKEN secret to the repo."), { link: run.html_url });
  }
  // the bot saves the PDF in a follow-up commit
  const path = `papercraft/${job.slug}.pdf`;
  let commit = null;
  for (let i = 0; i < 10 && !commit; i++) {
    commit = await lastCommitFor(REPO, path, BRANCH, token).catch(() => null);
    if (!commit) await sleep(3000);
  }
  if (!commit) throw Object.assign(new Error("The build finished but no PDF was saved."), { link: run.html_url });
  const blob = await getFileBlob(REPO, path, BRANCH, token);
  const url = URL.createObjectURL(new Blob([blob], { type: "application/pdf" }));
  $("download").href = url;
  $("download").download = `${job.slug}-papercraft.pdf`;
  $("open-gh").href = `https://github.com/${REPO}/blob/${BRANCH}/${path}`;
  $("pdf-view").src = url;
  $("result").hidden = false;
  step("done", "done");
  document.querySelector('#steps li[data-k="done"]').className = "done";
  clearInterval(timer);
  store.del(KEY_JOB);
  loadGallery();
}

function fail(msg) {
  clearInterval(timer);
  const active = document.querySelector("#steps li.active");
  if (active) active.className = "failed";
  const err = $("run-err");
  err.hidden = false;
  err.textContent = msg + " ";
}
function failWith(e) {
  fail((e && e.message) || String(e));
  if (e && e.link) {
    $("run-err").append(Object.assign(document.createElement("a"), { href: e.link, target: "_blank", rel: "noopener", textContent: "See the log" }));
  }
}

// ---------------------------------------------------------------- gallery
async function loadGallery() {
  const ul = $("gallery");
  try {
    const files = (await listFolder(REPO, "papercraft", BRANCH, token)).filter((f) => f.name.endsWith(".pdf"));
    ul.innerHTML = "";
    if (!files.length) { ul.innerHTML = '<li class="muted">Nothing yet.</li>'; return; }
    for (const f of files) {
      const li = document.createElement("li");
      const name = document.createElement("span");
      name.className = "name";
      name.textContent = f.name.replace(/\.pdf$/, "");
      const links = document.createElement("span");
      links.className = "row";
      const view = Object.assign(document.createElement("a"), { href: f.html_url, target: "_blank", rel: "noopener", textContent: "View" });
      const dl = Object.assign(document.createElement("a"), { href: f.download_url, target: "_blank", rel: "noopener", textContent: "Download", className: "btn ghost small" });
      links.append(view, dl);
      li.append(name, links);
      ul.append(li);
    }
  } catch (e) {
    ul.innerHTML = "";
    const li = document.createElement("li");
    li.className = "muted";
    li.textContent = `Could not load the list (${e.message}).`;
    ul.append(li);
  }
}
$("refresh").onclick = loadGallery;

// ---------------------------------------------------------------- start
hints();
refreshGo();
loadGallery();
const saved = store.get(KEY_TOKEN);
if (saved) {
  $("token").value = saved;
  connect(saved, true).then(() => {
    // resume a build that was running when the page was closed
    let job = null;
    try { job = JSON.parse(store.get(KEY_JOB) || "null"); } catch (_) {}
    if (token && job && Date.now() - job.started < 40 * 60 * 1000) {
      busy = true; refreshGo();
      follow(job).catch(failWith).finally(() => { busy = false; refreshGo(); });
    }
  });
}
