// Minimal GitHub REST helpers used by the upload page.
// Pure functions (no DOM) so they can be tested from Node too.

const API = "https://api.github.com";

export class GitHubError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

export async function api(path, { token, method = "GET", body, accept, raw } = {}) {
  const headers = {
    Accept: accept || "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
  };
  if (token) headers.Authorization = `Bearer ${token}`;
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const res = await fetch(API + path, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
    cache: "no-store",
  });
  if (!res.ok) {
    let msg = res.statusText;
    try {
      msg = (await res.json()).message || msg;
    } catch (_) {}
    throw new GitHubError(res.status, `${res.status}: ${msg}`);
  }
  if (raw) return res;
  if (res.status === 204) return null;
  return res.json();
}

/** Can this token push to the repo? Returns {ok, login, reason}. */
export async function checkToken(repo, token) {
  try {
    const r = await api(`/repos/${repo}`, { token });
    if (r.permissions && r.permissions.push) return { ok: true };
    return { ok: false, reason: "This token can read the repo but cannot upload to it. Give it Contents: Read and write." };
  } catch (e) {
    if (e.status === 401) return { ok: false, reason: "GitHub did not accept this token (expired or mistyped)." };
    if (e.status === 404) return { ok: false, reason: `This token cannot see ${repo}. Make sure the repo is selected when creating the token.` };
    return { ok: false, reason: e.message };
  }
}

/**
 * Put several files into ONE commit on `branch` (so the workflow runs once).
 * files: [{ path, base64 }]  ->  returns the new commit sha.
 */
export async function commitFiles({ repo, token, branch, files, message }) {
  for (let attempt = 0; attempt < 3; attempt++) {
    const ref = await api(`/repos/${repo}/git/ref/heads/${encodeURIComponent(branch)}`, { token });
    const parentSha = ref.object.sha;
    const parent = await api(`/repos/${repo}/git/commits/${parentSha}`, { token });
    const tree = [];
    for (const f of files) {
      const blob = await api(`/repos/${repo}/git/blobs`, {
        token, method: "POST", body: { content: f.base64, encoding: "base64" },
      });
      tree.push({ path: f.path, mode: "100644", type: "blob", sha: blob.sha });
    }
    const newTree = await api(`/repos/${repo}/git/trees`, {
      token, method: "POST", body: { base_tree: parent.tree.sha, tree },
    });
    const commit = await api(`/repos/${repo}/git/commits`, {
      token, method: "POST", body: { message, tree: newTree.sha, parents: [parentSha] },
    });
    try {
      await api(`/repos/${repo}/git/refs/heads/${encodeURIComponent(branch)}`, {
        token, method: "PATCH", body: { sha: commit.sha, force: false },
      });
      return commit.sha;
    } catch (e) {
      // someone (often the bot saving a PDF) pushed in between: rebuild on the new tip
      if (e.status === 422 && attempt < 2) continue;
      throw e;
    }
  }
}

/** The workflow run started by a given commit, or null if it has not appeared yet. */
export async function findRun(repo, sha, token) {
  const r = await api(`/repos/${repo}/actions/runs?head_sha=${sha}&per_page=5`, { token });
  const runs = (r.workflow_runs || []).filter((x) => x.name === "Make papercraft");
  return runs[0] || null;
}

export async function getRun(repo, id, token) {
  return api(`/repos/${repo}/actions/runs/${id}`, { token });
}

/** Last commit that touched `path` on `branch` (or null). */
export async function lastCommitFor(repo, path, branch, token) {
  const r = await api(
    `/repos/${repo}/commits?path=${encodeURIComponent(path)}&sha=${encodeURIComponent(branch)}&per_page=1`,
    { token });
  return r[0] || null;
}

/** Download a file from the repo as a Blob. */
export async function getFileBlob(repo, path, branch, token) {
  const res = await api(
    `/repos/${repo}/contents/${path.split("/").map(encodeURIComponent).join("/")}?ref=${encodeURIComponent(branch)}`,
    { token, accept: "application/vnd.github.raw", raw: true });
  return res.blob();
}

/** List entries of a folder (empty list if it does not exist). */
export async function listFolder(repo, folder, branch, token) {
  try {
    const r = await api(`/repos/${repo}/contents/${folder}?ref=${encodeURIComponent(branch)}`, { token });
    return Array.isArray(r) ? r : [];
  } catch (e) {
    if (e.status === 404) return [];
    throw e;
  }
}

/** "My Red Mug!.JPG" -> "my-red-mug" */
export function slugify(name) {
  return (name || "")
    .replace(/\.[^.]+$/, "")
    .normalize("NFKD").replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 40) || "model";
}

/** Base64 of an ArrayBuffer/Uint8Array (works in browsers and Node). */
export function toBase64(buf) {
  const bytes = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  }
  return btoa(s);
}

export function utf8Base64(text) {
  return toBase64(new TextEncoder().encode(text));
}
