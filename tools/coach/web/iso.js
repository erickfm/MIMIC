// Reads the files the replay viewer needs from the player's own Melee ISO, in the browser, and
// keeps them in this browser (IndexedDB) so the ISO is asked for once. Nothing is uploaded.

const NEEDED = /^(Pl|Gr|Ef|It|If|Lb)[^/]*\.(dat|usd)$/i;
const DB = "coach-game-files", STORE = "files", VERSION = 1;

function u32(view, off) { return view.getUint32(off, false); }

async function readBytes(file, off, len) {
  return new DataView(await file.slice(off, off + len).arrayBuffer());
}

/** Check the disc and list the files to keep: [{name, blob}] (blobs are slices of the ISO). */
export async function readIso(file) {
  const head = await readBytes(file, 0, 0x440);
  const id = String.fromCharCode(...new Uint8Array(head.buffer, 0, 6));
  const rev = head.getUint8(7);
  if (id !== "GALE01") throw new Error(id.startsWith("GAL")
    ? "This is a Melee disc, but not the NTSC (US) version. The coach needs NTSC 1.02."
    : "This file isn't a Melee disc image (ISO).");
  if (rev !== 2) throw new Error(`This is NTSC Melee version 1.0${rev}. The coach needs version 1.02.`);
  const dolOff = u32(head, 0x420), fstOff = u32(head, 0x424), fstSize = u32(head, 0x428);
  const fst = await readBytes(file, fstOff, fstSize);
  const n = u32(fst, 8), strBase = 12 * n;
  const bytes = new Uint8Array(fst.buffer);
  const nameAt = (o) => { let e = strBase + o; while (bytes[e]) e++; return new TextDecoder().decode(bytes.subarray(strBase + o, e)); };
  const out = [];
  for (let i = 1; i < n; i++) {
    const w = u32(fst, 12 * i);
    if (w >>> 24) continue;                      // a directory
    const name = nameAt(w & 0xffffff);
    if (!NEEDED.test(name)) continue;
    const off = u32(fst, 12 * i + 4), len = u32(fst, 12 * i + 8);
    out.push({ name, blob: file.slice(off, off + len) });
  }
  // main.dol: its size is the end of its furthest section
  const dh = await readBytes(file, dolOff, 0xe0);
  let end = 0x100;
  for (let s = 0; s < 18; s++) end = Math.max(end, u32(dh, 4 * s) + u32(dh, 0x90 + 4 * s));
  out.push({ name: "main.dol", blob: file.slice(dolOff, dolOff + end) });
  return out;
}

function openDb() {
  return new Promise((res, rej) => {
    const r = indexedDB.open(DB, VERSION);
    r.onupgradeneeded = () => r.result.createObjectStore(STORE);
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
}

/** Copy the listed files into this browser. onProgress(fraction). */
export async function keepFiles(list, onProgress) {
  const db = await openDb();
  const total = list.reduce((s, f) => s + f.blob.size, 0);
  let done = 0;
  for (let i = 0; i < list.length; i += 20) {
    const chunk = list.slice(i, i + 20);
    // read first (outside the transaction), then write in one go
    const bufs = await Promise.all(chunk.map((f) => f.blob.arrayBuffer()));
    await new Promise((res, rej) => {
      const tx = db.transaction(STORE, "readwrite");
      chunk.forEach((f, j) => tx.objectStore(STORE).put(new Blob([bufs[j]]), f.name));
      tx.oncomplete = res; tx.onerror = () => rej(tx.error); tx.onabort = () => rej(tx.error || new Error("storage full"));
    });
    done += chunk.reduce((s, f) => s + f.blob.size, 0);
    onProgress?.(done / total);
  }
  await new Promise((res, rej) => {
    const tx = db.transaction(STORE, "readwrite");
    tx.objectStore(STORE).put(new Blob([JSON.stringify({ n: list.length, at: Date.now() })]), "__ok__");
    tx.oncomplete = res; tx.onerror = () => rej(tx.error);
  });
  try { await navigator.storage?.persist?.(); } catch {}
}

/** The kept files as File objects, or null if the ISO hasn't been given yet. */
export async function keptFiles() {
  const db = await openDb();
  return new Promise((res, rej) => {
    const tx = db.transaction(STORE, "readonly");
    const st = tx.objectStore(STORE);
    const out = [];
    let ok = false;
    const cur = st.openCursor();
    cur.onsuccess = () => {
      const c = cur.result;
      if (!c) return res(ok ? out : null);
      if (c.key === "__ok__") ok = true;
      else out.push(new File([c.value], c.key));
      c.continue();
    };
    cur.onerror = () => rej(cur.error);
  });
}

export async function forgetFiles() {
  const db = await openDb();
  await new Promise((res) => { const tx = db.transaction(STORE, "readwrite"); tx.objectStore(STORE).clear(); tx.oncomplete = res; });
}
