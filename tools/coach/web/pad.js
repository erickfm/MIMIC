// The player's controller for "Try it yourself": a GameCube controller adapter (read directly over
// WebUSB, the official adapter's protocol, the same one Dolphin uses), any gamepad the browser
// sees (Gamepad API, standard layout), or the keyboard. read() gives the server's pad format:
// {stick: [x, y], c: [x, y] (-1..1, 1 = the game's full tilt), l, r (0..1), buttons: [names]}.

export const KEYS = [
  ["W A S D", "move (hold Shift to tilt)"], ["J", "A"], ["K", "B"], ["I or Space", "jump"],
  ["U", "Z"], ["L", "shield / air dodge"], ["arrow keys", "C-stick"],
];

const GC_VENDOR = 0x057e, GC_PRODUCT = 0x0337;

export class PadReader {
  constructor() {
    this.keys = new Set();
    this.adapter = null;      // {device, port, state} once a GameCube adapter is connected
    this.onKey = (e) => {
      if (e.target.closest?.("input,select,textarea")) return;
      const k = e.code;
      if (["KeyW", "KeyA", "KeyS", "KeyD", "KeyJ", "KeyK", "KeyI", "KeyU", "KeyL", "Space", "ShiftLeft", "ShiftRight",
           "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Enter"].includes(k)) {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "keydown") this.keys.add(k); else this.keys.delete(k);
      }
    };
  }

  start() {
    window.addEventListener("keydown", this.onKey, true);
    window.addEventListener("keyup", this.onKey, true);
    this.blur = () => this.keys.clear();
    window.addEventListener("blur", this.blur);
  }

  stop() {
    window.removeEventListener("keydown", this.onKey, true);
    window.removeEventListener("keyup", this.onKey, true);
    window.removeEventListener("blur", this.blur);
    this.keys.clear();
  }

  /** What the player is using: "GameCube adapter (port 1)", a gamepad's name, or "keyboard". */
  source() {
    if (this.adapter?.state) return `GameCube adapter (port ${this.adapter.port + 1})`;
    const gp = this.gamepad();
    return gp ? gp.id.replace(/\s*\(.*$/, "") : "keyboard";
  }

  gamepad() {
    const pads = navigator.getGamepads ? [...navigator.getGamepads()].filter(Boolean) : [];
    return pads.find((p) => p.connected) || null;
  }

  read() {
    if (this.adapter?.state) return this.adapter.state;
    const gp = this.gamepad();
    if (gp) return fromGamepad(gp);
    return this.fromKeys();
  }

  fromKeys() {
    const k = this.keys, has = (c) => k.has(c);
    const tilt = has("ShiftLeft") || has("ShiftRight") ? 0.5 : 1;
    const x = (has("KeyD") ? 1 : 0) - (has("KeyA") ? 1 : 0), y = (has("KeyW") ? 1 : 0) - (has("KeyS") ? 1 : 0);
    const n = x && y ? Math.SQRT1_2 : 1;
    const cx = (has("ArrowRight") ? 1 : 0) - (has("ArrowLeft") ? 1 : 0), cy = (has("ArrowUp") ? 1 : 0) - (has("ArrowDown") ? 1 : 0);
    const buttons = [];
    if (has("KeyJ")) buttons.push("a");
    if (has("KeyK")) buttons.push("b");
    if (has("KeyI")) buttons.push("x");
    if (has("Space")) buttons.push("y");
    if (has("KeyU")) buttons.push("z");
    if (has("KeyL")) buttons.push("r");
    if (has("Enter")) buttons.push("start");
    return { stick: [x * n * tilt, y * n * tilt], c: [cx, cy], l: 0, r: has("KeyL") ? 1 : 0, buttons };
  }

  /** Ask for the GameCube adapter (the browser shows its device picker). */
  async connectAdapter() {
    if (!navigator.usb) throw new Error("This browser can't read USB devices directly. Chrome or Edge can.");
    const device = await navigator.usb.requestDevice({ filters: [{ vendorId: GC_VENDOR, productId: GC_PRODUCT }] });
    await device.open();
    if (device.configuration === null) await device.selectConfiguration(1);
    await device.claimInterface(0);
    await device.transferOut(2, new Uint8Array([0x13]));   // start reporting
    this.adapter = { device, port: 0, state: null };
    this.pollAdapter();
    return device;
  }

  async pollAdapter() {
    const a = this.adapter;
    while (this.adapter === a) {
      let r;
      try { r = await a.device.transferIn(1, 37); } catch { this.adapter = null; return; }
      const d = r.data;
      if (!d || d.byteLength < 37 || d.getUint8(0) !== 0x21) continue;
      // the first port with a controller in it
      let port = -1;
      for (let p = 0; p < 4; p++) if (d.getUint8(1 + 9 * p) >> 4) { port = p; break; }
      if (port < 0) { a.state = null; continue; }
      a.port = port;
      const o = 1 + 9 * port, b1 = d.getUint8(o + 1), b2 = d.getUint8(o + 2);
      const buttons = [];
      const bit = (v, m, n) => { if (v & m) buttons.push(n); };
      bit(b1, 0x01, "a"); bit(b1, 0x02, "b"); bit(b1, 0x04, "x"); bit(b1, 0x08, "y");
      bit(b1, 0x10, "dleft"); bit(b1, 0x20, "dright"); bit(b1, 0x40, "ddown"); bit(b1, 0x80, "dup");
      bit(b2, 0x01, "start"); bit(b2, 0x02, "z"); bit(b2, 0x04, "r"); bit(b2, 0x08, "l");
      // raw stick bytes (128 = center) and analog triggers, kept exact (the server scales by 80 / 140)
      a.state = {
        stick: [(d.getUint8(o + 3) - 128) / 80, (d.getUint8(o + 4) - 128) / 80],
        c: [(d.getUint8(o + 5) - 128) / 80, (d.getUint8(o + 6) - 128) / 80],
        l: d.getUint8(o + 7) / 140, r: d.getUint8(o + 8) / 140, buttons,
      };
    }
  }
}

/** A gamepad in the standard layout (Xbox-style): left stick = control stick, right stick =
 *  C-stick, A = A, X = B, B and Y = jump, RB = Z, triggers = L and R (analog; fully in = click). */
function fromGamepad(gp) {
  const ax = (i) => gp.axes[i] ?? 0, btn = (i) => gp.buttons[i]?.pressed, val = (i) => gp.buttons[i]?.value ?? 0;
  const dz = (v) => (Math.abs(v) < 0.08 ? 0 : v);
  const buttons = [];
  if (btn(0)) buttons.push("a");
  if (btn(2)) buttons.push("b");
  if (btn(1)) buttons.push("x");
  if (btn(3)) buttons.push("y");
  if (btn(5)) buttons.push("z");
  if (val(6) > 0.95 || btn(4)) buttons.push("l");
  if (val(7) > 0.95) buttons.push("r");
  if (btn(9)) buttons.push("start");
  if (btn(12)) buttons.push("dup");
  if (btn(13)) buttons.push("ddown");
  if (btn(14)) buttons.push("dleft");
  if (btn(15)) buttons.push("dright");
  return { stick: [dz(ax(0)), -dz(ax(1))], c: [dz(ax(2)), -dz(ax(3))], l: Math.max(val(6), btn(4) ? 1 : 0), r: val(7), buttons };
}

export function samePad(a, b) {
  return !!a && !!b && a.stick[0] === b.stick[0] && a.stick[1] === b.stick[1] && a.c[0] === b.c[0] && a.c[1] === b.c[1] &&
    a.l === b.l && a.r === b.r && a.buttons.join() === b.buttons.join();
}
