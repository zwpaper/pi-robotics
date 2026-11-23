// Exercises the LLM-in-the-loop vision primitives: `look` returns a camera
// image (base64) the model can see, `nudge` moves a bit and returns a fresh
// frame. Writes the frames to .pi/robotics/runtime/captures/ for inspection.
import { createJiti } from "jiti";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { writeFileSync } from "node:fs";

const execFileAsync = promisify(execFile);
const REPO_ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const jiti = createJiti(import.meta.url, { interopDefault: true });
const mod = await jiti.import(join(REPO_ROOT, ".pi/extensions/robotics/index.ts"));
const factory = mod.default ?? mod;

const tools = new Map();
const pi = {
  on() {},
  registerTool(d) { tools.set(d.name, d); },
  registerCommand() {},
  async exec(cmd, args, opts = {}) {
    try {
      const { stdout, stderr } = await execFileAsync(cmd, args, { timeout: opts.timeout ?? 30000, maxBuffer: 64 * 1024 * 1024, cwd: opts.cwd, env: process.env });
      return { code: 0, stdout, stderr };
    } catch (e) { return { code: e.code ?? 1, stdout: e.stdout ?? "", stderr: e.stderr ?? e.message }; }
  },
};
process.env.PI_ROBOTICS_PYTHON = process.env.PI_ROBOTICS_PYTHON ?? join(REPO_ROOT, ".venv", "bin", "python");
await factory(pi);
const tool = tools.get("robotics");
const ctx = { cwd: REPO_ROOT, hasUI: false, ui: { setStatus() {}, notify() {} } };

const outDir = join(REPO_ROOT, ".pi", "robotics", "runtime", "captures");
let n = 0;
async function call(action, extra = {}) {
  const res = await tool.execute("t", { action, ...extra }, undefined, undefined, ctx);
  const kinds = res.content.map((c) => c.type).join("+");
  const text = res.content.find((c) => c.type === "text")?.text ?? "";
  const img = res.content.find((c) => c.type === "image");
  let file = null;
  if (img) { file = join(outDir, `llm_${String(n++).padStart(2, "0")}_${action}.png`); writeFileSync(file, Buffer.from(img.data, "base64")); }
  console.log(`[${action}] content=${kinds}${file ? " -> " + file.replace(REPO_ROOT + "/", "") : ""}${res.isError ? " (isError)" : ""}`);
  console.log("    " + text.split("\n").join("\n    ").slice(0, 600));
  return res;
}

await call("scene", { scene: "stretch" });
await call("reset");
await call("reel_clear");
await call("look", { cameras: ["camera_rgb", "camera_depth"] });
// A few nudge steps as an LLM would do them (move, then look at the result).
await call("nudge", { extend: 0.15, settle: 0.8 });
await call("nudge", { extend: 0.13, lift: -0.07, wrist: 0.15, settle: 0.9 });
await call("nudge", { grip: -0.005, settle: 1.0 });
await call("nudge", { lift: 0.3, settle: 1.5 });
// Compose the recorded 3D panel + cameras into an operation video.
await call("reel", { fps: 4, height: 300, reel_name: "llm_loop" });
await call("clear");
console.log("DONE");
