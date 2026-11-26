// Harness: load the pi-robotics extension and exercise its tool without a model.
import { createJiti } from "jiti";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const execFileAsync = promisify(execFile);
const REPO_ROOT = dirname(dirname(fileURLToPath(import.meta.url)));

const jiti = createJiti(import.meta.url, { interopDefault: true });

const mod = await jiti.import(join(REPO_ROOT, ".pi/extensions/robotics/index.ts"));
const factory = mod.default ?? mod;

const tools = new Map();
const commands = new Map();

const pi = {
  on() {},
  registerTool(def) { tools.set(def.name, def); },
  registerCommand(name, def) { commands.set(name, def); },
  async exec(cmd, args, opts = {}) {
    try {
      const { stdout, stderr } = await execFileAsync(cmd, args, {
        timeout: opts.timeout ?? 30000,
        maxBuffer: 64 * 1024 * 1024,
        cwd: opts.cwd,
        env: process.env,
      });
      return { code: 0, stdout, stderr, killed: false, truncated: false };
    } catch (error) {
      return {
        code: error.code ?? 1,
        stdout: error.stdout ?? "",
        stderr: error.stderr ?? error.message,
        killed: false,
        truncated: false,
      };
    }
  },
};

process.env.PI_ROBOTICS_PYTHON =
  process.env.PI_ROBOTICS_PYTHON ?? join(REPO_ROOT, ".venv", "bin", "python");

await factory(pi);
console.log("registered tools:", [...tools.keys()]);
console.log("registered commands:", [...commands.keys()]);

const tool = tools.get("robotics");
const ctx = {
  cwd: REPO_ROOT,
  hasUI: false,
  ui: { setStatus() {}, notify() {} },
};

async function call(action, extra = {}) {
  const res = await tool.execute("t", { action, ...extra }, undefined, undefined, ctx);
  console.log(`\n=== ${action} ===\n${res.content[0].text}`);
  if (res.isError) console.log("(isError)");
  return res;
}

await call("help");
await call("status");
await call("start", { scene: "arm3" });
await call("state");
await call("ctrl", { name: "joint0", value: 0.8 });
await call("qpos", { values: [0.1, 0.2, 0.3] });
await call("step", { n: 20 });
await call("audit", {});
await call("capture", { width: 320, height: 240 });
// Stretch: load the Menagerie robot and run the scripted pick.
await call("scene", { scene: "stretch" });
await call("cameras");
// Camera -> detect -> control closed loop.
await call("vision_pick", { object: "object" });
await call("reset");
await call("pick", { object: "object" });
await call("stop");
await call("status");
