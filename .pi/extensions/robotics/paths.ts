/**
 * Path and environment resolution for the pi-robotics extension.
 *
 * Resolves the repository's Python entry point, the per-project runtime
 * directory, and the Python interpreter that has ``mujoco.studio`` installed.
 */

import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const EXTENSION_DIR = dirname(fileURLToPath(import.meta.url));

export interface RoboticsPaths {
  /** Repository root (contains ``python/robotics_daemon.py``). */
  root: string;
  /** Path to ``python/robotics_daemon.py``. */
  daemonScript: string;
  /** Path to the ``python`` package directory. */
  pythonDir: string;
  /** Per-project state directory: ``<cwd>/.pi/robotics``. */
  stateDir: string;
  /** Runtime directory handed to the daemon. */
  runtimeDir: string;
  /** Session file describing the running daemon. */
  sessionFile: string;
  /** Daemon log file. */
  logFile: string;
  /** Visible Live Python Studio workspace (scaffold + hot-reload plugins). */
  workspaceDir: string;
}

/** Walks up from ``start`` looking for the repository marker file. */
function findRepoRoot(start: string): string | null {
  const marker = join("python", "robotics_daemon.py");
  let dir = start;
  for (let i = 0; i < 8; i += 1) {
    if (existsSync(join(dir, marker))) return dir;
    const parent = dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return null;
}

export function resolvePaths(cwd: string): RoboticsPaths {
  const root = findRepoRoot(EXTENSION_DIR) ?? findRepoRoot(resolve(cwd)) ?? resolve(cwd);
  const pythonDir = join(root, "python");
  const stateDir = join(cwd, ".pi", "robotics");
  return {
    root,
    daemonScript: join(pythonDir, "robotics_daemon.py"),
    pythonDir,
    stateDir,
    runtimeDir: join(stateDir, "runtime"),
    sessionFile: join(stateDir, "session.json"),
    logFile: join(stateDir, "daemon.log"),
    // A visible, committable directory (not .pi/): the agent/user edit these
    // Python files and Studio hot-reloads them while the daemon runs.
    workspaceDir: process.env.PI_ROBOTICS_WORKSPACE_DIR || join(cwd, "studio_workspace"),
  };
}

const CHECK_SNIPPET =
  "import mujoco, mujoco.studio, mujoco_studio_wasm; print(mujoco.__version__)";

/** Candidate interpreters, most specific first. */
export function pythonCandidates(root: string): string[] {
  const out: string[] = [];
  const env = process.env.PI_ROBOTICS_PYTHON;
  if (env) out.push(env);
  out.push(
    join(root, ".venv", "bin", "python"),
    join(root, ".venv", "Scripts", "python.exe"),
    join(homedir(), ".venvs", "pi-robotics", "bin", "python"),
    join(homedir(), ".venvs", "pi-robotics", "Scripts", "python.exe"),
    "python3",
    "python",
  );
  return out;
}

export interface PythonCheck {
  path: string;
  ok: boolean;
  version?: string;
  error?: string;
}

let cachedPython: PythonCheck | null = null;

/**
 * Finds an interpreter that can import ``mujoco.studio``.
 * ``exec`` is ``pi.exec``.
 */
export async function findPython(
  root: string,
  exec: (cmd: string, args: string[], opts?: { timeout?: number }) => Promise<{
    code: number;
    stdout: string;
    stderr: string;
  }>,
  force = false,
): Promise<PythonCheck> {
  if (cachedPython && !force) return cachedPython;
  const errors: string[] = [];
  for (const candidate of pythonCandidates(root)) {
    if (candidate.startsWith("/") && !existsSync(candidate)) continue;
    try {
      const res = await exec(candidate, ["-c", CHECK_SNIPPET], { timeout: 20000 });
      if (res.code === 0) {
        cachedPython = {
          path: candidate,
          ok: true,
          version: res.stdout.trim().split("\n").pop(),
        };
        return cachedPython;
      }
      errors.push(`${candidate}: ${res.stderr.trim().split("\n").pop() ?? "failed"}`);
    } catch (error) {
      errors.push(`${candidate}: ${(error as Error).message}`);
    }
  }
  cachedPython = {
    path: "",
    ok: false,
    error: errors.join("; ") || "no python interpreter found",
  };
  return cachedPython;
}

export interface SessionInfo {
  pid: number;
  viewer_port: number;
  control_port: number;
  viewer_url: string;
  runtime_dir: string;
  scene: string;
  model_path: string;
  started_at: number;
}

export function readSession(sessionFile: string): SessionInfo | null {
  try {
    const raw = readFileSync(sessionFile, "utf8");
    return JSON.parse(raw) as SessionInfo;
  } catch {
    return null;
  }
}

export function processAlive(pid: number): boolean {
  if (!pid) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}
