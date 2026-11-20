/**
 * Spawns and supervises the Python MuJoCo Studio daemon.
 *
 * The daemon binds a web-viewer port (the live 3D scene) and a loopback control
 * port. Both are published in the session file, which is the single source of
 * truth for "is a simulation running".
 */

import { spawn } from "node:child_process";
import { closeSync, existsSync, mkdirSync, openSync, readFileSync, rmSync } from "node:fs";
import { type RoboticsPaths, type SessionInfo, processAlive, readSession } from "./paths.ts";

export interface StartOptions {
  python: string;
  paths: RoboticsPaths;
  scene?: string;
  model?: string;
  httpPort?: number;
  controlPort?: number;
  compact?: boolean;
}

const START_TIMEOUT_MS = 30000;
const POLL_INTERVAL_MS = 250;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Reads the tail of the daemon log for diagnostics. */
function tailLog(paths: RoboticsPaths, maxBytes = 4000): string {
  try {
    const raw = readFileSync(paths.logFile, "utf8");
    return raw.slice(-maxBytes);
  } catch {
    return "";
  }
}

export async function startDaemon(options: StartOptions): Promise<SessionInfo> {
  const { paths, python } = options;
  mkdirSync(paths.runtimeDir, { recursive: true });

  // Reuse an already-running daemon for this project.
  const existing = readSession(paths.sessionFile);
  if (existing && processAlive(existing.pid)) return existing;

  const args = [
    paths.daemonScript,
    "--runtime-dir",
    paths.runtimeDir,
    "--session-file",
    paths.sessionFile,
  ];
  if (options.model) {
    args.push("--model", options.model);
  } else {
    args.push("--scene", options.scene ?? "box_drop");
  }
  if (options.httpPort) args.push("--http-port", String(options.httpPort));
  if (options.controlPort) args.push("--control-port", String(options.controlPort));
  if (options.compact === false) args.push("--no-ui");

  const logFd = openSync(paths.logFile, "a");
  const child = spawn(python, args, {
    cwd: paths.root,
    detached: true,
    stdio: ["ignore", logFd, logFd],
    env: {
      ...process.env,
      PYTHONPATH: [paths.pythonDir, process.env.PYTHONPATH]
        .filter(Boolean)
        .join(":"),
    },
  });
  child.unref();
  closeSync(logFd);

  const deadline = Date.now() + START_TIMEOUT_MS;
  while (Date.now() < deadline) {
    await sleep(POLL_INTERVAL_MS);
    const session = readSession(paths.sessionFile);
    if (session && processAlive(session.pid)) return session;
    if (child.exitCode !== null) {
      throw new Error(
        `daemon exited with code ${child.exitCode}. Log tail:\n${tailLog(paths)}`,
      );
    }
  }
  throw new Error(
    `daemon did not become ready within ${START_TIMEOUT_MS / 1000}s. Log tail:\n${tailLog(paths)}`,
  );
}

export async function stopDaemon(paths: RoboticsPaths): Promise<SessionInfo | null> {
  const session = readSession(paths.sessionFile);
  if (!session) return null;
  try {
    process.kill(session.pid, "SIGTERM");
  } catch {
    /* already gone */
  }
  // Wait for the daemon to remove its session file (clean shutdown).
  const deadline = Date.now() + 4000;
  while (Date.now() < deadline) {
    if (!existsSync(paths.sessionFile) || !processAlive(session.pid)) {
      try {
        rmSync(paths.sessionFile, { force: true });
      } catch {
        /* ignore */
      }
      return session;
    }
    await sleep(POLL_INTERVAL_MS);
  }
  return session;
}

export function daemonStatus(paths: RoboticsPaths): {
  running: boolean;
  session: SessionInfo | null;
} {
  if (!existsSync(paths.sessionFile)) return { running: false, session: null };
  const session = readSession(paths.sessionFile);
  const running = !!session && processAlive(session.pid);
  return { running, session: running ? session : null };
}

export { tailLog };
