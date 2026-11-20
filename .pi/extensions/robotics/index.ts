/**
 * pi-robotics extension: MuJoCo simulation + live 3D panel for the pi agent.
 *
 * Ports the capabilities of google-deepmind/robotics-antigravity-plugin into a
 * pi extension:
 *   - spawns a headless MuJoCo Studio web viewer (the real-time 3D scene),
 *   - exposes a loopback control API as the `robotics` tool (load scenes, drive
 *     actuators, step, reset, read state, capture frames),
 *   - points the pi-dashboard canvas at the live viewer so a human can watch and
 *     the agent can drive the same simulation.
 *
 * The heavy lifting lives in `python/robotics_daemon.py`; this file is the
 * agent-facing surface plus process supervision.
 */

import { resolve } from "node:path";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { controlGet, controlPost } from "./client.ts";
import { startPanelProxy, type PanelProxy } from "./panel-shim.ts";
import {
  type StartOptions,
  daemonStatus,
  startDaemon,
  stopDaemon,
  tailLog,
} from "./daemon.ts";
import {
  type RoboticsPaths,
  type SessionInfo,
  findPython,
  resolvePaths,
} from "./paths.ts";

const STATUS_KEY = "robotics";
const ACTIONS = [
  "help",
  "setup",
  "status",
  "start",
  "stop",
  "open_panel",
  "scenes",
  "scene",
  "load",
  "state",
  "step",
  "pause",
  "reset",
  "ctrl",
  "speed",
  "capture",
  "panel",
] as const;
const DEFAULT_SCENES = [
  "arm3",
  "box_drop",
  "cartpole",
  "empty",
  "falling_objects",
  "table_task",
];

interface ToolResult {
  content: Array<{ type: "text"; text: string }>;
  details: Record<string, unknown>;
  isError?: boolean;
}

function ok(text: string, details: Record<string, unknown> = {}): ToolResult {
  return { content: [{ type: "text", text }], details };
}

function fail(text: string, details: Record<string, unknown> = {}): ToolResult {
  return { content: [{ type: "text", text }], details, isError: true };
}

function json(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function setStatus(ctx: ExtensionContext, paths: RoboticsPaths): void {
  if (!ctx.hasUI) return;
  const { running, session } = daemonStatus(paths);
  if (running && session) {
    ctx.ui.setStatus(STATUS_KEY, `robotics: ${session.scene} :${session.viewer_port}`);
  } else {
    ctx.ui.setStatus(STATUS_KEY, "robotics: stopped");
  }
}

/** One shim proxy per viewer port, shared across sessions in this pi process. */
const panelProxies = new Map<number, Promise<PanelProxy>>();

/**
 * Returns a loopback port that serves the viewer with a shim making its
 * root-absolute requests survive pi-dashboard's ``/live/<id>/`` path proxy.
 * Falls back to the raw viewer port if the proxy cannot start.
 */
async function ensurePanelProxy(viewerPort: number): Promise<number> {
  let promise = panelProxies.get(viewerPort);
  if (!promise) {
    promise = startPanelProxy(viewerPort);
    panelProxies.set(viewerPort, promise);
  }
  try {
    return (await promise).port;
  } catch {
    panelProxies.delete(viewerPort);
    return viewerPort;
  }
}

/** Declares the live 3D viewer on the pi-dashboard canvas. */
async function openPanel(
  ctx: ExtensionContext,
  session: SessionInfo,
): Promise<{ ok: boolean; via?: string; port?: number; note: string }> {
  const title = `MuJoCo Studio \u2014 ${session.scene}`;
  const port = await ensurePanelProxy(session.viewer_port);
  const call = async (args: unknown) => {
    // `executeTool` returns isError for unknown tools instead of throwing.
    const res = (await (ctx as any).executeTool?.("canvas", args)) as
      | { isError?: boolean }
      | undefined;
    return res && !res.isError;
  };
  try {
    if (await call({ target: { kind: "server", port, path: "/" }, title, mode: "pin" })) {
      return {
        ok: true,
        via: "canvas:server",
        port,
        note: "declared on the pi-dashboard canvas (note: the sandboxed preview cannot run the WASM viewer; use the browser tab)",
      };
    }
    if (await call({ target: { kind: "url", url: `http://127.0.0.1:${port}/` }, title, mode: "pin" })) {
      return { ok: true, via: "canvas:url", port, note: "declared on the pi-dashboard canvas" };
    }
  } catch {
    /* fall through to URL-only */
  }
  return {
    ok: false,
    port,
    note: `canvas chip not available; use the browser viewer`,
  };
}

const SCENE_PARAMS = Type.Record(Type.String(), Type.Any(), {
  description: "Scene parameters, e.g. { height: 1.5, color: '0.9 0.2 0.2 1' }.",
});

export default function roboticsExtension(pi: ExtensionAPI) {
  /**
   * Opens a URL in the user's default browser.
   *
   * The pi-dashboard canvas renders a declared server inside a sandboxed,
   * opaque-origin iframe, where Chromium's Private Network Access blocks the
   * `fetch()` calls the MuJoCo WASM client needs (WASM binary + model bundles).
   * A real browser tab is therefore the reliable way to view the live scene.
   */
  const openInBrowser = async (url: string): Promise<boolean> => {
    try {
      if (process.platform === "darwin") {
        return (await pi.exec("open", [url], {} as never)).code === 0;
      }
      if (process.platform === "win32") {
        return (await pi.exec("cmd", ["/c", "start", "", url], {} as never)).code === 0;
      }
      return (await pi.exec("xdg-open", [url], {} as never)).code === 0;
    } catch {
      return false;
    }
  };

  const startOptions = async (
    ctx: ExtensionContext,
    paths: RoboticsPaths,
    overrides: Partial<StartOptions>,
  ): Promise<StartOptions> => {
    const py = await findPython(paths.root, (cmd, args, opts) =>
      pi.exec(cmd, args, opts as never),
    );
    if (!py.ok) {
      throw new Error(
        "No Python with mujoco.studio found. Run `robotics setup` first, or set PI_ROBOTICS_PYTHON.\n" +
          `Details: ${py.error}`,
      );
    }
    return { python: py.path, paths, ...overrides };
  };

  const ensureDaemon = async (
    ctx: ExtensionContext,
    paths: RoboticsPaths,
  ): Promise<SessionInfo> => {
    const status = daemonStatus(paths);
    if (status.running && status.session) return status.session;
    const options = await startOptions(ctx, paths, {});
    const session = await startDaemon(options);
    setStatus(ctx, paths);
    return session;
  };

  // Declare the live panel at most once per viewer port in this session, so any
  // first interaction with the sim puts the 3D scene on the dashboard canvas.
  let declaredViewerPort: number | null = null;
  const declarePanel = async (
    ctx: ExtensionContext,
    session: SessionInfo,
  ): Promise<{ ok: boolean; via?: string; port?: number; note: string } | null> => {
    if (declaredViewerPort === session.viewer_port) return null;
    const panel = await openPanel(ctx, session);
    if (panel.ok) declaredViewerPort = session.viewer_port;
    return panel;
  };

  // -- session lifecycle ---------------------------------------------------
  pi.on("session_start", async (_event, ctx) => {
    const paths = resolvePaths(ctx.cwd);
    setStatus(ctx, paths);
    const { running, session } = daemonStatus(paths);
    if (running && session) {
      // Warm the shim proxy so the panel opens instantly when asked for.
      await ensurePanelProxy(session.viewer_port);
      if (ctx.hasUI) {
        ctx.ui.setWidget("robotics", [
          `robotics: live sim "${session.scene}"`,
          `viewer  http://127.0.0.1:${session.viewer_port}/`,
          "ask: \"open the MuJoCo 3D panel\" to show it on the canvas",
        ]);
      }
    }
  });

  // -- the `robotics` tool -------------------------------------------------
  pi.registerTool({
    name: "robotics",
    label: "Robotics",
    description:
      "Control a MuJoCo robotics simulation and show its live 3D scene on the pi-dashboard canvas. " +
      "Use it to start a simulation, load or build scenes, drive actuators, step/reset physics, read state, and capture frames.",
    promptSnippet:
      "Drive a MuJoCo simulation (start/stop, scenes, actuators, step/reset, state) with a live 3D panel",
    promptGuidelines: [
      "Use the robotics tool when the user wants to simulate a robot, run MuJoCo, test a controller in simulation, or see a live 3D physics scene.",
      "Call robotics with action 'start' to boot the simulation; it opens the live 3D scene in the user's browser automatically.",
      "The MuJoCo WASM viewer cannot render in the pi-dashboard sandboxed preview (Chromium blocks its fetch of the WASM binary and /model from an opaque-origin iframe); view it in a browser tab at the viewer URL. robotics 'open_panel' opens that tab and also declares a canvas chip.",
      "Use robotics action 'scene' to switch built-in scenes (arm3, box_drop, cartpole, falling_objects, table_task, empty) and 'load' for an XML/URDF/MJB file.",
      "Drive the sim with robotics 'ctrl', 'qpos', 'pause', 'step', 'reset', and read it with 'state'; the numbers come from the running physics engine.",
      "If robotics reports no Python environment, run action 'setup' once to install mujoco>=4.0.1 into the project venv.",
    ],
    parameters: Type.Object({
      action: Type.Union(
        ACTIONS.map((v) => Type.Literal(v)),
        {
          description:
            "What to do. help, setup, status, start, stop, open_panel, scenes, scene, load, state, step, pause, reset, ctrl, speed, capture.",
        },
      ),
      scene: Type.Optional(
        Type.String({ description: "Built-in scene name (for action 'scene'/'start')." }),
      ),
      params: Type.Optional(SCENE_PARAMS),
      path: Type.Optional(
        Type.String({ description: "Model file path relative to the project (for action 'load')." }),
      ),
      values: Type.Optional(
        Type.Array(Type.Number(), { description: "Actuator controls or qpos vector." }),
      ),
      name: Type.Optional(Type.String({ description: "Actuator name for a named 'ctrl'." })),
      value: Type.Optional(Type.Number({ description: "Value for a named 'ctrl' or 'speed'." })),
      n: Type.Optional(Type.Number({ description: "Number of steps (for action 'step')." })),
      paused: Type.Optional(Type.Boolean({ description: "Pause state (for action 'pause')." })),
      width: Type.Optional(Type.Number()),
      height: Type.Optional(Type.Number()),
    }),

    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      const paths = resolvePaths(ctx.cwd);
      const action = (params.action === "panel" ? "open_panel" : params.action) as string;

      if (action === "help") {
        return ok(
          [
            "pi-robotics — MuJoCo simulation with a live 3D panel.",
            "",
            "Lifecycle:  start · stop · status · open_panel (alias: panel, opens browser) · setup",
            "Scenes:     scenes · scene {name,params} · load {path}",
            "Control:    ctrl {values}|{name,value} · qpos {values} · pause {paused} · speed {value}",
            "Stepping:   step {n} · reset",
            "Reading:    state (time, qpos, qvel, ctrl, sites, contacts)",
            "Frames:     capture {width,height}",
            "",
            `Built-in scenes: ${DEFAULT_SCENES.join(", ")}`,
          ].join("\n"),
          { action },
        );
      }

      if (action === "setup") {
        const venv = resolve(paths.root, ".venv");
        const uv = await pi.exec(
          "bash",
          [
            "-lc",
            `set -e
if command -v uv >/dev/null 2>&1; then
  uv venv "${venv}" >/dev/null 2>&1 || true
  uv pip install --python "${venv}/bin/python" "mujoco>=4.0.1" imageio pillow
else
  python3 -m venv "${venv}"
  "${venv}/bin/pip" install --upgrade pip >/dev/null
  "${venv}/bin/pip" install "mujoco>=4.0.1" imageio pillow
fi`,
          ],
          { timeout: 900000 } as never,
        );
        if (uv.code !== 0) {
          return fail(`setup failed (exit ${uv.code}):\n${uv.stderr || uv.stdout}`, { action });
        }
        const check = await findPython(paths.root, (cmd, args, opts) =>
          pi.exec(cmd, args, opts as never),
          true,
        );
        return ok(
          `Installed mujoco into ${venv}.\nInterpreter: ${check.path} (${check.version ?? "?"})`,
          { action, python: check.path },
        );
      }

      if (action === "status") {
        const { running, session } = daemonStatus(paths);
        if (!running || !session) {
          setStatus(ctx, paths);
          return ok("Simulation is not running.", { action, running: false });
        }
        const res = await controlGet(session.control_port, "/status");
        setStatus(ctx, paths);
        return ok(json({ session, status: res.body }), { action, session });
      }

      if (action === "stop") {
        const session = await stopDaemon(paths);
        setStatus(ctx, paths);
        return ok(
          session ? `Stopped simulation (pid ${session.pid}).` : "No simulation was running.",
          { action, stopped: !!session },
        );
      }

      if (action === "start") {
        try {
          const options = await startOptions(ctx, paths, {
            scene: params.scene ?? undefined,
          });
          const session = await startDaemon(options);
          setStatus(ctx, paths);
          const panel = await openPanel(ctx, session);
          if (panel.ok) declaredViewerPort = session.viewer_port;
          const opened = await openInBrowser(session.viewer_url);
          const state = await controlGet(session.control_port, "/state");
          return ok(
            [
              `Simulation running.`,
              `scene=${session.scene}`,
              `viewer=${session.viewer_url}` +
                (opened ? " (opened in your browser)" : ""),
              `control=:${session.control_port}`,
              `panel: ${panel.note}`,
            ].join("\n"),
            { action, session, state: state.body, panel, browserOpened: opened },
          );
        } catch (error) {
          return fail(`failed to start: ${(error as Error).message}`, { action });
        }
      }

      if (action === "open_panel") {
        const { running, session } = daemonStatus(paths);
        if (!running || !session) return fail("Start the simulation first.", { action });
        const panel = await openPanel(ctx, session);
        if (panel.ok) declaredViewerPort = session.viewer_port;
        const opened = await openInBrowser(session.viewer_url);
        return ok(
          [
            panel.note,
            `viewer: ${session.viewer_url}` + (opened ? " (opened in your browser)" : ""),
          ].join("\n"),
          { action, session, panel, browserOpened: opened },
        );
      }

      // Everything below needs a running daemon.
      let session: SessionInfo;
      try {
        session = await ensureDaemon(ctx, paths);
      } catch (error) {
        return fail(
          `${(error as Error).message}\nLog tail:\n${tailLog(paths)}`,
          { action },
        );
      }
      const port = session.control_port;

      // Any first sim interaction this session also puts the live scene on the
      // dashboard canvas (no-op once declared).
      await declarePanel(ctx, session);

      switch (action) {
        case "scenes": {
          const res = await controlGet(port, "/scenes");
          return ok(json(res.body ?? { scenes: DEFAULT_SCENES }), { action });
        }
        case "scene": {
          if (!params.scene) return fail("action 'scene' needs a scene name.", { action });
          const res = await controlPost(port, "/scene", {
            name: params.scene,
            params: params.params ?? {},
          });
          setStatus(ctx, paths);
          return ok(json(res.body), { action, session });
        }
        case "load": {
          if (!params.path) return fail("action 'load' needs a path.", { action });
          const absolute = resolve(ctx.cwd, params.path);
          const res = await controlPost(port, "/load", { path: absolute });
          setStatus(ctx, paths);
          return ok(json(res.body), { action, session });
        }
        case "state": {
          const res = await controlGet(port, "/state");
          return ok(json(res.body), { action, session });
        }
        case "step": {
          const res = await controlPost(port, "/command", {
            cmd: "step",
            n: params.n ?? 1,
          });
          return ok(json(res.body), { action });
        }
        case "pause": {
          const res = await controlPost(port, "/command", {
            cmd: "pause",
            paused: params.paused ?? true,
          });
          return ok(json(res.body), { action });
        }
        case "speed": {
          const res = await controlPost(port, "/command", {
            cmd: "speed",
            value: params.value ?? 100,
          });
          return ok(json(res.body), { action });
        }
        case "reset": {
          const res = await controlPost(port, "/command", { cmd: "reset" });
          return ok(json(res.body), { action });
        }
        case "ctrl": {
          if (params.name !== undefined) {
            const res = await controlPost(port, "/command", {
              cmd: "ctrl_named",
              name: params.name,
              value: params.value ?? 0,
            });
            return ok(json(res.body), { action });
          }
          if (!params.values) return fail("action 'ctrl' needs values[] or name+value.", { action });
          const res = await controlPost(port, "/command", {
            cmd: "ctrl",
            values: params.values,
          });
          return ok(json(res.body), { action });
        }
        case "capture": {
          const res = await controlPost(port, "/capture", {
            width: params.width ?? 640,
            height: params.height ?? 480,
          });
          const body = res.body as { ok?: boolean; path?: string; error?: string };
          if (!body?.ok) return fail(`capture failed: ${body?.error ?? "unknown"}`, { action });
          return ok(`Saved frame to ${body.path}`, { action, path: body.path });
        }
        default:
          return fail(`Unknown action '${action}'. Try action 'help'.`, { action });
      }
    },
  });

  // -- `/robotics` slash command ------------------------------------------
  const runningControlPort = (paths: RoboticsPaths): number | null => {
    const { running, session } = daemonStatus(paths);
    return running && session ? session.control_port : null;
  };
  const runningViewerUrl = (paths: RoboticsPaths): string | null => {
    const { running, session } = daemonStatus(paths);
    return running && session ? session.viewer_url : null;
  };

  pi.registerCommand("robotics", {
    description:
      "Control the MuJoCo simulation (start [scene] | stop | open | status | scenes | scene <name> | capture | setup)",
    handler: async (args, ctx) => {
      const paths = resolvePaths(ctx.cwd);
      const [sub, ...rest] = (args ?? "").trim().split(/\s+/).filter(Boolean);
      try {
        if (sub === "start") {
          const py = await findPython(paths.root, (cmd, a, opts) =>
            pi.exec(cmd, a, opts as never),
          );
          if (!py.ok) {
            ctx.ui.notify("No python with mujoco.studio. Run /robotics setup.", "error");
            return;
          }
          const session = await startDaemon({ python: py.path, paths, scene: rest[0] });
          setStatus(ctx, paths);
          const opened = await openInBrowser(session.viewer_url);
          ctx.ui.notify(
            `robotics: ${session.scene} running \u2014 ${session.viewer_url}` +
              (opened ? " (opened in your browser)" : ""),
            "info",
          );
          return;
        }
        if (sub === "open" || sub === "panel") {
          const url = runningViewerUrl(paths);
          if (url === null) {
            ctx.ui.notify("robotics: not running \u2014 try /robotics start", "warning");
            return;
          }
          const opened = await openInBrowser(url);
          ctx.ui.notify(
            opened ? `robotics: opened ${url}` : `robotics viewer: ${url}`,
            opened ? "info" : "warning",
          );
          return;
        }
        if (sub === "stop") {
          await stopDaemon(paths);
          setStatus(ctx, paths);
          ctx.ui.notify("robotics: stopped", "info");
          return;
        }
        if (sub === "scene" || sub === "scenes") {
          const port = runningControlPort(paths);
          if (port === null) {
            ctx.ui.notify(`robotics: not running. Built-in scenes: ${DEFAULT_SCENES.join(", ")}`, "info");
            return;
          }
          if (sub === "scenes" || !rest[0]) {
            const res = await controlGet(port, "/scenes");
            const scenes = (res.body as { scenes?: string[] })?.scenes ?? DEFAULT_SCENES;
            ctx.ui.notify(`robotics scenes: ${scenes.join(", ")}`, "info");
            return;
          }
          await controlPost(port, "/scene", { name: rest[0], params: {} });
          setStatus(ctx, paths);
          ctx.ui.notify(`robotics: switched to scene '${rest[0]}'`, "info");
          return;
        }
        if (sub === "capture") {
          const port = runningControlPort(paths);
          if (port === null) {
            ctx.ui.notify("robotics: not running \u2014 try /robotics start", "warning");
            return;
          }
          const res = await controlPost(port, "/capture", { width: 640, height: 480 });
          const body = res.body as { ok?: boolean; path?: string; error?: string };
          ctx.ui.notify(body?.ok ? `robotics: saved ${body.path}` : `robotics capture failed: ${body?.error}`, body?.ok ? "info" : "error");
          return;
        }
        if (sub === "setup") {
          ctx.ui.notify("Run the robotics tool action 'setup' to install mujoco.", "info");
          return;
        }
        const { running, session } = daemonStatus(paths);
        ctx.ui.notify(
          running && session
            ? `robotics: ${session.scene} at ${session.viewer_url} \u2014 /robotics open to show it`
            : "robotics: stopped \u2014 /robotics start",
          "info",
        );
      } catch (error) {
        ctx.ui.notify(`robotics error: ${(error as Error).message}`, "error");
      }
    },
  });
}
