/**
 * pi-robotics extension: MuJoCo simulation + live 3D panel for the pi agent.
 *
 * Ports the capabilities of google-deepmind/robotics-antigravity-plugin into a
 * pi extension:
 *   - spawns a headless MuJoCo Studio web viewer (the real-time 3D scene),
 *   - exposes a loopback control API as the `robotics` tool (load scenes, drive
 *     actuators, step, reset, read state, capture frames),
 *   - opens the live viewer in the user's browser so a human can watch while the
 *     agent drives the same simulation.
 *
 * The heavy lifting lives in `python/robotics_daemon.py`; this file is the
 * agent-facing surface plus process supervision.
 */

import { resolve } from "node:path";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { controlGet, controlPost } from "./client.ts";
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
  "pick",
  "vision_pick",
  "vision_stop",
  "fetch",
  "cameras",
  "stream",
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
// Menagerie-backed scenes: fetched on first use into <cwd>/.pi/robotics/models/.
const MENAGERIE_SCENES = ["stretch", "stretch3"];

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

const SCENE_PARAMS = Type.Record(Type.String(), Type.Any(), {
  description: "Scene parameters, e.g. { height: 1.5, color: '0.9 0.2 0.2 1' }.",
});

export default function roboticsExtension(pi: ExtensionAPI) {
  /**
   * Opens a URL in the user's default browser.
   *
   * The MuJoCo Studio viewer is a WASM/WebGL app served from a loopback port;
   * a real browser tab is the view the extension controls.
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

  // -- session lifecycle ---------------------------------------------------
  pi.on("session_start", async (_event, ctx) => {
    const paths = resolvePaths(ctx.cwd);
    setStatus(ctx, paths);
    const { running, session } = daemonStatus(paths);
    if (running && session && ctx.hasUI) {
      ctx.ui.setWidget("robotics", [
        `robotics: live sim "${session.scene}"`,
        `viewer  ${session.viewer_url}`,
        "run /robotics open to reopen the 3D panel in your browser",
      ]);
    }
  });

  // -- the `robotics` tool -------------------------------------------------
  pi.registerTool({
    name: "robotics",
    label: "Robotics",
    description:
      "Control a MuJoCo robotics simulation and open its live 3D scene in the browser. " +
      "Use it to start a simulation, load or build scenes, drive actuators, step/reset physics, read state, and capture frames.",
    promptSnippet:
      "Drive a MuJoCo simulation (start/stop, scenes, actuators, step/reset, state) with a live 3D panel in the browser",
    promptGuidelines: [
      "Use the robotics tool when the user wants to simulate a robot, run MuJoCo, test a controller in simulation, or see a live 3D physics scene.",
      "Call robotics with action 'start' to boot the simulation; it opens the live 3D scene in the user's default browser automatically.",
      "robotics 'open_panel' reopens the browser view (e.g. after the tab was closed).",
      "Use robotics action 'scene' to switch built-in scenes (arm3, box_drop, cartpole, falling_objects, table_task, empty; Menagerie robots: stretch, stretch3) and 'load' for an XML/URDF/MJB file.",
      "For Hello Robot Stretch: scene 'stretch' (fetched once from MuJoCo Menagerie into .pi/robotics/models/), then action 'pick' to grasp and lift the tabletop object.",
      "Use robotics action 'stream' to play the robot's camera as a live MJPEG video in the browser (the Stretch head RealSense is 'camera_rgb'); 'cameras' lists them.",
      "robotics 'vision_pick' runs a camera->detect->control closed loop: it localizes the object from the robot camera and servos the arm onto it, then grasps. Prefer it over the fixed 'pick' when the object is placed by the user.",
      "Drive the sim with robotics 'ctrl', 'qpos', 'pause', 'step', 'reset', and read it with 'state'; the numbers come from the running physics engine.",
      "If robotics reports no Python environment, run action 'setup' once to install mujoco>=4.0.1 into the project venv.",
    ],
    parameters: Type.Object({
      action: Type.Union(
        ACTIONS.map((v) => Type.Literal(v)),
        {
          description:
            "What to do. help, setup, status, start, stop, open_panel, scenes, scene, load, state, step, pause, reset, ctrl, speed, capture, pick, vision_pick, vision_stop, cameras, stream.",
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
      object: Type.Optional(
        Type.String({ description: "Body name to pick up (for action 'pick'; default 'object')." }),
      ),
      camera: Type.Optional(
        Type.String({ description: "Camera name for action 'stream' (default 'camera_rgb')." }),
      ),
      fps: Type.Optional(
        Type.Number({ description: "Stream frame rate for action 'stream' (default 15)." }),
      ),
      plane_z: Type.Optional(
        Type.Number({
          description: "Tabletop plane height (m) for 'vision_pick' back-projection (default 0.52).",
        }),
      ),
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
            "Scenes:     scenes · scene {name,params} · load {path} · fetch {scene}",
            "Control:    ctrl {values}|{name,value} · qpos {values} · pause {paused} · speed {value}",
            "Stepping:   step {n} · reset",
            "Reading:    state (time, qpos, qvel, ctrl, sites, pick)",
            "Frames:     capture {width,height}",
            "Manip:      pick {object}   (scripted grasp+lift)\n" +
            "            vision_pick {object,camera}  (camera->detect->control servo)",
            "Cameras:    cameras \u00b7 stream {camera,fps,width,height}  (live MJPEG in the browser)",
            "",
            `Built-in scenes: ${DEFAULT_SCENES.join(", ")}`,
            `Menagerie scenes: ${MENAGERIE_SCENES.join(", ")} (fetched on first use)`,
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
          const opened = await openInBrowser(session.viewer_url);
          const state = await controlGet(session.control_port, "/state");
          return ok(
            [
              `Simulation running.`,
              `scene=${session.scene}`,
              `viewer=${session.viewer_url}` +
                (opened ? " (opened in your browser)" : ""),
              `control=:${session.control_port}`,
            ].join("\n"),
            { action, session, state: state.body, browserOpened: opened },
          );
        } catch (error) {
          return fail(`failed to start: ${(error as Error).message}`, { action });
        }
      }

      if (action === "open_panel") {
        const { running, session } = daemonStatus(paths);
        if (!running || !session) return fail("Start the simulation first.", { action });
        const opened = await openInBrowser(session.viewer_url);
        return ok(
          `viewer: ${session.viewer_url}` + (opened ? " (opened in your browser)" : ""),
          { action, session, browserOpened: opened },
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

      switch (action) {
        case "scenes": {
          const res = await controlGet(port, "/scenes");
          return ok(json(res.body ?? { scenes: DEFAULT_SCENES }), { action });
        }
        case "scene": {
          if (!params.scene) return fail("action 'scene' needs a scene name.", { action });
          // Menagerie scenes are fetched on first use; allow the download.
          const res = await controlPost(
            port,
            "/scene",
            { name: params.scene, params: params.params ?? {} },
            900000,
          );
          setStatus(ctx, paths);
          return ok(json(res.body), { action, session });
        }
        case "load": {
          if (!params.path) return fail("action 'load' needs a path.", { action });
          const absolute = resolve(ctx.cwd, params.path);
          const res = await controlPost(port, "/load", { path: absolute }, 900000);
          setStatus(ctx, paths);
          return ok(json(res.body), { action, session });
        }
        case "fetch": {
          if (!params.scene) return fail("action 'fetch' needs a scene name.", { action });
          const res = await controlPost(port, "/fetch", { name: params.scene }, 900000);
          return ok(json(res.body), { action });
        }
        case "cameras": {
          const res = await controlGet(port, "/cameras");
          return ok(json(res.body ?? { cameras: [] }), { action });
        }
        case "stream": {
          // Live MJPEG of a model camera (e.g. the robot's head RealSense).
          const query = new URLSearchParams({
            width: String(params.width ?? 480),
            height: String(params.height ?? 360),
            fps: String(params.fps ?? 15),
          });
          if (params.camera) query.set("camera", params.camera);
          const url = `http://127.0.0.1:${port}/camera?${query.toString()}`;
          const opened = await openInBrowser(url);
          return ok(
            `Robot camera stream: ${url}` + (opened ? " (opened in your browser)" : ""),
            { action, url, browserOpened: opened },
          );
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
        case "pick": {
          const objectName = params.object ?? "object";
          const start = await controlPost(port, "/command", { cmd: "pick", object: objectName });
          if (!(start.body as { ok?: boolean })?.ok) {
            return fail(`pick failed to start: ${json(start.body)}`, { action });
          }
          // The program runs on the sim clock (~7 s at 100% speed); wait for it.
          const deadline = Date.now() + 30000;
          let pick: Record<string, unknown> = { active: true };
          while (Date.now() < deadline) {
            await new Promise((r) => setTimeout(r, 500));
            const s = await controlGet(port, "/state");
            pick = ((s.body as { pick?: Record<string, unknown> })?.pick ?? {}) as Record<string, unknown>;
            if (pick.done === true || pick.error) break;
          }
          if (pick.done !== true) {
            return fail(`pick did not finish: ${json(pick)}`, { action, pick });
          }
          return ok(json(pick), { action, pick, success: true });
        }
        case "vision_pick": {
          // Camera -> detect -> control: the daemon runs the servo thread.
          const start = await controlPost(port, "/command", {
            cmd: "vision_pick",
            object: params.object ?? "object",
            camera: params.camera ?? "camera_rgb",
            ...(params.plane_z !== undefined ? { plane_z: params.plane_z } : {}),
          });
          if (!(start.body as { ok?: boolean })?.ok) {
            return fail(`vision_pick failed to start: ${json(start.body)}`, { action });
          }
          const deadline = Date.now() + 45000;
          let servo: Record<string, unknown> = { active: true };
          while (Date.now() < deadline) {
            await new Promise((r) => setTimeout(r, 500));
            const s = await controlGet(port, "/state");
            servo = ((s.body as { servo?: Record<string, unknown> })?.servo ?? {}) as Record<string, unknown>;
            if (servo.done === true || servo.error) break;
          }
          if (servo.done !== true) {
            return fail(`vision_pick did not finish: ${json(servo)}`, { action, servo });
          }
          return ok(json(servo), { action, servo, success: true });
        }
        case "vision_stop": {
          const res = await controlPost(port, "/command", { cmd: "vision_stop" });
          return ok(json(res.body), { action });
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
      "Control the MuJoCo simulation (start [scene] | stop | open | status | scenes | scene <name> | pick [object] | vision_pick [object] [camera] | camera [name] | cameras | capture | setup)",
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
          await controlPost(port, "/scene", { name: rest[0], params: {} }, 900000);
          setStatus(ctx, paths);
          ctx.ui.notify(`robotics: switched to scene '${rest[0]}'`, "info");
          return;
        }
        if (sub === "pick") {
          const port = runningControlPort(paths);
          if (port === null) {
            ctx.ui.notify("robotics: not running \u2014 try /robotics start", "warning");
            return;
          }
          ctx.ui.notify("robotics: picking up the object\u2026", "info");
          await controlPost(port, "/command", { cmd: "pick", object: rest[0] ?? "object" });
          const deadline = Date.now() + 30000;
          let pick: Record<string, unknown> = {};
          while (Date.now() < deadline) {
            await new Promise((r) => setTimeout(r, 500));
            const s = await controlGet(port, "/state");
            pick = ((s.body as { pick?: Record<string, unknown> })?.pick ?? {}) as Record<string, unknown>;
            if (pick.done === true || pick.error) break;
          }
          ctx.ui.notify(
            pick.done === true
              ? `robotics: picked up \u2014 lifted ${pick.lifted} m`
              : `robotics pick failed: ${JSON.stringify(pick)}`,
            pick.done === true ? "info" : "error",
          );
          return;
        }
        if (sub === "vision" || sub === "vision_pick") {
          const port = runningControlPort(paths);
          if (port === null) {
            ctx.ui.notify("robotics: not running \u2014 try /robotics start", "warning");
            return;
          }
          ctx.ui.notify("robotics: vision pick (camera \u2192 detect \u2192 control)\u2026", "info");
          await controlPost(port, "/command", {
            cmd: "vision_pick",
            object: rest[0] ?? "object",
            camera: rest[1] ?? "camera_rgb",
          });
          const deadline = Date.now() + 45000;
          let servo: Record<string, unknown> = {};
          while (Date.now() < deadline) {
            await new Promise((r) => setTimeout(r, 500));
            const s = await controlGet(port, "/state");
            servo = ((s.body as { servo?: Record<string, unknown> })?.servo ?? {}) as Record<string, unknown>;
            if (servo.done === true || servo.error) break;
          }
          ctx.ui.notify(
            servo.done === true
              ? `robotics: vision pick lifted ${servo.lifted} m (aligned=${servo.aligned})`
              : `robotics vision pick failed: ${JSON.stringify(servo)}`,
            servo.done === true ? "info" : "error",
          );
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
        if (sub === "camera" || sub === "cameras" || sub === "stream") {
          const port = runningControlPort(paths);
          if (port === null) {
            ctx.ui.notify("robotics: not running \u2014 try /robotics start", "warning");
            return;
          }
          if (sub === "cameras") {
            const res = await controlGet(port, "/cameras");
            const cams = (res.body as { cameras?: string[] })?.cameras ?? [];
            ctx.ui.notify(`robotics cameras: ${cams.join(", ") || "(none)"}`, "info");
            return;
          }
          const q = new URLSearchParams({ width: "480", height: "360", fps: "15" });
          if (sub === "camera" && rest[0]) q.set("camera", rest[0]);
          const url = `http://127.0.0.1:${port}/camera?${q.toString()}`;
          const opened = await openInBrowser(url);
          ctx.ui.notify(
            opened ? `robotics: opened camera stream ${url}` : `robotics camera: ${url}`,
            opened ? "info" : "warning",
          );
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
