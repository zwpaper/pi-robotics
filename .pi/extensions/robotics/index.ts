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
  "qpos",
  "qvel",
  "speed",
  "capture",
  "look",
  "nudge",
  "move_ee",
  "clear",
  "reel",
  "reel_clear",
  "record_start",
  "record_stop",
  "chapter",
  "contains",
  "audit",
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
const MENAGERIE_SCENES = [
  "stretch",
  "stretch3",
  "panda",
  "franka_fr3",
  "ur5e",
  "aloha",
  "so101",
  "go2",
  "g1",
];

interface ToolResult {
  content: Array<
    { type: "text"; text: string } | { type: "image"; data: string; mimeType: string }
  >;
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

/** A tool result that carries a camera frame the model can see. */
function withFrame(
  text: string,
  pngBase64: string,
  details: Record<string, unknown>,
): ToolResult {
  return {
    content: [
      { type: "text", text },
      { type: "image", data: pngBase64, mimeType: "image/png" },
    ],
    details,
  };
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
      "Use robotics action 'scene' to switch built-in scenes (arm3, box_drop, cartpole, falling_objects, table_task, empty; Menagerie robots: stretch, stretch3, panda, franka_fr3, ur5e, aloha, so101, go2, g1) and 'load' for an XML/URDF/MJB file.",
      "For Hello Robot Stretch: scene 'stretch' (fetched once from MuJoCo Menagerie into .pi/robotics/models/), then action 'pick' to grasp and lift the tabletop object.",
      "Use robotics action 'stream' to play the robot's camera as a live MJPEG video in the browser (the Stretch head RealSense is 'camera_rgb'); 'cameras' lists them.",
      "robotics 'vision_pick' runs a camera->detect->control closed loop: it localizes the object from the robot camera and servos the arm onto it, then grasps. Prefer it over the fixed 'pick' when the object is placed by the user.",
      "To grasp by looking at the camera yourself (LLM in the loop): call robotics 'look' to get a robot-camera image, then robotics 'nudge' {extend,lift,wrist,grip,settle} to move a little and get a fresh image, and repeat until the gripper is over the object; then nudge {grip:-0.005} to close and nudge {lift:0.3} to raise. 'clear' releases the manual hold.",
      "Drive the sim with robotics 'ctrl', 'qpos', 'pause', 'step', 'reset', and read it with 'state'; the numbers come from the running physics engine.",
      "Run robotics 'audit' after authoring or loading a scene to catch penetration, instability, mass/inertia problems, joint-limit hits, actuator saturation, and energy blow-up; fix the structured findings and re-audit.",
      "Use robotics 'move_ee' for IK end-effector moves; it clamps joint limits and stops before a new collision (reported as blocked:true with the {geom1,geom2} pairs), unless allow_collision is set.",
      "If robotics reports no Python environment, run action 'setup' once to install mujoco>=4.0.1 into the project venv.",
    ],
    parameters: Type.Object({
      action: Type.Union(
        ACTIONS.map((v) => Type.Literal(v)),
        {
          // Generated from ACTIONS so the schema and the handler cannot drift.
          description: `What to do. ${ACTIONS.join(", ")}.`,
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
        Type.Array(Type.Number(), { description: "Actuator controls, qpos, or qvel vector." }),
      ),
      name: Type.Optional(Type.String({ description: "Actuator name for a named 'ctrl'." })),
      object: Type.Optional(
        Type.String({ description: "Body name to pick up / to test (for 'pick'/'contains')." }),
      ),
      container: Type.Optional(
        Type.String({ description: "Container body for action 'contains' (is 'object' inside its AABB?)." }),
      ),
      camera: Type.Optional(
        Type.String({ description: "Camera name for action 'stream' (default 'camera_rgb')." }),
      ),
      fps: Type.Optional(
        Type.Number({
          description: "Video frame rate for 'stream' (default 15) and the output frame rate for 'reel' (default 10).",
        }),
      ),
      plane_z: Type.Optional(
        Type.Number({
          description: "Tabletop plane height (m) for 'vision_pick' back-projection (default 0.52).",
        }),
      ),
      extend: Type.Optional(
        Type.Number({ description: "arm_extend delta (m) for action 'nudge'." }),
      ),
      lift: Type.Optional(
        Type.Number({ description: "lift delta (m) for action 'nudge'." }),
      ),
      wrist: Type.Optional(
        Type.Number({ description: "wrist_yaw delta (rad) for action 'nudge'." }),
      ),
      grip: Type.Optional(
        Type.Number({ description: "absolute gripper target (0.04 open, -0.005 closed) for action 'nudge'." }),
      ),
      settle: Type.Optional(
        Type.Number({ description: "seconds to let the arm settle after a 'nudge' (default 0.8)." }),
      ),
      cameras: Type.Optional(
        Type.Array(Type.String(), { description: "camera names to capture on 'look' (default: [camera])." }),
      ),
      panel: Type.Optional(
        Type.Boolean({
          description:
            "include the 3D-panel overview image on 'look'/'nudge'/'move_ee' (default false: the model sees only the robot CAMERA image, like real hardware).",
        }),
      ),
      reel_name: Type.Optional(
        Type.String({ description: "output name (no extension) for action 'reel'." }),
      ),
      title: Type.Optional(
        Type.String({ description: "chapter title for action 'chapter' (marks a new attempt)." }),
      ),
      pos: Type.Optional(
        Type.Array(Type.Number(), {
          description: "target world position [x,y,z] for action 'move_ee'.",
        }),
      ),
      quat: Type.Optional(
        Type.Array(Type.Number(), {
          description: "target orientation [w,x,y,z] for action 'move_ee' (optional; else position only).",
        }),
      ),
      site: Type.Optional(
        Type.String({
          description:
            "end-effector site for 'move_ee' (omit to auto-detect a site, then a gripper body).",
        }),
      ),
      body: Type.Optional(
        Type.String({ description: "end-effector body for 'move_ee' (alternative to `site`)." }),
      ),
      posture: Type.Optional(
        Type.Record(Type.String(), Type.Number(), {
          description:
            "{joint_name: angle} null-space posture bias for 'move_ee' (keeps redundant joints near a preferred pose).",
        }),
      ),
      allow_collision: Type.Optional(
        Type.Boolean({
          description:
            "allow 'move_ee' to proceed through contacts (default false: stop before a new collision).",
        }),
      ),
      seconds: Type.Optional(
        Type.Number({
          description: "settle time (s) to simulate for the 'audit' action (default 2.0).",
        }),
      ),
      gripper: Type.Optional(
        Type.String({
          description: "gripper actuator name for 'vision_pick' (default: auto-detect).",
        }),
      ),
      steps: Type.Optional(
        Type.Number({ description: "waypoints for 'move_ee' straight-line path (default 6)." }),
      ),
      actuators: Type.Optional(
        Type.Record(Type.String(), Type.Number(), {
          description: "{actuator_name: delta} for action 'nudge' (any robot).",
        }),
      ),
      hold: Type.Optional(
        Type.Number({
          description: "seconds each recorded frame is shown on 'reel' (default: play at the recorded real pace).",
        }),
      ),
      cols: Type.Optional(
        Type.Number({ description: "grid columns for 'reel' (default: auto, square-ish)." }),
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
            "Control:    ctrl {values}|{name,value} · qpos {values} · qvel {values} · pause {paused} · speed {value}",
            "Stepping:   step {n} · reset",
            "Reading:    state (time, qpos, qvel, ctrl, sites, pick)",
            "Frames:     capture {width,height} · look {camera}  (returns an image you can see)",
            "Vision:     vision_pick {object,camera,site,gripper} (auto servo) · nudge {extend,lift,wrist,grip,actuators} · move_ee {pos,quat,site|body,posture} · clear",
            "Audit:      audit {scene|path,seconds}  (initial penetration, settle, mass/inertia, limits, saturation, energy)",
            "Reel:       record_start {fps,cameras} → chapter {title} per attempt → reel {height} (record_stop/reel_clear)",
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
        case "qpos": {
          if (!params.values) return fail("action 'qpos' needs values[].", { action });
          const res = await controlPost(port, "/command", {
            cmd: "qpos",
            values: params.values,
          });
          return ok(json(res.body), { action });
        }
        case "qvel": {
          if (!params.values) return fail("action 'qvel' needs values[].", { action });
          const res = await controlPost(port, "/command", {
            cmd: "qvel",
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
            ...(params.site !== undefined ? { site: params.site } : {}),
            ...(params.gripper !== undefined ? { gripper: params.gripper } : {}),
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
        case "clear": {
          const res = await controlPost(port, "/command", { cmd: "clear" });
          return ok(json(res.body), { action });
        }
        case "look": {
          // Returns the robot camera frame as an image the model can see; the
          // 3D-panel overview frame is recorded for the operation reel.
          const cameras = params.cameras ?? [params.camera ?? "camera_rgb"];
          const res = await controlPost(port, "/command", {
            cmd: "look",
            camera: cameras[0],
            ...(params.cameras ? { cameras: params.cameras } : {}),
            panel: params.panel ?? false,
            width: params.width ?? 640,
            height: params.height ?? 480,
          });
          const body = res.body as {
            ok?: boolean;
            png_base64?: string;
            panel_base64?: string;
            camera?: string;
            cameras?: string[];
            model_cameras?: string[];
            recorded?: number;
            error?: string;
          };
          if (!body?.ok || !body.png_base64) {
            return fail(`look failed: ${body?.error ?? "no frame"}`, { action });
          }
          const captured = body.cameras ?? cameras;
          const text =
            `Looked via camera '${body.camera ?? cameras[0]}' ` +
            `(captured: ${captured.join(", ")}; model cameras: ${(body.model_cameras ?? []).join(", ") || "none"}); ` +
            `operation reel recorded ${body.recorded ?? 0} step(s) [panel + ${captured.length} camera(s)].`;
          const details = { action, cameras: captured, recorded: body.recorded };
          if (params.panel && body.panel_base64) {
            return {
              content: [
                { type: "text", text },
                { type: "image", data: body.png_base64, mimeType: "image/png" },
                { type: "image", data: body.panel_base64, mimeType: "image/png" },
              ],
              details,
            };
          }
          return withFrame(text, body.png_base64, details);
        }
        case "nudge": {
          // Move the arm a little and return a fresh camera frame.
          const res = await controlPost(port, "/command", {
            cmd: "nudge",
            extend: params.extend,
            lift: params.lift,
            wrist: params.wrist,
            grip: params.grip,
            actuators: params.actuators,
            settle: params.settle,
            camera: params.camera ?? "camera_rgb",
            panel: params.panel ?? false,
            width: params.width ?? 640,
            height: params.height ?? 480,
          });
          const body = res.body as {
            ok?: boolean;
            png_base64?: string;
            panel_base64?: string;
            ctrl?: Record<string, number>;
            recorded?: number;
            error?: string;
          };
          if (!body?.ok || !body.png_base64) {
            return fail(`nudge failed: ${body?.error ?? "no frame"}`, { action });
          }
          const deltas = [
            params.extend !== undefined ? `extend${params.extend >= 0 ? "+" : ""}${params.extend}` : null,
            params.lift !== undefined ? `lift${params.lift >= 0 ? "+" : ""}${params.lift}` : null,
            params.wrist !== undefined ? `wrist${params.wrist >= 0 ? "+" : ""}${params.wrist}` : null,
            params.grip !== undefined ? `grip=${params.grip}` : null,
          ].filter(Boolean);
          const text =
            `Applied ${deltas.join(", ") || "(no change)"}. Current actuators: ${json(body.ctrl)} ` +
            `(reel recorded ${body.recorded ?? 0} steps)`;
          const details = { action, ctrl: body.ctrl, recorded: body.recorded };
          if (params.panel && body.panel_base64) {
            return {
              content: [
                { type: "text", text },
                { type: "image", data: body.png_base64, mimeType: "image/png" },
                { type: "image", data: body.panel_base64, mimeType: "image/png" },
              ],
              details,
            };
          }
          return withFrame(text, body.png_base64, details);
        }
        case "move_ee": {
          // IK-move a site to a world pose (for multi-DOF arms).
          if (!params.pos || params.pos.length !== 3) {
            return fail("action 'move_ee' needs pos [x,y,z].", { action });
          }
          const res = await controlPost(
            port,
            "/command",
            {
              cmd: "move_ee",
              pos: params.pos,
              quat: params.quat,
              ...(params.site ? { site: params.site } : {}),
              ...(params.body ? { body: params.body } : {}),
              ...(params.posture ? { posture: params.posture } : {}),
              ...(params.allow_collision !== undefined
                ? { allow_collision: params.allow_collision }
                : {}),
              steps: params.steps ?? 6,
              settle: params.settle ?? 1.5,
              camera: params.camera ?? "camera_rgb",
              panel: params.panel ?? false,
              width: params.width ?? 640,
              height: params.height ?? 480,
            },
            120000,
          );
          const body = res.body as {
            ok?: boolean;
            png_base64?: string;
            panel_base64?: string;
            ik_error?: unknown;
            recorded?: number;
            blocked?: boolean;
            planned?: number;
            site?: string;
            collision?: Array<{ geom1: string; geom2: string; dist: number }>;
            error?: string;
          };
          if (!body?.ok || !body.png_base64) {
            return fail(`move_ee failed: ${body?.error ?? "no frame"}`, { action });
          }
          const ee = body.site ?? params.site ?? params.body ?? "end-effector";
          const blocked = body.blocked
            ? ` BLOCKED at ${body.collision?.map((c) => `${c.geom1}/${c.geom2}`).join(", ")}`
            : "";
          const text =
            `Moved ${ee} to ${json(params.pos)} ` +
            `(ik_error ${json(body.ik_error)}; ${body.planned ?? "?"} planned, ${body.blocked ? "stopped early" : "full"} path${blocked}; ` +
            `reel recorded ${body.recorded ?? 0} steps)`;
          const details = {
            action,
            ik_error: body.ik_error,
            recorded: body.recorded,
            blocked: body.blocked ?? false,
            collision: body.collision,
          };
          if (params.panel && body.panel_base64) {
            return {
              content: [
                { type: "text", text },
                { type: "image", data: body.png_base64, mimeType: "image/png" },
                { type: "image", data: body.panel_base64, mimeType: "image/png" },
              ],
              details,
            };
          }
          return withFrame(text, body.png_base64, details);
        }
        case "record_start": {
          // Continuous capture of the 3D panel + cameras (real video, not key frames).
          const res = await controlPost(port, "/command", {
            cmd: "record_start",
            fps: params.fps ?? 6,
            cameras: params.cameras,
            width: params.width ?? 640,
            height: params.height ?? 480,
          });
          return ok(json(res.body), { action });
        }
        case "record_stop": {
          const res = await controlPost(port, "/command", { cmd: "record_stop" });
          return ok(json(res.body), { action });
        }
        case "contains": {
          // VLABench-style `contain` success check: is `object` inside `container`?
          const res = await controlPost(port, "/command", {
            cmd: "contains",
            object: params.object,
            container: params.container,
          });
          return ok(json(res.body), { action, ...(res.body as object) });
        }
        case "audit": {
          // Headless physics audit: penetration, settle, mass/inertia, limits,
          // saturation, energy. Defaults to the loaded model; `scene`/`path`
          // audits another model without loading it.
          const res = await controlPost(
            port,
            "/command",
            {
              cmd: "audit",
              ...(params.scene ? { scene: params.scene } : {}),
              ...(params.path ? { path: params.path } : {}),
              ...(params.seconds !== undefined ? { seconds: params.seconds } : {}),
            },
            900000,
          );
          const report = res.body as {
            ok?: boolean;
            source?: string;
            summary?: { status?: string; errors?: number; warnings?: number };
            error?: string;
          };
          if (report?.error) return fail(`audit failed: ${report.error}`, { action });
          const status = report?.summary?.status ?? "unknown";
          return ok(
            `Physics audit (${report?.source ?? "loaded"}): ${status} — ` +
              `${report?.summary?.errors ?? 0} error(s), ${report?.summary?.warnings ?? 0} warning(s).\n\n` +
              json(res.body),
            { action, ...(res.body as object) },
          );
        }
        case "chapter": {
          // Marks a new attempt in the recorded reel (a QuickTime chapter).
          const res = await controlPost(port, "/command", { cmd: "chapter", title: params.title });
          return ok(json(res.body), { action });
        }
        case "reel": {
          // Compose the recorded 3D-panel + camera frames into a video.
          const res = await controlPost(
            port,
            "/command",
            {
              cmd: "reel",
              fps: params.fps ?? 10,
              hold: params.hold,
              height: params.height ?? 480,
              cols: params.cols,
              name: params.reel_name,
            },
            120000,
          );
          const body = res.body as {
            ok?: boolean;
            path?: string;
            format?: string;
            steps?: number;
            frames?: number;
            seconds?: number;
            hold_seconds?: number[];
            columns?: string[];
            layout?: string;
            rows?: number;
            cols?: number;
            chapters?: Array<{ start: number; end: number; title: string }>;
            error?: string;
          };
          if (!body?.ok || !body.path) {
            return fail(`reel failed: ${body?.error ?? "unknown"}`, { action });
          }
          if (body.path) await openInBrowser(body.path);
          return ok(
            `Operation reel (${body.format}, ${body.steps} steps / ${body.frames} frames, ${body.seconds}s, ` +
              `${body.layout ?? "grid"} ${body.rows}x${body.cols}, ${(body.chapters ?? []).length} chapter(s); ` +
              `sources: ${(body.columns ?? []).join(" | ")}) -> ${body.path}`,
            { action, ...body },
          );
        }
        case "reel_clear": {
          const res = await controlPost(port, "/command", { cmd: "reel_clear" });
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
      "Control the MuJoCo simulation (start [scene] | stop | open | status | scenes | scene <name> | audit [scene] | pick [object] | vision_pick [object] [camera] | camera [name] | cameras | capture | setup)",
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
        if (sub === "audit") {
          const port = runningControlPort(paths);
          if (port === null) {
            ctx.ui.notify("robotics: not running \u2014 try /robotics start", "warning");
            return;
          }
          const body: Record<string, unknown> = { cmd: "audit" };
          if (rest[0]) body.scene = rest[0];
          const res = await controlPost(port, "/command", body, 900000);
          const report = res.body as {
            source?: string;
            summary?: { status?: string; errors?: number; warnings?: number };
            error?: string;
          };
          if (report?.error) {
            ctx.ui.notify(`robotics audit failed: ${report.error}`, "error");
            return;
          }
          const level = report?.summary?.errors
            ? "error"
            : report?.summary?.warnings
              ? "warning"
              : "info";
          ctx.ui.notify(
            `robotics audit (${report?.source ?? "loaded"}): ${report?.summary?.status ?? "?"} \u2014 ` +
              `${report?.summary?.errors ?? 0} error(s), ${report?.summary?.warnings ?? 0} warning(s)`,
            level,
          );
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
