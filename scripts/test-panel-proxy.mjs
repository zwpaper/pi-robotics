// Verify the panel shim proxy fronts the MuJoCo viewer correctly:
// HTML shim injection, HTTP passthrough, and WebSocket upgrade tunneling.
import http from "node:http";
import net from "node:net";
import crypto from "node:crypto";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { createJiti } from "jiti";

const REPO_ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const jiti = createJiti(import.meta.url, { interopDefault: true });
const { startPanelProxy } = await jiti.import(
  join(REPO_ROOT, ".pi/extensions/robotics/panel-shim.ts"),
);

const VIEWER = 8080;

function get(port, path) {
  return new Promise((resolve, reject) => {
    http
      .get({ host: "127.0.0.1", port, path }, (res) => {
        const chunks = [];
        res.on("data", (c) => chunks.push(c));
        res.on("end", () => resolve({ status: res.statusCode, headers: res.headers, body: Buffer.concat(chunks) }));
      })
      .on("error", reject);
  });
}

const proxy = await startPanelProxy(VIEWER);
console.log("proxy port:", proxy.port);

const root = await get(proxy.port, "/");
console.log("GET / ->", root.status, "html", root.headers["content-type"]);
console.log("  shim injected:", root.body.toString().includes("data-pi-robotics-shim"));
console.log("  COOP stripped:", root.headers["cross-origin-opener-policy"] === undefined);
console.log("  COEP stripped:", root.headers["cross-origin-embedder-policy"] === undefined);

const js = await get(proxy.port, "/web_client.js");
console.log("GET /web_client.js ->", js.status, js.body.length, "bytes");

const model = await get(proxy.port, "/model");
console.log("GET /model ->", model.status, model.body.length, "bytes");

// Raw WebSocket upgrade tunnel: expect HTTP/1.1 101 then server frames.
const wsKey = crypto.randomBytes(16).toString("base64");
const wsOk = await new Promise((resolve) => {
  const sock = net.connect(proxy.port, "127.0.0.1", () => {
    sock.write(
      `GET /state?sid=test HTTP/1.1\r\nHost: 127.0.0.1:${proxy.port}\r\n` +
        "Upgrade: websocket\r\nConnection: Upgrade\r\n" +
        `Sec-WebSocket-Key: ${wsKey}\r\nSec-WebSocket-Version: 13\r\n\r\n`,
    );
  });
  let buf = Buffer.alloc(0);
  let done = false;
  const finish = (v) => { if (!done) { done = true; sock.destroy(); resolve(v); } };
  sock.on("data", (d) => {
    buf = Buffer.concat([buf, d]);
    if (buf.includes("101")) finish({ upgraded: true, bytes: buf.length });
  });
  sock.on("error", () => finish({ upgraded: false }));
  setTimeout(() => finish({ upgraded: buf.includes("101"), bytes: buf.length }), 2000);
});
console.log("WS /state upgrade ->", JSON.stringify(wsOk));

proxy.close();
console.log("DONE");
