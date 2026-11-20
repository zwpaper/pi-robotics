/**
 * Panel shim proxy for embedding the MuJoCo Studio web viewer in pi-dashboard.
 *
 * Why this exists: the MuJoCo web viewer builds its WebSocket URLs from
 * ``window.location.host`` and fetches ``/model`` with a root-absolute path.
 * pi-dashboard embeds a declared loopback server through a same-origin path
 * proxy at ``/live/<id>/…`` (optionally ``/live/<id>/~c/<token>/…``), which
 * strips the prefix before forwarding upstream. The page therefore has to make
 * its root-absolute requests relative to the CURRENT path so they survive the
 * prefix.
 *
 * This proxy sits in front of the viewer and:
 *   - injects a tiny shim that rewrites ``WebSocket`` and ``fetch`` targets from
 *     root-absolute to the current directory (no-op when served top-level),
 *   - forwards HTTP and tunnels WebSocket upgrades to the viewer,
 *   - drops COOP/COEP (which the viewer emits twice, invalidly, and which are
 *     unnecessary when framed) plus ``Content-Length`` for the rewritten HTML.
 *
 * It binds loopback only and lives for the pi process; no auth, no egress.
 */

import http from "node:http";
import net from "node:net";

/** Response headers dropped on the proxied hop. */
const STRIP_RESPONSE_HEADERS = new Set([
  "content-length",
  "cross-origin-opener-policy",
  "cross-origin-embedder-policy",
]);

/**
 * Injected before the viewer's own scripts. Makes root-absolute HTTP and
 * WebSocket targets resolve under the page's current directory, so the same
 * page works top-level (base ``/``) and inside pi-dashboard's ``/live/<id>/``
 * path proxy.
 */
const PANEL_SHIM = `<script>(function () {
  var base = location.pathname || "/";
  if (base.charAt(base.length - 1) !== "/") base += "/";
  if (base === "/") return;
  function rewrite(u) {
    if (u.charAt(0) !== "/" || u.charAt(1) === "/") return u;
    return base + u.slice(1);
  }
  var OrigWS = window.WebSocket;
  function ShimWS(url, protocols) {
    if (typeof url === "string" && /^wss?:\\/\\//i.test(url)) {
      try { var parsed = new URL(url); parsed.pathname = rewrite(parsed.pathname); url = parsed.toString(); } catch (e) {}
    }
    return protocols === undefined ? new OrigWS(url) : new OrigWS(url, protocols);
  }
  ShimWS.prototype = OrigWS.prototype;
  ["CONNECTING", "OPEN", "CLOSING", "CLOSED"].forEach(function (k) { try { ShimWS[k] = OrigWS[k]; } catch (e) {} });
  window.WebSocket = ShimWS;
  var origFetch = window.fetch;
  if (origFetch) {
    window.fetch = function (input, init) {
      if (typeof input === "string") input = rewrite(input);
      return origFetch.call(this, input, init);
    };
  }
})();</script>`;

export interface PanelProxy {
  /** Loopback port the dashboard should embed. */
  port: number;
  close(): void;
}

function injectShim(html: string): string {
  if (html.includes("data-pi-robotics-shim")) return html;
  const tagged = PANEL_SHIM.replace("<script>", '<script data-pi-robotics-shim="1">');
  if (/<head[^>]*>/i.test(html)) {
    return html.replace(/<head[^>]*>/i, (m) => `${m}${tagged}`);
  }
  return tagged + html;
}

/** Starts a loopback proxy in front of the viewer on ``upstreamPort``. */
export function startPanelProxy(upstreamPort: number): Promise<PanelProxy> {
  const server = http.createServer((req, res) => {
    const headers = { ...req.headers, host: `127.0.0.1:${upstreamPort}` };
    const upstream = http.request(
      {
        host: "127.0.0.1",
        port: upstreamPort,
        method: req.method,
        path: req.url,
        headers,
      },
      (up) => {
        const outHeaders: http.OutgoingHttpHeaders = { ...up.headers };
        for (const key of Object.keys(outHeaders)) {
          if (STRIP_RESPONSE_HEADERS.has(key.toLowerCase())) delete outHeaders[key];
        }
        const contentType = String(up.headers["content-type"] ?? "");
        if (contentType.includes("text/html")) {
          const chunks: Buffer[] = [];
          up.on("data", (c: Buffer) => chunks.push(c));
          up.on("end", () => {
            const body = Buffer.from(injectShim(Buffer.concat(chunks).toString("utf8")));
            outHeaders["content-length"] = String(body.byteLength);
            outHeaders["content-type"] = "text/html; charset=utf-8";
            res.writeHead(up.statusCode ?? 200, outHeaders);
            res.end(body);
          });
          return;
        }
        res.writeHead(up.statusCode ?? 200, outHeaders);
        up.pipe(res);
      },
    );
    upstream.on("error", () => {
      if (!res.headersSent) res.writeHead(502);
      res.end("panel upstream unavailable");
    });
    req.pipe(upstream);
  });

  // WebSocket upgrades are a raw byte tunnel after the 101 response.
  server.on("upgrade", (req, clientSocket, head) => {
    const upstream = net.connect(upstreamPort, "127.0.0.1", () => {
      const lines = [`${req.method} ${req.url} HTTP/1.1`];
      for (let i = 0; i < req.rawHeaders.length; i += 2) {
        lines.push(`${req.rawHeaders[i]}: ${req.rawHeaders[i + 1]}`);
      }
      upstream.write(lines.join("\r\n") + "\r\n\r\n");
      if (head && head.length) upstream.write(head);
      clientSocket.pipe(upstream);
      upstream.pipe(clientSocket);
    });
    upstream.on("error", () => clientSocket.destroy());
    clientSocket.on("error", () => upstream.destroy());
  });

  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      const port = typeof address === "object" && address ? address.port : 0;
      resolve({ port, close: () => server.close() });
    });
  });
}
