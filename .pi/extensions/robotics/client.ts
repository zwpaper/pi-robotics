/** HTTP client for the pi-robotics loopback control API. */

export interface ControlResult {
  ok: boolean;
  status: number;
  body: unknown;
  error?: string;
}

async function request(
  method: "GET" | "POST",
  port: number,
  path: string,
  body: unknown,
  timeoutMs: number,
): Promise<ControlResult> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`http://127.0.0.1:${port}${path}`, {
      method,
      headers: { "content-type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
    });
    const text = await res.text();
    let parsed: unknown = text;
    try {
      parsed = JSON.parse(text);
    } catch {
      /* leave as text */
    }
    return { ok: res.ok, status: res.status, body: parsed };
  } catch (error) {
    return {
      ok: false,
      status: 0,
      body: null,
      error: (error as Error).name === "AbortError" ? "timeout" : (error as Error).message,
    };
  } finally {
    clearTimeout(timer);
  }
}

export function controlGet(
  port: number,
  path: string,
  timeoutMs = 8000,
): Promise<ControlResult> {
  return request("GET", port, path, undefined, timeoutMs);
}

export function controlPost(
  port: number,
  path: string,
  body: unknown,
  timeoutMs = 30000,
): Promise<ControlResult> {
  return request("POST", port, path, body, timeoutMs);
}
