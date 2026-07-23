import { app, BrowserWindow, dialog, session } from "electron";
import { access, mkdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import extractZip from "extract-zip";
import { startTabsmithServer } from "../server.mjs";

const projectRoot = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const singleInstance = app.requestSingleInstanceLock();
let window = null;
let runtime = null;

if (!singleInstance) app.quit();

function firstRunWindow() {
  const splash = new BrowserWindow({
    width: 520,
    height: 270,
    frame: false,
    resizable: false,
    backgroundColor: "#10120f",
    webPreferences: { contextIsolation: true, nodeIntegration: false, sandbox: true },
  });
  const html = `<!doctype html><style>body{margin:0;display:grid;place-items:center;height:100vh;color:#eff2e8;background:#10120f;font-family:Segoe UI,sans-serif}.card{text-align:center;padding:34px}.mark{margin:auto;width:58px;height:58px;border:7px solid #c7f36b;border-radius:50%;color:#ff895f;font-size:38px;line-height:50px}h1{font-size:24px;margin:19px 0 8px}p{color:#999f92;line-height:1.5}.bar{height:4px;background:#30362c;overflow:hidden;border-radius:9px}.bar:after{content:"";display:block;width:38%;height:100%;background:#c7f36b;animation:scan 1.2s ease-in-out infinite alternate}@keyframes scan{to{transform:translateX(165%)}}</style><div class="card"><div class="mark">⌁</div><h1>Preparing Tabsmith</h1><p>Unpacking the local guitar-separation runtime.<br>This happens once and may take a few minutes.</p><div class="bar"></div></div>`;
  void splash.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(html)}`);
  return splash;
}

async function configureRuntime() {
  if (app.isPackaged) {
    const runtimeRoot = path.join(app.getPath("userData"), "runtime", `${app.getVersion()}-${process.arch}-py31314-amd64-v1`);
    const marker = path.join(runtimeRoot, ".ready");
    try {
      await access(marker);
    } catch {
      const expectedParent = path.join(app.getPath("userData"), "runtime");
      if (path.dirname(runtimeRoot) !== expectedParent) throw new Error("Unsafe desktop runtime path.");
      const splash = firstRunWindow();
      await rm(runtimeRoot, { recursive: true, force: true });
      await mkdir(runtimeRoot, { recursive: true });
      try {
        await extractZip(path.join(process.resourcesPath, "tabsmith-runtime.zip"), { dir: runtimeRoot });
        await writeFile(marker, `${new Date().toISOString()}\n`);
      } catch (error) {
        await rm(runtimeRoot, { recursive: true, force: true }).catch(() => {});
        throw error;
      } finally {
        splash.close();
      }
    }
    process.env.TABSMITH_PYTHON = path.join(
      runtimeRoot,
      process.platform === "win32" ? "python.exe" : "bin/python",
    );
    process.env.TABSMITH_MODEL_DIR = path.join(runtimeRoot, "models", "audio-separator");
    process.env.TABSMITH_SEPARATOR_SCRIPT = path.join(runtimeRoot, "separate_guitar.py");
  }
  process.env.TABSMITH_DIST_DIR = path.join(app.getAppPath(), "dist");
  process.env.TABSMITH_CACHE_DIR = path.join(app.getPath("userData"), "cache", "separation");
  process.env.TABSMITH_TRANSCRIPTION_CACHE_DIR = path.join(
    app.getPath("userData"),
    "cache",
    "transcription",
  );
  process.env.TABSMITH_EVALUATION_DIR = path.join(app.getPath("userData"), "evaluation");
}

async function startRuntime() {
  await configureRuntime();
  runtime = await startTabsmithServer({
    production: true,
    port: 0,
    staticRoot: process.env.TABSMITH_DIST_DIR,
  });
  return runtime;
}

function createWindow(url) {
  window = new BrowserWindow({
    title: "Tabsmith",
    width: 1280,
    height: 900,
    minWidth: 860,
    minHeight: 640,
    backgroundColor: "#10120f",
    show: false,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      devTools: !app.isPackaged,
    },
  });
  window.removeMenu();
  window.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  window.webContents.on("will-navigate", (event, target) => {
    if (!target.startsWith(`${url}/`)) event.preventDefault();
  });
  window.once("ready-to-show", () => window?.show());
  window.on("closed", () => { window = null; });
  void window.loadURL(url);
}

app.setAppUserModelId("com.tabsmith.desktop");
app.on("second-instance", () => {
  if (window?.isMinimized()) window.restore();
  window?.focus();
});
app.on("window-all-closed", () => app.quit());
app.on("before-quit", () => { void runtime?.close().catch(() => {}); });

if (singleInstance) {
  app.whenReady().then(async () => {
    session.defaultSession.setPermissionRequestHandler((_webContents, _permission, callback) => callback(false));
    const server = await startRuntime();
    if (process.argv.includes("--smoke-test")) {
      const health = await fetch(`${server.url}/api/health`).then((response) => response.json());
      await new Promise((resolve) => process.stdout.write(`${JSON.stringify({ desktop: true, health })}\n`, resolve));
      await server.close();
      runtime = null;
      process.exit(0);
      return;
    }
    createWindow(server.url);
  }).catch((error) => {
    dialog.showErrorBox("Tabsmith could not start", error instanceof Error ? error.message : String(error));
    app.quit();
  });
}
