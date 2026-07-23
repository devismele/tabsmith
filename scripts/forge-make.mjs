import { spawn } from "node:child_process";
import path from "node:path";

const environment = { ...process.env };
if (process.platform === "win32") {
  const windows = environment.SystemRoot || environment.WINDIR || "C:\\Windows";
  const powershell = path.join(windows, "System32", "WindowsPowerShell", "v1.0");
  environment.PATH = `${powershell};${environment.PATH || ""}`;
}
const cli = path.resolve("node_modules", "@electron-forge", "cli", "dist", "electron-forge.js");
const child = spawn(process.execPath, [cli, "make"], { env: environment, stdio: "inherit", windowsHide: true });
child.on("error", (error) => { throw error; });
child.on("exit", (code, signal) => {
  if (signal) process.kill(process.pid, signal);
  else process.exitCode = code ?? 1;
});
