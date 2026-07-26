module.exports = {
  packagerConfig: {
    name: "Tabsmith",
    executableName: "Tabsmith",
    icon: "assets/icon",
    appBundleId: "com.tabsmith.desktop",
    asar: true,
    extraResource: [
      ".runtime/tabsmith-runtime.zip",
    ],
    ignore: [
      /^\/.venv($|\/)/,
      /^\/.models($|\/)/,
      /^\/.runtime($|\/)/,
      /^\/dist-desktop($|\/)/,
      /^\/out($|\/)/,
      /^\/coverage($|\/)/,
      // Human reference metadata and reports are local evaluation artifacts,
      // not application content and must never ship in the installer.
      /^\/evaluation($|\/)/,
      /^\/tests($|\/)/,
      /^\/scripts($|\/)/,
      /^\/ml($|\/)/,
      // Development-only learned weights are code-split in flagged dev builds
      // and explicitly excluded from Electron release source packaging.
      /^\/src\/learnedHarmony\/model($|\/)/,
      /^\/server.*\.log$/,
    ],
  },
  makers: [
    {
      name: "@electron-forge/maker-zip",
      platforms: ["win32"],
    },
    {
      name: "@electron-forge/maker-squirrel",
      config: {
        name: "Tabsmith",
        authors: "Tabsmith",
        description: "Local guitar stem separation, chord recognition, and tablature transcription.",
        setupExe: "Tabsmith-Setup.exe",
        setupIcon: "assets/icon.ico",
      },
    },
  ],
};
