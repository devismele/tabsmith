# Third-party software and release checklist

Tabsmith currently depends on the following major projects. This is an engineering inventory, not legal advice; include the exact license texts and notices from the locked release artifacts before public distribution.

| Component | Purpose | License / source |
|---|---|---|
| Electron | Desktop runtime | MIT — https://github.com/electron/electron |
| Python 3.13 embedded distribution | Private packaged interpreter | PSF License — https://www.python.org/downloads/release/python-31314/ |
| Express | Localhost HTTP service | MIT — https://github.com/expressjs/express |
| Spotify Basic Pitch | Polyphonic note transcription | Apache-2.0 — https://github.com/spotify/basic-pitch |
| TensorFlow.js | Basic Pitch inference | Apache-2.0 — https://github.com/tensorflow/tfjs |
| Audio Separator | Stem-separation API | MIT — https://github.com/nomadkaraoke/python-audio-separator |
| Demucs / `htdemucs_6s` | Six-stem model | MIT — https://github.com/facebookresearch/demucs |
| PyTorch | Demucs inference | BSD-style — https://github.com/pytorch/pytorch |
| yt-dlp | Authorized YouTube audio import | Unlicense — https://github.com/yt-dlp/yt-dlp |
| FFmpeg | Audio decoding/encoding | LGPL/GPL depending on build options — https://ffmpeg.org/legal.html |

## Before a public installer

- Capture all transitive production licenses from both npm and Python lock files.
- Preserve copyright and attribution files required by Apache-2.0, MIT, BSD, and other packages.
- The currently cached FFmpeg binary reports `--enable-gpl --enable-version3`; review the resulting GPL obligations, ship the appropriate license, and provide corresponding source/build information where required.
- Document the origin, checksum, license, and version of every bundled model and binary.
- Sign the Windows installer and executables with an Authenticode certificate.
- Run `npm audit --omit=dev`; production dependencies currently report zero known vulnerabilities. Electron Forge's build-only dependency tree currently has advisories and should be updated or overridden only after packaging regression tests.
