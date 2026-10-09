// Reproducible renderer for the three simulation camera videos.
// Usage: node render-packing-video.cjs [fps]
const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { chromium } = require('C:/Users/wadhw/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

const root = __dirname;
const fps = Math.max(1, Math.min(15, Number(process.argv[2]) || 8));
const cameras = process.argv.includes('--egocentric-only') ? ['egocentric'] : ['overview', 'operator', 'egocentric'];
const duration = 90;
const frameCount = duration * fps;
const media = path.join(root, 'packing', 'media');
const work = path.join(root, '.packing-video-frames');
fs.mkdirSync(media, { recursive: true });
fs.rmSync(work, { recursive: true, force: true });

async function main() {
  const browser = await chromium.launch({ headless: true, executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', args: ['--allow-file-access-from-files'] });
  try {
    const page = await browser.newPage({ viewport: { width: 1600, height: 3500 }, deviceScaleFactor: 2 });
    page.on('console', message => { if (message.type() === 'error') console.error(message.text()); });
    await page.goto(`file:///${path.join(root, 'packing', 'index.html').replace(/\\/g, '/')}`, { waitUntil: 'load' });
    await page.waitForFunction(() => document.querySelector('#overview')?.width > 0 && document.querySelector('#operator')?.width > 0);
    await page.addStyleTag({ content: '.stage-grid{grid-template-columns:1fr!important}.stage-grid .film-card{max-width:none!important}.hud{visibility:hidden!important}' });
    await page.waitForTimeout(500);
    for (const camera of cameras) fs.mkdirSync(path.join(work, camera), { recursive: true });
    for (let frame = 0; frame < frameCount; frame++) {
      const time = frame * duration / frameCount;
      await page.evaluate(t => {
        const input = document.querySelector('#timeline');
        input.value = String(t);
        input.dispatchEvent(new Event('input', { bubbles: true }));
      }, time);
      for (const camera of cameras.filter(camera => camera !== 'egocentric')) {
        const canvas = page.locator(`#${camera}`);
        const target = path.join(work, camera, `${String(frame).padStart(5, '0')}.jpg`);
        await canvas.screenshot({ path: target, type: 'jpeg', quality: 90, scale: 'device' });
      }
      if (cameras.includes('egocentric')) {
        await page.locator('#view-mode').click();
        await page.locator('.stage-grid .film-card').nth(1).locator('.viewport').screenshot({ path: path.join(work, 'egocentric', `${String(frame).padStart(5, '0')}.jpg`), type: 'jpeg', quality: 90, scale: 'device' });
        await page.locator('#view-mode').click();
      }
      if (frame % Math.max(1, Math.floor(fps * 10)) === 0) console.log(`Rendered ${frame}/${frameCount} frames (${time.toFixed(1)}s)`);
    }
  } finally {
    await browser.close();
  }
  const ffmpeg = 'C:\\Users\\wadhw\\AppData\\Local\\Microsoft\\WinGet\\Links\\ffmpeg.exe';
  for (const camera of cameras) {
  const input = path.join(work, camera, '%05d.jpg');
  const output = path.join(media, `packing-${camera}.mp4`);
    const inputArg = path.relative(root, input).replace(/\\/g, '/');
    const outputArg = path.relative(root, output).replace(/\\/g, '/');
    const posterIndex = camera === 'egocentric' ? Math.floor(frameCount * 0.61) : 0;
    fs.copyFileSync(path.join(work, camera, `${String(posterIndex).padStart(5, '0')}.jpg`), path.join(media, `packing-${camera}.jpg`));
    const result = spawnSync(ffmpeg, ['-y', '-hide_banner', '-loglevel', 'error', '-framerate', String(fps), '-i', inputArg, '-vf', 'scale=2560:1440:flags=lanczos', '-c:v', 'libx264', '-preset', 'medium', '-crf', '21', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', outputArg], { stdio: 'inherit', cwd: root });
    if (result.status !== 0) throw new Error(`ffmpeg failed for ${camera}: ${result.status}`);
    console.log(`Wrote ${path.relative(root, output)}`);
  }
  fs.rmSync(work, { recursive: true, force: true });
}

main().catch(error => { console.error(error); process.exitCode = 1; });
