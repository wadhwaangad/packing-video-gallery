// Runs the symbolic simulator (packing/*.py, numpy only) in Pyodide, off the page's thread.
// Messages: {id, cmd, args} -> {id, ok, out | error}. The commands are web.handle()'s.
const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/';
importScripts(PYODIDE + 'pyodide.js');

let py = null;
const ready = (async () => {
  py = await loadPyodide({ indexURL: PYODIDE });
  postMessage({ progress: 'Loading numpy…' });
  // the wheel is 12 MB off the CDN and a dropped download only logs, so check and retry
  for (let i = 1; ; i++) {
    await py.loadPackage('numpy');
    try { py.runPython('import numpy'); break; } catch (e) {
      if (i === 3) throw new Error('numpy did not download (3 tries); check the connection and reload');
      postMessage({ progress: 'Loading numpy (retry ' + i + ')…' });
      await new Promise(r => setTimeout(r, 2000 * i));
    }
  }
  const mods = await (await fetch('modules.json', { cache: 'no-cache' })).json();
  py.FS.mkdirTree('/sim');
  for (const m of mods) {
    postMessage({ progress: 'Loading ' + m + '…' });
    const r = await fetch(new URL('../packing/' + m, self.location), { cache: 'no-cache' });
    if (!r.ok) throw new Error('could not fetch packing/' + m + ' (' + r.status + ')');
    py.FS.writeFile('/sim/' + m, await r.text());
  }
  postMessage({ progress: 'Starting the simulator…' });
  py.runPython("import sys, json\nsys.path.insert(0, '/sim')\nimport web");
  postMessage({ ready: true });
})().catch(err => postMessage({ fatal: String(err && err.message || err) }));

self.onmessage = async (e) => {
  const { id, cmd, args } = e.data;
  try {
    await ready;
    if (!py) throw new Error('Python did not load');
    py.globals.set('REQ', JSON.stringify({ cmd, args: args || {} }));
    const out = py.runPython('json.dumps(web.handle(json.loads(REQ)))');
    postMessage({ id, ok: true, out: JSON.parse(out) });
  } catch (err) {
    postMessage({ id, ok: false, error: String(err && err.message || err) });
  }
};
