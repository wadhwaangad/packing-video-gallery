// Runner.step() of packing/runner.py, for a whole motion at once, in the page's worker: the
// commands Python collected (carriage, rotation and finger targets at 500 Hz) are applied
// here every physics step with the same rotation PD, dial detents and suitcase mechanisms
// (trolley latch, zip pulls in the dock, lock button, buckle, zip holding the lid), and a
// frame is recorded every 40 steps, as Runner._record does. Keep it in step with runner.py.
self.RunnerBatch = (() => {
  // ---------------------------------------------------------------- Runner._rot_torque
  function rotTorque(d, c) {
    for (let s = 0; s < 2; s++) {
      const qa = c.rotQadr[s], qt = c.rotQ.subarray(4 * s, 4 * s + 4);
      const w1 = d.qpos[qa], x1 = -d.qpos[qa + 1], y1 = -d.qpos[qa + 2], z1 = -d.qpos[qa + 3];
      const [w2, x2, y2, z2] = qt;
      let ew = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, ex = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
      let ey = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, ez = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;
      if (ew < 0) { ew = -ew; ex = -ex; ey = -ey; ez = -ez; }
      const sn = Math.sqrt(ex * ex + ey * ey + ez * ez);
      let tx = 0, ty = 0, tz = 0;
      if (sn >= 1e-12) {
        const ang = 2 * Math.atan2(sn, ew), k = c.rotKp * ang / sn;
        tx = k * ex; ty = k * ey; tz = k * ez;
        const n = Math.sqrt(tx * tx + ty * ty + tz * tz);
        if (n > c.torqueCap) { const f = c.torqueCap / n; tx *= f; ty *= f; tz *= f; }
      }
      const ids = c.rotAid.subarray(3 * s, 3 * s + 3);
      d.ctrl[ids[0]] = tx; d.ctrl[ids[1]] = ty; d.ctrl[ids[2]] = tz;
    }
  }
  // ---------------------------------------------------------------- the mechanisms
  function latchTrolley(m, c, level) {
    const j = c.hw.trolley[0];
    if (level == null) { m.jnt_range[2 * j] = c.range0[2 * j]; m.jnt_range[2 * j + 1] = c.range0[2 * j + 1]; }
    else { const q = c.trolleyLevels[level]; m.jnt_range[2 * j] = q - 0.0005; m.jnt_range[2 * j + 1] = q + 0.0005; }
    c.mech.trolley_level = level;
  }
  function latchTab(m, d, c, k, on) {
    const jt = c.hw.tab[k][0], [js, qs] = c.hw.slider[k];
    if (on) {
      m.jnt_range[2 * jt] = Math.PI - 0.04; m.jnt_range[2 * jt + 1] = Math.PI;
      const q = d.qpos[qs]; m.jnt_range[2 * js] = q - 0.0005; m.jnt_range[2 * js + 1] = q + 0.0005;
    } else {
      for (const j of [jt, js]) { m.jnt_range[2 * j] = c.range0[2 * j]; m.jnt_range[2 * j + 1] = c.range0[2 * j + 1]; }
    }
    c.mech.tabs[k] = on;
  }
  function latchBuckle(m, c, on) {
    const j = c.hw.buckle[0];
    if (on) { m.jnt_range[2 * j] = 0.0; m.jnt_range[2 * j + 1] = 0.0008; }
    else { m.jnt_range[2 * j] = c.range0[2 * j]; m.jnt_range[2 * j + 1] = c.range0[2 * j + 1]; }
    c.mech.buckle = on;
  }
  const dialDigits = (d, c) => c.hw.dial.map(([, , qa]) => (((Math.round(d.qpos[qa] / (2 * Math.PI / 10))) % 10) + 10) % 10);
  function zipClosed(d, c) {
    let sum = 0;
    for (const k of ['a', 'b']) {
      const [j, qa] = c.hw.slider[k], lo = c.range0[2 * j], hi = c.range0[2 * j + 1], q = d.qpos[qa];
      sum += k === 'a' ? (q - lo) / (hi - lo) : (hi - q) / (hi - lo);
    }
    return Math.min(1, Math.max(0, sum / 2));
  }
  function hardwareReading(d, c) {
    const r4 = v => Math.round(v * 1e4) / 1e4;
    return { dials: dialDigits(d, c), trolley: r4(d.qpos[c.hw.trolley[1]]), trolley_level: c.mech.trolley_level,
             zip_closed: Math.round(zipClosed(d, c) * 1000) / 1000,
             tabs: { a: Math.round(d.qpos[c.hw.tab.a[1]] * 1000) / 1000, b: Math.round(d.qpos[c.hw.tab.b[1]] * 1000) / 1000 },
             docked: { ...c.mech.tabs }, buckle: r4(d.qpos[c.hw.buckle[1]]), buckled: c.mech.buckle };
  }
  function mechanisms(mj, m, d, c, ev, k) {
    const H = c.hw, mech = c.mech;
    const q = d.qpos[H.trolley[1]], bq = d.qpos[H.trolley_btn];
    if (bq > 0.75 * c.trolleyBtnTravel) {
      if (mech.trolley_level != null) { latchTrolley(m, c, null); ev.push([k, 'trolley_unlatched', { q: Math.round(q * 1e4) / 1e4 }]); }
    } else if (mech.trolley_level == null && bq < 0.3 * c.trolleyBtnTravel) {
      for (let i = 0; i < c.trolleyLevels.length; i++)
        if (Math.abs(q - c.trolleyLevels[i]) < 0.003) { latchTrolley(m, c, i); ev.push([k, 'trolley_latched', { level: i, q: Math.round(q * 1e4) / 1e4 }]); break; }
    }
    for (const t of ['a', 'b']) {
      const qt = H.tab[t][1], [js, qs] = H.slider[t];
      const lo = c.range0[2 * js], hi = c.range0[2 * js + 1];
      const atDock = Math.abs(d.qpos[qs] - (t === 'a' ? hi : lo)) < 0.004;
      if (mech.tab_block[t] && d.qpos[qt] < Math.PI - 0.4) mech.tab_block[t] = false;
      if (!mech.tabs[t] && !mech.tab_block[t] && atDock && d.qpos[qt] > Math.PI - 0.10) { latchTab(m, d, c, t, true); ev.push([k, 'pull_docked', { tab: t }]); }
    }
    if (d.qpos[H.lock_btn] > 0.7 * c.lockBtnTravel) {
      if (mech.tabs.a || mech.tabs.b) {
        if (dialDigits(d, c).join('') === mech.code) {
          latchTab(m, d, c, 'a', false); latchTab(m, d, c, 'b', false);
          mech.tab_block = { a: true, b: true };
          ev.push([k, 'pulls_released', { dials: dialDigits(d, c) }]);
        } else if (mech.lock_events < 1) { mech.lock_events += 1; ev.push([k, 'lock_button_blocked', { dials: dialDigits(d, c) }]); }
      }
    } else mech.lock_events = 0;
    const [, qb] = H.buckle;
    if (!mech.buckle && d.qpos[qb] < 0.0015) { latchBuckle(m, c, true); ev.push([k, 'buckle_latched', {}]); }
    else if (mech.buckle) {
      let squeezed = 0;
      const vec = d.contact, f = c.f6;
      for (let i = 0, n = d.ncon; i < n; i++) {
        const ct = vec.get(i), g1 = ct.geom1, g2 = ct.geom2; ct.delete();
        const ti = c.buckleTabs.indexOf(g1) >= 0 ? g1 : c.buckleTabs.indexOf(g2) >= 0 ? g2 : -1;
        if (ti < 0) continue;
        const other = ti === g1 ? g2 : g1;
        if (c.hand.has(m.geom_bodyid[other])) {
          mj.mj_contactForce(m, d, i, f);
          if (Math.abs(f[0]) > 1.5) squeezed |= 1 << c.buckleTabs.indexOf(ti);
        }
      }
      if (squeezed === 3) { latchBuckle(m, c, false); ev.push([k, 'buckle_released', {}]); }
    }
    const zc = zipClosed(d, c), lidQ = d.qpos[m.jnt_qposadr[H.lid]], L = H.lid;
    if (!c.zipLocked && zc > 0.05 && lidQ < 2.5 * Math.PI / 180) {
      m.jnt_range[2 * L] = 0.0; m.jnt_range[2 * L + 1] = 0.003; c.zipLocked = true;
      ev.push([k, 'zipper_engaged', { lid_angle_deg: Math.round(lidQ * 18000 / Math.PI) / 100, closed_fraction: Math.round(zc * 1000) / 1000 }]);
    } else if (c.zipLocked && zc < 0.04) {
      m.jnt_range[2 * L] = c.range0[2 * L]; m.jnt_range[2 * L + 1] = c.range0[2 * L + 1]; c.zipLocked = false;
      ev.push([k, 'zipper_released', { lid_angle_deg: Math.round(lidQ * 18000 / Math.PI) / 100 }]);
    }
    c.zipProgress = zc;
  }
  // ---------------------------------------------------------------- the batch
  // p: {n, ctrlAt (Int32), ctrlIds (Int32), ctrlVals (F64, rows of ctrlIds), rotQs (F64, 8 per row),
  //     recAt (Int32), frameEvery, cfg (see mujoco_pyodide.batch)}
  function run(mj, m, d, p) {
    const c = p.cfg;
    c.rotQ = c.rotQ || new Float64Array(8);
    c.f6 = new Float64Array(6);
    c.hand = new Set(c.handBodies);
    const nb = m.nbody, nI = p.ctrlIds.length;
    const frames = [], fx = [], fq = [], ev = [];
    let row = 0, rec = 0;
    for (let k = 0; k < p.n; k++) {
      if (row < p.ctrlAt.length && p.ctrlAt[row] === k) {
        const base = row * nI;
        for (let i = 0; i < nI; i++) d.ctrl[p.ctrlIds[i]] = p.ctrlVals[base + i];
        c.rotQ.set(p.rotQs.subarray(row * 8, row * 8 + 8));
        mechanisms(mj, m, d, c, ev, k);
        row++;
      }
      for (let i = 0; i < c.detentDof.length; i++) d.qfrc_applied[c.detentDof[i]] = -c.detentK * Math.sin(10 * d.qpos[c.detentQadr[i]]);
      rotTorque(d, c);
      mj.mj_step(m, d);
      if (rec < p.recAt.length && p.recAt[rec] === k) {
        fx.push(Float32Array.from(d.xpos)); fq.push(Float32Array.from(d.xquat));
        frames.push({ t: Math.round(d.time * 1e4) / 1e4, lid: d.qpos[m.jnt_qposadr[c.hw.lid]], zip_progress: c.zipProgress,
                      zip: c.zipLocked, hw: hardwareReading(d, c) });
        rec++;
      }
    }
    return { frames, xpos: fx, xquat: fq, events: ev, mech: c.mech, zipLocked: c.zipLocked, zipProgress: c.zipProgress,
             rotQ: Array.from(c.rotQ) };
  }
  return { run };
})();
