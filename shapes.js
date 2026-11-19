// 数学の立体（結び目・メビウスの帯・トーラスなど）を WebGL で描く。
// 黒い体に、式の u・v の線（格子）を光らせ、その上を光が 1 周ずつ走る（play(create(…)) の見立て）。
// <canvas class="shape" data-shape="knot" data-a="線の色" data-b="走る光の色"> を探して回す。
// WebGL の文脈は 1 つだけ作り、描いた絵をそれぞれの canvas に写す。
(() => {
  const TAU = Math.PI * 2;
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const els = [...document.querySelectorAll("canvas.shape")];
  if (!els.length) return;

  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const len = (a) => Math.hypot(a[0], a[1], a[2]);
  const norm = (a) => { const l = len(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

  // 曲線 c(u) のまわりの管。r は太さ（u の関数でもよい）
  const tube = (c, r) => (u, v) => {
    const p = c(u), q = c(u + 1e-3);
    const T = norm(sub(q, p));
    const B = norm(cross(T, add(p, q)));
    const N = cross(B, T);
    const a = v * TAU, rr = typeof r === "function" ? r(u) : r;
    const cs = Math.cos(a) * rr, sn = Math.sin(a) * rr;
    return [p[0] + cs * N[0] + sn * B[0], p[1] + cs * N[1] + sn * B[1], p[2] + cs * N[2] + sn * B[2]];
  };

  const SURFACES = {
    // (2,3) のトーラス結び目
    knot: [tube((u) => {
      const t = u * TAU, r = 1 + 0.42 * Math.cos(3 * t);
      return [r * Math.cos(2 * t), r * Math.sin(2 * t), 0.42 * Math.sin(3 * t) * 1.5];
    }, 0.27), 260, 22, [54, 8]],
    torus: [(u, v) => {
      const a = u * TAU, b = v * TAU, r = 1 + 0.42 * Math.cos(b);
      return [r * Math.cos(a), 0.42 * Math.sin(b), r * Math.sin(a)];
    }, 72, 32, [30, 12]],
    // メビウスの帯
    mobius: [(u, v) => {
      const a = u * TAU, w = (v - 0.5) * 0.95, r = 1 + w * Math.cos(a / 2);
      return [r * Math.cos(a), w * Math.sin(a / 2), r * Math.sin(a)];
    }, 120, 14, [40, 6]],
    // 球面調和関数のような、でこぼこの球
    blob: [(u, v) => {
      const a = u * TAU, b = v * Math.PI;
      const r = 1 + 0.2 * Math.sin(3 * a) * Math.sin(b) * Math.sin(b) * Math.cos(2 * b) + 0.1 * Math.cos(4 * b);
      return [r * Math.sin(b) * Math.cos(a), r * Math.cos(b), r * Math.sin(b) * Math.sin(a)];
    }, 96, 64, [24, 14]],
    // ばね（らせんの管。両端は丸める）
    spring: [tube((u) => {
      const t = u * TAU * 3.5;
      return [0.62 * Math.cos(t), (u - 0.5) * 2.1, 0.62 * Math.sin(t)];
    }, (u) => 0.2 * Math.sqrt(Math.max(0, 1 - Math.pow(Math.abs(2 * u - 1), 16)))), 260, 18, [70, 6]],
    // 波紋: y = cos(ρ) の円板
    wave: [(u, v) => {
      const rho = u * 1.25, a = v * TAU;
      return [rho * Math.cos(a), 0.2 * Math.cos(rho * 8.5) * (1 - u * 0.45), rho * Math.sin(a)];
    }, 72, 72, [9, 28]],
  };

  function build([fn, nu, nv, grid]) {
    const pos = [], nor = [], uv = [], idx = [], e = 1e-3;
    let rad = 0;
    for (let i = 0; i <= nu; i++) for (let j = 0; j <= nv; j++) {
      const u = i / nu, v = j / nv, p = fn(u, v);
      let n = cross(sub(fn(u + e, v), fn(u - e, v)), sub(fn(u, v + e), fn(u, v - e)));
      if (!(len(n) > 1e-12)) n = len(p) > 1e-9 ? p : [0, 1, 0];
      n = norm(n);
      pos.push(p[0], p[1], p[2]); nor.push(n[0], n[1], n[2]); uv.push(u, v);
      rad = Math.max(rad, len(p));
    }
    for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) {
      const a = i * (nv + 1) + j, b = a + nv + 1;
      idx.push(a, b, a + 1, b, b + 1, a + 1);
    }
    return { pos: new Float32Array(pos.map((x) => x / rad)), nor: new Float32Array(nor), uv: new Float32Array(uv), idx: new Uint16Array(idx), grid };
  }

  const glc = document.createElement("canvas");
  const gl = glc.getContext("webgl", { alpha: true, antialias: true, premultipliedAlpha: true });
  if (!gl) { els.forEach((c) => c.classList.add("no-gl")); return; }

  const deriv = !!gl.getExtension("OES_standard_derivatives");
  const VS = `
    attribute vec3 aPos; attribute vec3 aNor; attribute vec2 aUV;
    uniform mat3 uR; uniform float uAspect;
    varying vec3 vN; varying vec3 vP; varying vec2 vUV;
    void main() {
      vec3 p = uR * aPos; vN = uR * aNor; vP = p; vUV = aUV;
      float w = 6.0 - p.z;
      gl_Position = vec4(p.x * 4.3 / uAspect, p.y * 4.3, (21.0 * w - 40.0) / 19.0, w);
    }`;
  const FS = `${deriv ? "#extension GL_OES_standard_derivatives : enable" : ""}
    precision highp float;
    uniform vec3 uA; uniform vec3 uB; uniform vec2 uGrid; uniform float uT;
    varying vec3 vN; varying vec3 vP; varying vec2 vUV;
    void main() {
      vec3 n = normalize(vN);
      vec3 v = normalize(vec3(0.0, 0.0, 6.0) - vP);
      if (dot(n, v) < 0.0) n = -n;
      vec3 l = normalize(vec3(-0.55, 0.8, 0.6));
      float d = smoothstep(0.1, 0.95, dot(n, l) * 0.5 + 0.5);
      float rim = pow(1.0 - max(dot(n, v), 0.0), 3.0);
      // 体: ほぼ黒。ふちだけ白く光る
      vec3 col = vec3(0.03) + vec3(0.085) * d + vec3(0.42) * rim;
      col += vec3(1.0) * pow(max(dot(n, normalize(l + v)), 0.0), 60.0) * 0.16;
      // 格子線
      vec2 g = vUV * uGrid;
      vec2 f = abs(fract(g - 0.5) - 0.5);
      ${deriv ? "f /= fwidth(g); float line = 1.0 - smoothstep(0.55, 1.5, min(f.x, f.y));"
              : "float line = 1.0 - smoothstep(0.03, 0.07, min(f.x, f.y));"}
      // 走る光: u に沿って頭が明るく、後ろへ尾を引く
      float s = fract(uT - vUV.x);
      float band = pow(1.0 - s, 7.0);
      vec3 lc = uA * (0.2 + 0.45 * d) + mix(uB, vec3(1.0), band * band) * band * 1.25;
      col = mix(col, lc, line);
      col += uB * band * 0.07;
      gl_FragColor = vec4(col, 1.0);
    }`;
  const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return s; };
  const prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS));
  gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) { els.forEach((c) => c.classList.add("no-gl")); return; }
  gl.useProgram(prog);
  gl.enable(gl.DEPTH_TEST);
  const loc = {
    pos: gl.getAttribLocation(prog, "aPos"), nor: gl.getAttribLocation(prog, "aNor"),
    R: gl.getUniformLocation(prog, "uR"), aspect: gl.getUniformLocation(prog, "uAspect"),
    A: gl.getUniformLocation(prog, "uA"), B: gl.getUniformLocation(prog, "uB"),
    uv: gl.getAttribLocation(prog, "aUV"), grid: gl.getUniformLocation(prog, "uGrid"), T: gl.getUniformLocation(prog, "uT"),
  };
  gl.enableVertexAttribArray(loc.pos); gl.enableVertexAttribArray(loc.nor); gl.enableVertexAttribArray(loc.uv);

  const meshes = {};
  function mesh(name) {
    if (meshes[name]) return meshes[name];
    const m = build(SURFACES[name] || SURFACES.torus);
    const buf = (target, data) => { const b = gl.createBuffer(); gl.bindBuffer(target, b); gl.bufferData(target, data, gl.STATIC_DRAW); return b; };
    return (meshes[name] = { pos: buf(gl.ARRAY_BUFFER, m.pos), nor: buf(gl.ARRAY_BUFFER, m.nor), uv: buf(gl.ARRAY_BUFFER, m.uv), grid: m.grid, idx: buf(gl.ELEMENT_ARRAY_BUFFER, m.idx), count: m.idx.length });
  }

  const rgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  const items = els.map((el, i) => ({
    el, ctx: el.getContext("2d"), name: el.dataset.shape,
    a: rgb(el.dataset.a || "#e5e5e5"), b: rgb(el.dataset.b || "#ff8833"),
    spin: parseFloat(el.dataset.spin || "0.35"), tilt: parseFloat(el.dataset.tilt || "0.5"),
    roll: parseFloat(el.dataset.roll || "0.2"), phase: i * 1.7, on: false, drawn: false,
  }));

  // 3x3 の回転（行ごと）。渡すときに転置する
  const mul = (a, b) => { const r = []; for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) r.push(a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j]); return r; };
  const rx = (t) => [1, 0, 0, 0, Math.cos(t), -Math.sin(t), 0, Math.sin(t), Math.cos(t)];
  const ry = (t) => [Math.cos(t), 0, Math.sin(t), 0, 1, 0, -Math.sin(t), 0, Math.cos(t)];
  const rz = (t) => [Math.cos(t), -Math.sin(t), 0, Math.sin(t), Math.cos(t), 0, 0, 0, 1];

  let mx = 0, my = 0, tx = 0, ty = 0;
  addEventListener("pointermove", (e) => { tx = e.clientX / innerWidth - 0.5; ty = e.clientY / innerHeight - 0.5; }, { passive: true });

  function draw(it, time) {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const w = Math.round(it.el.clientWidth * dpr), h = Math.round(it.el.clientHeight * dpr);
    if (!w || !h) return;
    if (it.el.width !== w || it.el.height !== h) { it.el.width = w; it.el.height = h; }
    if (glc.width < w || glc.height < h) { glc.width = Math.max(glc.width, w); glc.height = Math.max(glc.height, h); }
    const m = mesh(it.name);
    gl.viewport(0, 0, w, h);
    gl.enable(gl.SCISSOR_TEST); gl.scissor(0, 0, w, h);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    const ang = it.phase + time * it.spin + scrollY * 0.0016;
    const R = mul(rx(it.tilt + my * 0.5), mul(rz(it.roll + mx * 0.35), ry(ang)));
    gl.uniformMatrix3fv(loc.R, false, [R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]]);
    gl.uniform1f(loc.aspect, w / h);
    gl.uniform3fv(loc.A, it.a); gl.uniform3fv(loc.B, it.b);
    gl.bindBuffer(gl.ARRAY_BUFFER, m.pos); gl.vertexAttribPointer(loc.pos, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, m.nor); gl.vertexAttribPointer(loc.nor, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, m.uv); gl.vertexAttribPointer(loc.uv, 2, gl.FLOAT, false, 0, 0);
    gl.uniform2fv(loc.grid, m.grid);
    gl.uniform1f(loc.T, reduce ? 0.35 : time * 0.16 + it.phase * 0.13);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, m.idx);
    gl.drawElements(gl.TRIANGLES, m.count, gl.UNSIGNED_SHORT, 0);
    it.ctx.clearRect(0, 0, w, h);
    it.ctx.drawImage(glc, 0, glc.height - h, w, h, 0, 0, w, h);
    it.drawn = true;
  }

  const io = new IntersectionObserver((es) => es.forEach((e) => {
    const it = items.find((x) => x.el === e.target);
    if (it) it.on = e.isIntersecting;
  }), { rootMargin: "120px" });
  items.forEach((it) => io.observe(it.el));

  function frame(ms) {
    const time = reduce ? 0 : ms / 1000;
    mx += (tx - mx) * 0.06; my += (ty - my) * 0.06;
    for (const it of items) if (it.on && !(reduce && it.drawn)) draw(it, time);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
