// R-1177 F-008 보기 창 — 같은 장면 · 내 캐릭터(마음대로 이동) · 초파리 뇌 캐릭터(따로 · 서 있음) · 카메라 넷.
// 장면 · 자세 · 눈 표본은 /view/<이름>.json(파이썬 view.py 가 만듦). 장면 짓는 순서와 재질 묶기는 렌더러와 같다
// (bind.js · 오른손 · y 위 · 앞 = (sin yaw, 0, cos yaw) · 왼 = (cos yaw, 0, −sin yaw)). 판정용 렌더러는 건드리지 않는다.
import { loadIndex, bindContainer } from './bind.js';

const B = window.BABYLON;
const q = new URLSearchParams(location.search);
const NAME = q.get('scene') ?? 'arena-b';

if (B.KhronosTextureContainer2?.URLConfig) {   // KTX2 해독기는 로컬(render.js 와 같음)
  const u = B.KhronosTextureContainer2.URLConfig, base = '/babylon/ktx2';
  u.jsDecoderModule = `${base}/babylon.ktx2Decoder.js`;
  u.wasmUASTCToASTC = `${base}/ktx2Transcoders/1/uastc_astc.wasm`;
  u.wasmUASTCToBC7 = `${base}/ktx2Transcoders/1/uastc_bc7.wasm`;
  u.wasmUASTCToRGBA_UNORM = `${base}/ktx2Transcoders/1/uastc_rgba8_unorm_v2.wasm`;
  u.wasmUASTCToRGBA_SRGB = `${base}/ktx2Transcoders/1/uastc_rgba8_srgb_v2.wasm`;
  u.wasmUASTCToR8_UNORM = `${base}/ktx2Transcoders/1/uastc_r8_unorm.wasm`;
  u.wasmUASTCToRG8_UNORM = `${base}/ktx2Transcoders/1/uastc_rg8_unorm.wasm`;
  u.jsMSCTranscoder = `${base}/ktx2Transcoders/1/msc_basis_transcoder.js`;
  u.wasmMSCTranscoder = `${base}/ktx2Transcoders/1/msc_basis_transcoder.wasm`;
  u.wasmZSTDDecoder = `${base}/zstddec.wasm`;
}

const $ = (id) => document.getElementById(id);
const fwd = (yaw) => [Math.sin(yaw), 0, Math.cos(yaw)];
const lft = (yaw) => [Math.cos(yaw), 0, -Math.sin(yaw)];
const look = (yaw, pitch) => [Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch)];
const V = (a) => new B.Vector3(a[0], a[1], a[2]);

let engine, scene, cam, eyeCam, rtt, h, canvas;
let mode = 2;                                   // 1 1인칭 · 2 3인칭 · 3 자유 · 4 초파리 눈으로
const me = { pos: [0, 0, 0], yaw: 0, pitch: 0 };
const flyLook = { yaw: 0, pitch: 0 };
const keys = new Set();
const nodes = {};
let meNode, meTag, flyNode, collide = true, drag = null, play = null, eyeBusy = false, lastEye = 0;
const notInEye = new Set();                     // 초파리 눈 그림에서 뺄 것(제 몸 · 이름표 · 고리)

async function loadContainer(url) {
  if (B.LoadAssetContainerAsync) return await B.LoadAssetContainerAsync(url, scene);
  const i = url.lastIndexOf('/');
  return await B.SceneLoader.LoadAssetContainerAsync(url.slice(0, i + 1), url.slice(i + 1), scene);
}

function place(node, pos, yaw, s = 1) {
  node.position.set(pos[0], pos[1], pos[2]);
  node.rotation.set(0, yaw, 0);
  node.scaling.set(s, s, s);
}

function avatar(container, id, rgb) {
  const node = new B.TransformNode(`ava:${id}`, scene);
  const inst = container.instantiateModelsToScene((n) => `${id}:${n}`, false);
  const m = new B.StandardMaterial(`ava:${id}:mat`, scene);
  m.diffuseColor = new B.Color3(rgb[0], rgb[1], rgb[2]);
  m.specularColor = new B.Color3(0.1, 0.1, 0.1);
  for (const r of inst.rootNodes) {
    r.parent = node;
    for (const mesh of r.getChildMeshes(false)) mesh.material = m;
  }
  return node;
}

function label(text, css, parent, y) {
  const W = 512, H = 128;
  const plane = B.MeshBuilder.CreatePlane(`lbl:${text}`, { width: 1.4, height: 0.35 }, scene);
  const dt = new B.DynamicTexture(`lblTex:${text}`, { width: W, height: H }, scene, false);
  dt.hasAlpha = true;
  const ctx = dt.getContext();
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = 'rgba(17,20,24,0.75)';
  ctx.fillRect(40, 14, W - 80, H - 28);
  ctx.font = 'bold 60px -apple-system, "Apple SD Gothic Neo", sans-serif';
  ctx.fillStyle = css;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, W / 2, H / 2 + 2);
  dt.update();
  const mat = new B.StandardMaterial(`lblMat:${text}`, scene);
  mat.diffuseTexture = dt;
  mat.emissiveColor = new B.Color3(1, 1, 1);
  mat.disableLighting = true;
  mat.useAlphaFromDiffuseTexture = true;
  mat.backFaceCulling = false;
  plane.material = mat;
  plane.billboardMode = B.Mesh.BILLBOARDMODE_ALL;
  plane.parent = parent;
  plane.position.y = y;
  return plane;
}

function setMode(m) {
  mode = m;
  for (const [i, id] of [[1, 'm1'], [2, 'm2'], [3, 'm3'], [4, 'mf']]) $(id).classList.toggle('on', i === m);
  if (m === 3) {
    cam.attachControl(canvas, true);
  } else {
    cam.detachControl();
  }
  meNode.setEnabled(m !== 1);                   // 1인칭에서는 내 몸을 숨김(머리 안에 카메라)
  if (meTag) meTag.setEnabled(m === 3 || m === 4);   // 1 · 3인칭에서는 내 이름표가 시야를 가림
  $('mode').textContent = {
    1: '1인칭 — 내 캐릭터 눈높이 · W A S D 로 걷기 · 마우스 끌기로 둘러보기',
    2: '3인칭 — 내 캐릭터 뒤에서 · W A S D 로 걷기 · 마우스 끌기로 방향 돌리기',
    3: '자유 — 카메라만 날아다님(내 캐릭터는 제자리) · W A S D · 마우스 끌기',
    4: '초파리 눈으로 — 초파리 뇌 캐릭터 눈 자리(사람 해상도) · 마우스 끌기로 둘러보기',
  }[m];
  canvas.focus();
}

function stepMe(dt) {
  if (mode !== 1 && mode !== 2) return;
  const run = keys.has('ShiftLeft') || keys.has('ShiftRight');
  const sp = (run ? h.body.run_m_s : h.body.walk_m_s) * dt;
  let mz = 0, mx = 0;
  if (keys.has('KeyW') || keys.has('ArrowUp')) mz += 1;
  if (keys.has('KeyS') || keys.has('ArrowDown')) mz -= 1;
  if (keys.has('KeyA') || keys.has('ArrowLeft')) mx += 1;
  if (keys.has('KeyD') || keys.has('ArrowRight')) mx -= 1;
  if (mz || mx) {
    const n = Math.hypot(mz, mx), f = fwd(me.yaw), l = lft(me.yaw);
    me.pos[0] += ((f[0] * mz + l[0] * mx) / n) * sp;
    me.pos[2] += ((f[2] * mz + l[2] * mx) / n) * sp;
  }
  for (const c of collide ? h.colliders : []) {  // 물체 · 초파리 뇌 캐릭터와 겹치면 밀려남(세계 서버와 같은 원기둥 · C 로 끔)
    const dx = me.pos[0] - c.x, dz = me.pos[2] - c.z, d = Math.hypot(dx, dz), min = c.r + h.body.radius_m;
    if (d < min && d > 1e-6) { me.pos[0] = c.x + (dx / d) * min; me.pos[2] = c.z + (dz / d) * min; }
  }
  const lim = h.ground.size_m / 2 - 1;
  me.pos[0] = Math.max(-lim, Math.min(lim, me.pos[0]));
  me.pos[2] = Math.max(-lim, Math.min(lim, me.pos[2]));
  place(meNode, me.pos, me.yaw);
}

function stepCam() {
  if (mode === 1) {
    const e = [me.pos[0], me.pos[1] + h.body.eye_m, me.pos[2]];
    const d = look(me.yaw, me.pitch);
    cam.position.copyFrom(V(e));
    cam.setTarget(V([e[0] + d[0], e[1] + d[1], e[2] + d[2]]));
  } else if (mode === 2) {
    const t = [me.pos[0], me.pos[1] + h.body.eye_m, me.pos[2]];
    const d = look(me.yaw, me.pitch - 0.25);
    cam.position.copyFrom(V([t[0] - 4 * d[0], t[1] - 4 * d[1] + 0.4, t[2] - 4 * d[2]]));
    cam.setTarget(V(t));
  } else if (mode === 4) {
    const e = h.eye, d = look(h.fly_yaw + flyLook.yaw, flyLook.pitch);
    cam.position.copyFrom(V(e));
    cam.setTarget(V([e[0] + d[0], e[1] + d[1], e[2] + d[2]]));
  }
}

async function updateEye() {                    // 초파리 뇌 캐릭터 두 눈 큐브맵(렌더러와 같은 면 · 같은 밝기 식) → 기둥 밝기
  eyeBusy = true;
  try {
    const N = h.face_px, out = [new Float32Array(6 * N * N), new Float32Array(6 * N * N)];
    rtt.renderList = scene.meshes.filter((m) => !notInEye.has(m));
    for (let e = 0; e < 2; e++) {
      const eye = h.rig[e], pos = V(eye.pos);
      for (let f = 0; f < 6; f++) {
        const fc = eye.faces[f];
        eyeCam.position.copyFrom(pos);
        eyeCam.upVector = V(fc.up);
        eyeCam.setTarget(pos.add(V(fc.forward)));
        eyeCam.getViewMatrix(true);
        rtt.render();
        const px = await rtt.readPixels();
        for (let i = 0; i < N * N; i++) out[e][f * N * N + i] = 0.2 * px[4 * i] + 0.7 * px[4 * i + 1] + 0.1 * px[4 * i + 2];
      }
    }
    const cv = $('eyecv'), ctx = cv.getContext('2d');
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, cv.width, cv.height);
    for (const [e, s] of [[0, 'L'], [1, 'R']]) {
      const S = h.sampler[s], K = S.k;
      for (let c = 0; c < S.az.length; c++) {
        let v = 0;
        for (let j = 0; j < K; j++) v += S.w[c * K + j] * out[e][S.idx[c * K + j]];
        const g = Math.round(Math.min(255, v));
        ctx.fillStyle = `rgb(${g},${g},${g})`;
        ctx.beginPath();
        ctx.arc(((180 - S.az[c]) / 360) * cv.width, ((70 - S.el[c]) / 140) * cv.height, 3.2, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  } finally {
    eyeBusy = false;
  }
}

async function main() {
  h = await (await fetch(`/view/${NAME}.json`, { cache: 'no-store' })).json();
  $('name').textContent = `${NAME} · ${h.version}`;
  canvas = $('c');
  engine = new B.Engine(canvas, true, { stencil: true, preserveDrawingBuffer: true }, true);
  scene = new B.Scene(engine);
  scene.useRightHandedSystem = true;
  const sky = h.sky?.color ?? [0.6, 0.75, 0.9];
  scene.clearColor = new B.Color4(sky[0], sky[1], sky[2], 1);
  cam = new B.UniversalCamera('view', B.Vector3.Zero(), scene);
  cam.minZ = 0.05; cam.maxZ = 1000; cam.fov = 1.1;
  cam.speed = 0.15; cam.angularSensibility = 2500;
  cam.keysUp.push(87); cam.keysDown.push(83); cam.keysLeft.push(65); cam.keysRight.push(68);
  scene.activeCamera = cam;
  eyeCam = new B.UniversalCamera('flyEye', B.Vector3.Zero(), scene);   // 장면 카메라와 따로(렌더러 10-17 고침과 같은 까닭)
  eyeCam.fovMode = B.Camera.FOVMODE_VERTICAL_FIXED;
  eyeCam.fov = Math.PI / 2; eyeCam.minZ = 0.05; eyeCam.maxZ = 1000;
  rtt = new B.RenderTargetTexture('flyEyeRT', { width: h.face_px, height: h.face_px }, scene, false, true,
    B.Constants.TEXTURETYPE_UNSIGNED_BYTE);
  rtt.activeCamera = eyeCam;
  const hemi = new B.HemisphericLight('hemi', new B.Vector3(0, 1, 0), scene);
  hemi.intensity = h.light?.ambient ?? 0.55;
  const sun = new B.DirectionalLight('sun', new B.Vector3(-0.4, -1, -0.3), scene);
  if (h.light?.sun_dir) sun.direction = V(h.light.sun_dir);
  const gs = h.ground?.size_m ?? 60;
  const ground = B.MeshBuilder.CreateGround('ground', { width: gs, height: gs }, scene);
  const gm = new B.StandardMaterial('groundMat', scene);
  const gc = h.ground?.color ?? [0.3, 0.45, 0.2];
  gm.diffuseColor = new B.Color3(gc[0], gc[1], gc[2]);
  gm.specularColor = new B.Color3(0, 0, 0);
  if (h.ground?.texture_url) {
    const tex = new B.Texture(h.ground.texture_url, scene);
    const tile = h.ground.tile_m ?? 1.4;
    tex.uScale = gs / tile; tex.vScale = gs / tile;
    gm.diffuseTexture = tex;
    gm.diffuseColor = new B.Color3(1, 1, 1);
  }
  ground.material = gm;
  const ti = h.texture_index ? await loadIndex(h.texture_index.url) : null;
  const containers = {}, missing = [];
  let bound = 0;
  for (const [key, url] of Object.entries(h.assets)) {
    try { containers[key] = await loadContainer(url); } catch (e) { missing.push(key); console.warn(e); continue; }
    if (ti) bound += bindContainer(B, scene, containers[key], ti, h.texture_index.base, h.texture_index.alpha_cut).bound.length;
  }
  for (const p of h.poses) {
    const c = containers[p.asset];
    if (!c) continue;
    const node = new B.TransformNode(`ent:${p.id}`, scene);
    const inst = c.instantiateModelsToScene((n) => `${p.id}:${n}`, false);
    for (const r of inst.rootNodes) r.parent = node;
    nodes[p.id] = node;
    place(node, p.pos, p.yaw, p.scale ?? 1);
  }
  const body = containers[h.avatar_asset];
  // 초파리 뇌 캐릭터(노랑 · 서 있음) — 눈 그림에서 제 몸은 뺀다
  flyNode = avatar(body, 'flybrain', [0.95, 0.75, 0.25]);
  place(flyNode, h.fly_pos, h.fly_yaw);
  const tag = label('초파리 뇌', '#f2c14e', flyNode, 2.15);
  const ring = B.MeshBuilder.CreateTorus('flyRing', { diameter: 2 * h.fly_radius, thickness: 0.04, tessellation: 48 }, scene);
  ring.position.set(h.fly_pos[0], 0.03, h.fly_pos[2]);
  const rm = new B.StandardMaterial('ringMat', scene);
  rm.emissiveColor = new B.Color3(0.95, 0.76, 0.3); rm.disableLighting = true;
  ring.material = rm;
  for (const m of [...flyNode.getChildMeshes(false), tag, ring]) notInEye.add(m);
  // 내 캐릭터(파랑) — 초파리 눈에 보인다
  meNode = avatar(body, 'me', [0.35, 0.55, 0.95]);
  meTag = label('나', '#6fa0ff', meNode, 2.15);
  me.pos = [...h.me_start.pos]; me.yaw = h.me_start.yaw;
  place(meNode, me.pos, me.yaw);

  $('m1').onclick = () => setMode(1);
  $('m2').onclick = () => setMode(2);
  $('m3').onclick = () => setMode(3);
  $('mf').onclick = () => setMode(4);
  $('appL').onclick = () => { play = { traj: h.approach.left, t0: performance.now() }; };
  $('appR').onclick = () => { play = { traj: h.approach.right, t0: performance.now() }; };
  const home = () => { me.pos = [...h.me_start.pos]; me.yaw = h.me_start.yaw; me.pitch = 0; place(meNode, me.pos, me.yaw); };
  const toggleCollide = () => { collide = !collide; $('col').textContent = `부딪힘 ${collide ? '켬' : '끔'} (C)`; };
  $('home').onclick = home;
  $('col').onclick = toggleCollide;
  window.addEventListener('keydown', (e) => {
    if (['Digit1', 'Digit2', 'Digit3', 'Digit4'].includes(e.code)) setMode(Number(e.code.slice(-1)));
    if (e.code === 'KeyR') home();
    if (e.code === 'KeyC') toggleCollide();
    keys.add(e.code);
  });
  window.addEventListener('keyup', (e) => keys.delete(e.code));
  window.addEventListener('blur', () => keys.clear());
  canvas.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY }; canvas.focus(); });
  window.addEventListener('pointerup', () => { drag = null; });
  window.addEventListener('pointermove', (e) => {
    if (!drag || mode === 3) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag = { x: e.clientX, y: e.clientY };
    const t = mode === 4 ? flyLook : me;
    t.yaw -= dx * 0.005;                        // 오른쪽으로 끌면 오른쪽으로 돎(yaw 증가 = 왼쪽)
    t.pitch = Math.max(-1.2, Math.min(1.2, t.pitch - dy * 0.004));
  });
  setMode(2);
  await scene.whenReadyAsync();
  $('status').textContent = `에셋 ${Object.keys(containers).length} 실림 · 빠짐 ${missing.length} · 재질 묶음 ${bound} · 물체 ${h.poses.length}`;
  engine.runRenderLoop(() => {
    const dt = Math.min(0.05, engine.getDeltaTime() / 1000);
    stepMe(dt);
    if (play) {                                 // 다가옴 시험 대본(10 ms 걸음)을 실제 시간으로
      const k = Math.floor((performance.now() - play.t0) / h.approach.dt_ms);
      const tr = play.traj;
      const p = tr[Math.min(k, tr.length - 1)];
      place(nodes.person, p.pos, p.yaw);
      if (k >= tr.length + 100) { place(nodes.person, h.person_rest.pos, h.person_rest.yaw); play = null; }
    }
    stepCam();
    scene.render();
    const now = performance.now();
    if (!eyeBusy && now - lastEye > h.eye_every_ms) { lastEye = now; updateEye(); }
  });
  window.addEventListener('resize', () => engine.resize());
}

main().catch((e) => { $('status').textContent = `오류: ${e}`; console.error(e); });
