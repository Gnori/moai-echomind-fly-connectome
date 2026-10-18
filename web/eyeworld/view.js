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
let meNode, meTag, flyNode, ring, collide = true, drag = null, play = null, eyeBusy = false, lastEye = 0;
const fly = { pos: [0, 0, 0], yaw: 0 };         // 초파리 뇌 캐릭터 자세(뇌 연결이 켜지면 뇌 명령으로 움직임)
let brain = null, dial = 1, tick = 0, hop = 0, flyRigBones = null, lastParts = null, lastSpeed = 0;
const notInEye = new Set();
let LIN = null;                                 // sRGB 값(0~255) → 선형(0~1) 표 — eye_linear 가 거짓이면 그대로(÷255)
function makeLin(linear) {
  LIN = new Float32Array(256);
  for (let v = 0; v < 256; v++) { const c = v / 255; LIN[v] = linear ? (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4) : c; }
}
const markers = [];                             // 찾기 표지(고리 · 기둥 · 이름표) — 초파리 눈 시점에서는 숨김                     // 초파리 눈 그림에서 뺄 것(제 몸 · 이름표 · 고리)

async function loadContainer(url) {
  if (B.LoadAssetContainerAsync) return await B.LoadAssetContainerAsync(url, scene);
  const i = url.lastIndexOf('/');
  return await B.SceneLoader.LoadAssetContainerAsync(url.slice(0, i + 1), url.slice(i + 1), scene);
}

const enc = new TextEncoder(), dec = new TextDecoder();
function pack(head, body) {
  const hb = enc.encode(JSON.stringify(head));
  const n = body ? body.byteLength : 0;
  const buf = new ArrayBuffer(4 + hb.length + n);
  new DataView(buf).setUint32(0, hb.length, true);
  new Uint8Array(buf, 4, hb.length).set(hb);
  if (body) new Uint8Array(buf, 4 + hb.length).set(body);
  return buf;
}
function unpack(buf) {
  const n = new DataView(buf).getUint32(0, true);
  return JSON.parse(dec.decode(new Uint8Array(buf, 4, n)));
}

function flyRig() {                             // 세계 서버 eye_rig 와 같은 식(앞 · 뒤 · 왼 · 오 · 위 · 아래 · 왼눈 = 머리 + 왼 × 간격/2)
  const f = fwd(fly.yaw), l = lft(fly.yaw), U = [0, 1, 0], neg = (a) => a.map((x) => -x);
  const fb = h.fly_body;                         // 10-18 초파리 크기 몸(눈 높이 · 눈 사이 — config view_brain.fly_body)
  const head = [fly.pos[0], fb.eye_m, fly.pos[2]], k = fb.eye_sep_m / 2;   // 뛰어오름은 보이게만 — 눈은 땅 기준 그대로
  const faces = [[f, U], [neg(f), U], [l, U], [neg(l), U], [U, neg(f)], [neg(U), f]]
    .map(([forward, up]) => ({ forward, up }));
  return [
    { pos: [head[0] + l[0] * k, head[1], head[2] + l[2] * k], faces },
    { pos: [head[0] - l[0] * k, head[1], head[2] - l[2] * k], faces },
  ];
}

function pushOut(p, r, others) {                // 원기둥끼리 겹치면 p 만 밀려남(세계 서버와 같은 식)
  for (const c of others) {
    const dx = p[0] - c.x, dz = p[2] - c.z, d = Math.hypot(dx, dz), min = c.r + r;
    if (d < min && d > 1e-6) { p[0] = c.x + (dx / d) * min; p[2] = c.z + (dz / d) * min; }
  }
  const lim = h.ground.size_m / 2 - 1;
  p[0] = Math.max(-lim, Math.min(lim, p[0]));
  p[2] = Math.max(-lim, Math.min(lim, p[2]));
}

function contacts() {                         // 초파리 뇌 캐릭터 몸 원기둥에 닿은 것 → 몸 기준 방위(+ 왼쪽) · 틈(뇌 서버가 부위 × 좌우로)
  const f = fwd(fly.yaw), l = lft(fly.yaw), out = [];
  const others = [...h.colliders.map((c) => ({ ...c, what: 'prop' })),
    { x: me.pos[0], z: me.pos[2], r: h.body.radius_m, what: 'me' }];
  for (const c of others) {
    const dx = c.x - fly.pos[0], dz = c.z - fly.pos[2];
    const gap = Math.hypot(dx, dz) - c.r - h.fly_radius;
    if (gap > h.fly_body.touch_margin_m) continue;
    const b = (Math.atan2(dx * l[0] + dz * l[2], dx * f[0] + dz * f[2]) * 180) / Math.PI;
    out.push({ bearing_deg: b, gap_m: gap, what: c.what });
  }
  return out;
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
  node.metadata = { skeleton: inst.skeletons?.[0] ?? null };   // 뼈대(몸 부위 대응 · R-5402)
  return node;
}

function flyBody(fb, mk) {                      // 초파리 크기 몸(가슴 · 배 · 머리 · 빨간 겹눈 둘) + 표지 기둥 — 앞 = +z(yaw 0)
  const node = new B.TransformNode('ava:flybrain', scene);
  const L = 2 * fb.radius_m, y = fb.eye_m * 0.75;
  const mat = (name, rgb, glow = 0) => {
    const m = new B.StandardMaterial(name, scene);
    m.diffuseColor = new B.Color3(...rgb); m.specularColor = new B.Color3(0.15, 0.15, 0.15);
    if (glow) m.emissiveColor = new B.Color3(...rgb.map((c) => c * glow));
    return m;
  };
  const part = (name, dims, pos, m) => {
    const p = B.MeshBuilder.CreateSphere(name, { diameterX: dims[0], diameterY: dims[1], diameterZ: dims[2], segments: 12 }, scene);
    p.position.set(...pos); p.material = m; p.parent = node; return p;
  };
  const body = mat('flyBodyMat', [0.55, 0.42, 0.25]);
  part('flyThorax', [0.42 * L, 0.36 * L, 0.38 * L], [0, y, -0.3 * L], body);   // 머리(눈)가 몸 자리 가운데 — 눈 자리(flyRig)와 맞춤
  part('flyAbdomen', [0.40 * L, 0.32 * L, 0.55 * L], [0, y * 0.9, -0.72 * L], mat('flyAbdMat', [0.35, 0.27, 0.17]));
  part('flyHead', [0.34 * L, 0.28 * L, 0.24 * L], [0, fb.eye_m, 0], body);
  const eye = mat('flyEyeMat', [0.75, 0.08, 0.06], 0.25);
  for (const sx of [1, -1]) part('flyEye', [0.14 * L, 0.24 * L, 0.18 * L], [sx * fb.eye_sep_m / 2, fb.eye_m, 0.02 * L], eye);
  const pole = B.MeshBuilder.CreateCylinder('flyPole', { height: mk.pole_m, diameter: 0.006, tessellation: 8 }, scene);
  pole.position.set(0, mk.pole_m / 2 + 0.01, 0); pole.parent = node; markers.push(pole);
  const pm = mat('flyPoleMat', [0.95, 0.76, 0.3], 1); pm.alpha = 0.55; pole.material = pm;
  return node;
}

// 몸 부위 대응(R-5402) — 뼈를 캐릭터 몸 기준 축(x 왼 · y 위 · z 앞)으로 돌린다. 쉬는 자세 회전 위에 덧붙임(보이게만 · 각도는 config view.pose).
function makeRig(node) {
  const sk = node.metadata?.skeleton;
  if (!sk) return null;
  node.computeWorldMatrix(true);
  const rootInv = B.Quaternion.FromRotationMatrix(node.getWorldMatrix().getRotationMatrix()).invert();
  const bones = {};
  for (const name of ['upperleg01.L', 'upperleg01.R', 'lowerleg01.L', 'lowerleg01.R', 'upperarm01.L', 'upperarm01.R',
    'lowerarm01.L', 'lowerarm01.R', 'neck01', 'head', 'jaw']) {
    const t = sk.bones.find((b) => b.name === name)?.getTransformNode();
    if (!t || !t.rotationQuaternion) continue;
    t.parent.computeWorldMatrix(true);
    const pw = B.Quaternion.FromRotationMatrix(t.parent.getWorldMatrix().getRotationMatrix());
    const toParent = rootInv.multiply(pw).invert();          // 몸 기준 축 → 부모 쪽 축
    bones[name] = { t, rest: t.rotationQuaternion.clone(), toParent };
  }
  const kid = { 'upperarm01.L': 'lowerarm01.L', 'upperarm01.R': 'lowerarm01.R', 'lowerarm01.L': 'wrist.L', 'lowerarm01.R': 'wrist.R' };
  for (const [name, child] of Object.entries(kid)) {   // 뼈 방향(쉬는 자세 · 부모 쪽) — 팔은 '가리킬 방향'으로 돌린다
    const b = bones[name], c = sk.bones.find((x) => x.name === child)?.getTransformNode();
    if (!b || !c) continue;
    c.computeWorldMatrix(true);
    const d = c.getAbsolutePosition().subtract(b.t.getAbsolutePosition());
    const m = new B.Matrix(); b.toParent.multiply(rootInv).toRotationMatrix(m);
    b.aP = B.Vector3.TransformNormal(d, m).normalize();
  }
  return { node, bones, cur: {}, phase: 0 };
}

function axisIn(b, ax) {                         // 몸 기준 축 ax([x, y, z]) 를 그 뼈의 부모 쪽 축으로
  const v = new B.Vector3(...ax);
  const m = new B.Matrix(); b.toParent.toRotationMatrix(m);
  return B.Vector3.TransformNormal(v, m).normalize();
}

function poseBone(rig, name, turns) {            // turns = [[축, 각도(도)] …] — 쉬는 자세 위에 차례로
  const b = rig.bones[name];
  if (!b) return;
  let q = b.rest.clone();
  for (const [ax, deg] of turns) if (deg) q = B.Quaternion.RotationAxis(axisIn(b, ax), (deg * Math.PI) / 180).multiply(q);
  b.t.rotationQuaternion = q;
}

function aim(rig, name, d) {                    // 그 뼈가 몸 기준 방향 d 를 가리키게(지금 부모 자세 기준)
  const b = rig.bones[name];
  if (!b || !b.aP) return;
  rig.node.computeWorldMatrix(true); b.t.parent.computeWorldMatrix(true);
  const root = B.Quaternion.FromRotationMatrix(rig.node.getWorldMatrix().getRotationMatrix());
  const par = B.Quaternion.FromRotationMatrix(b.t.parent.getWorldMatrix().getRotationMatrix());
  const m = new B.Matrix(); root.invert().multiply(par).invert().toRotationMatrix(m);
  const dp = B.Vector3.TransformNormal(new B.Vector3(...d), m).normalize();
  const delta = new B.Quaternion(); B.Quaternion.FromUnitVectorsToRef(b.aP, dp, delta);
  b.t.rotationQuaternion = delta.multiply(b.rest);
  b.t.computeWorldMatrix(true);
}

function mix(list) {                             // [[가중, [x, y, z]] …] → 단위 방향
  const v = [0, 0, 0];
  for (const [w, d] of list) for (let i = 0; i < 3; i++) v[i] += w * d[i];
  const n = Math.hypot(...v) || 1;
  return v.map((x) => x / n);
}

function stepPose(rig, parts, speed, dt) {       // 부위 값(뇌 연결 · 0~1) → 뼈 각도
  const P = h.pose, k = Math.min(1, (dt * 1000) / P.smooth_ms), c = rig.cur;
  const goal = { walk: Math.min(1, Math.abs(speed) / Math.max(1e-6, h.fly_body.fwd_max_m_s)), dir: Math.sign(speed),
    head: parts?.head_yaw_deg ?? 0, rub: parts?.hands_rub ?? 0, face: parts?.face_wipe ?? 0,
    earL: parts?.ear_L ?? 0, earR: parts?.ear_R ?? 0, jaw: parts?.mouth_open ?? 0 };
  for (const [key, v] of Object.entries(goal)) c[key] = (c[key] ?? 0) + (v - (c[key] ?? 0)) * k;
  rig.phase += Math.abs(speed) * P.walk_steps_per_m * dt * 2 * Math.PI;
  const t = performance.now() / 1000, wig = Math.sin(2 * Math.PI * P.wiggle_hz * t);
  const X = [1, 0, 0], Y = [0, 1, 0], Z = [0, 0, 1];
  for (const [s, sg] of [['L', 1], ['R', -1]]) {   // 다리: 걸음(앞으로 = −x 축 회전)
    const sw = Math.sin(rig.phase + (s === 'L' ? 0 : Math.PI)) * c.walk;
    poseBone(rig, `upperleg01.${s}`, [[X, -P.walk_hip_deg * sw * (c.dir || 1)]]);
    poseBone(rig, `lowerleg01.${s}`, [[X, P.walk_knee_deg * Math.max(0, -Math.cos(rig.phase + (s === 'L' ? 0 : Math.PI))) * c.walk]]);
  }
  for (const [s, sg] of [['L', 1], ['R', -1]]) {   // 팔: 쉬는 자세(내림 · 동작 아님) + 비비기 · 얼굴 닦기 · 같은 쪽 귀 — 가리킬 방향을 섞음
    const ear = s === 'L' ? c.earL : c.earR, rest = Math.max(0, 1 - Math.min(1, c.rub + c.face + ear));
    aim(rig, `upperarm01.${s}`, mix([[rest, [sg * 0.3, -0.95, 0.05]], [c.rub, [sg * 0.35, -0.55, 0.75]],
      [c.face, [sg * 0.15, -0.5, 0.85]], [ear, [sg * 0.8, -0.25, 0.55]]]));
    aim(rig, `lowerarm01.${s}`, mix([[rest, [sg * 0.25, -0.95, 0.15]], [c.rub, [-sg * 0.75, 0.12 + 0.15 * wig, 0.65]],
      [c.face, [-sg * (0.35 + 0.12 * wig), 0.85, -0.38]], [ear, [-sg * 0.68, 0.7, -0.2 + 0.12 * wig]]]));
  }
  poseBone(rig, 'neck01', [[Y, c.head * 0.5]]);
  poseBone(rig, 'head', [[Y, c.head * 0.5]]);
  poseBone(rig, 'jaw', [[X, P.jaw_open_deg * c.jaw]]);
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
  $('cam').value = String(m);
  cam.minZ = m === 4 ? Math.min(0.05, 2 * h.fly_body.near_m) : 0.05;   // 초파리 크기 몸이면 가까운 면도 작게
  for (const mk of markers) mk.setEnabled(m !== 4);
  if (m === 3) {
    cam.attachControl(canvas, true);
  } else {
    cam.detachControl();
  }
  meNode.setEnabled(m !== 1);                   // 1인칭에서는 내 몸을 숨김(머리 안에 카메라)
  if (meTag) meTag.setEnabled(m === 3 || m === 4);   // 1 · 3인칭에서는 내 이름표가 시야를 가림
  $('mode').textContent = {
    1: '내 눈높이에서 걷기',
    2: '내 뒤에서 따라가기',
    3: '카메라만 날기(나는 제자리)',
    4: '초파리 눈 자리 · 사람 해상도',
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
  // 물체 · 초파리 뇌 캐릭터와 겹치면 밀려남(세계 서버와 같은 원기둥 · C 로 물체 부딪힘 끔)
  pushOut(me.pos, h.body.radius_m, [...(collide ? h.colliders : []), { x: fly.pos[0], z: fly.pos[2], r: h.fly_radius }]);
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
    const r = flyRig(), e = [(r[0].pos[0] + r[1].pos[0]) / 2, r[0].pos[1], (r[0].pos[2] + r[1].pos[2]) / 2];
    const d = look(fly.yaw + flyLook.yaw, flyLook.pitch);
    cam.position.copyFrom(V(e));
    cam.setTarget(V([e[0] + d[0], e[1] + d[1], e[2] + d[2]]));
  }
}

async function grabEyes() {                     // 초파리 뇌 캐릭터 두 눈 큐브맵(렌더러와 같은 면 · 같은 밝기 식) → u8 · 패널
  const N = h.face_px, img = new Uint8Array(2 * 6 * N * N);
  rtt.renderList = scene.meshes.filter((m) => !notInEye.has(m));
  const rig = flyRig();
  for (let e = 0; e < 2; e++) {
    const pos = V(rig[e].pos);
    for (let f = 0; f < 6; f++) {
      const fc = rig[e].faces[f];
      eyeCam.position.copyFrom(pos);
      eyeCam.upVector = V(fc.up);
      eyeCam.setTarget(pos.add(V(fc.forward)));
      eyeCam.getViewMatrix(true);
      rtt.render();
      const px = await rtt.readPixels();
      const o = (e * 6 + f) * N * N;
      const [wr, wg, wb] = h.eye_rgb, L = LIN;  // 밝기 = 광수용체 감도 무게 × 선형 RGB(config view.eye_rgb · eye_linear)
      for (let i = 0; i < N * N; i++) img[o + i] = Math.round(255 * (wr * L[px[4 * i]] + wg * L[px[4 * i + 1]] + wb * L[px[4 * i + 2]]));
    }
  }
  const cv = $('eyecv'), ctx = cv.getContext('2d');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, cv.width, cv.height);
  for (const [e, s] of [[0, 'L'], [1, 'R']]) {
    const S = h.sampler[s], K = S.k, o = e * 6 * N * N;
    for (let c = 0; c < S.az.length; c++) {
      let v = 0;
      for (let j = 0; j < K; j++) v += S.w[c * K + j] * img[o + S.idx[c * K + j]];
      const g = Math.round(Math.min(255, v));
      ctx.fillStyle = `rgb(${g},${g},${g})`;
      ctx.beginPath();
      ctx.arc(((180 - S.az[c]) / 360) * cv.width, ((70 - S.el[c]) / 140) * cv.height, 3.2, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
  return img;
}

async function updateEye() {
  eyeBusy = true;
  try { await grabEyes(); } finally { eyeBusy = false; }
}

const PART = { 머리털: '앞', 앞다리: '앞', 가운데다리: '옆', '날개·앞등': '옆', 뒷다리: '뒤', 뒷등: '뒤' };
function touchWhere(on) {                       // ['머리털_R', '앞다리_R'] → '오른 앞'
  if (!on.length) return '—';
  const sides = new Set(on.map((k) => k.slice(-1))), where = new Set(on.map((k) => PART[k.slice(0, -2)] ?? '?'));
  const sd = sides.size > 1 ? '양쪽' : sides.has('L') ? '왼' : '오른';
  return `${sd} ${[...where].join('·')}`;
}

const GROUP = { body: '몸을 움직이는 뉴런', parts: '고개 · 손 · 입 뉴런', eye: '눈 · 도망 뉴런', touch: '닿음', top: '지금 많이 켜진 종류' };
const CHECK = { 원문: ['원문', 'ok'], 기억: ['기억', 'mem'], 잼: ['잼', 'meas'], 주석: ['분류', 'ann'] };

function nameCell(d) {                          // 실제 이름 + 쉬운 이름 · 마우스를 올리면 설명 · 근거 · 확인 수준
  const [ck, cls] = CHECK[d.check] ?? [d.check || '', 'ann'];
  const tip = `${d.type} — ${d.name}${d.note ? `\n${d.note}` : ''}\n근거: ${d.source || '—'} · 확인: ${ck}`;
  return `<td title="${tip.replaceAll('"', '&quot;')}"><b>${d.type || '(종류 이름 없음)'}</b><span class="ez">${d.name}</span><i class="ck ${cls}">${ck}</i></td>`;
}

function rowsFor(g, out) {                      // [이름 칸, 진짜, 섞은, 켜짐]
  const R = out.sums.real, C = out.sums[h.brain.brains[1]] ?? {};
  const lr = (S, t) => (S[`${t}_L`] === undefined ? ['—', 0] : [`${S[`${t}_L`]} / ${S[`${t}_R`]}`, S[`${t}_L`] + S[`${t}_R`]]);
  const plain = (name, note, a, b, lit) => [`<td><b>${name}</b><span class="ez">${note}</span></td>`, a, b, lit];
  if (g === 'top') {
    if (!out.top.length) return [plain('없음', '입력으로 켠 뉴런 말고는 조용함', '', '', false)];
    return out.top.map((d) => [nameCell(d), `${d.L} / ${d.R}`, `${d.cL} / ${d.cR}`, d.L + d.R > 0]);
  }
  const rows = (h.brain.neurons[g] ?? []).map((d) => {
    const [a, n] = lr(R, d.type);
    return [nameCell(d), a, lr(C, d.type)[0], n > 0];
  });
  if (g === 'eye') rows.unshift(plain('눈 입력', '앞→뒤 움직임 평균 Hz', `${out.input_hz.L.toFixed(1)} / ${out.input_hz.R.toFixed(1)}`, '같음', false));
  if (g === 'touch') {
    rows.unshift(plain('켠 촉각 털', touchWhere(out.touch.on), out.touch.on.length ? `${out.touch.neurons}개` : '없음', '같음', out.touch.on.length > 0));
    rows.push(plain('하강 뉴런 전체', '발화 합', String(out.dn_all.real), String(out.dn_all[h.brain.brains[1]] ?? '—'), false));
  }
  return rows;
}

let lastTab = '', lastOpts = '';                 // 10-18 요구자 "선택시 깜빡임이 심하고 선택도 잘 안됨" — 바뀔 때만 다시 쓴다

function showBrain(out) {
  const sel = $('bview'), want = sel.value;
  const order = want === 'all' ? ['body', 'parts', 'eye', 'touch', 'top'] : [want];
  let html = `<tr><th>최근 ${h.brain.window_ms} ms · 왼 / 오른</th><th>진짜 배선</th><th>섞은 배선</th></tr>`;
  for (const g of order) {
    if (want === 'all') html += `<tr class="grp"><td colspan="3">${GROUP[g]}</td></tr>`;
    for (const [cell, a, b, lit] of rowsFor(g, out)) html += `<tr class="${lit ? 'lit' : ''}">${cell}<td>${a}</td><td>${b}</td></tr>`;
  }
  if (html !== lastTab) { $('btab').innerHTML = html; lastTab = html; }   // 같으면 그대로(마우스 설명 · 깜빡임)
  if (document.activeElement !== sel) {         // 고르기 상자를 열어 둔 동안에는 글을 바꾸지 않는다(닫히고 다시 열리며 깜빡임)
    const texts = [...sel.options].map((o) => (o.value === 'all' ? '전체'
      : `${GROUP[o.value]}${(() => { const n = rowsFor(o.value, out).filter((r) => r[3]).length; return n ? ` · 켜짐 ${n}` : ''; })()}`));
    const key = texts.join('|');
    if (key !== lastOpts) { [...sel.options].forEach((o, i) => { if (o.textContent !== texts[i]) o.textContent = texts[i]; }); lastOpts = key; }
  }
  const c = out.cmd, t = c.turn_deg_s;
  $('tTurn').innerHTML = `${t === 0 ? '' : t > 0 ? '← ' : '→ '}${Math.abs(t).toFixed(0)}<small>°/s</small>`;
  $('tTurnBox').classList.toggle('hot', t !== 0);
  $('tSpeed').innerHTML = c.jump ? '뛰어오름!' : `${c.speed_m_s.toFixed(1)}<small>m/s</small>`;
  $('tSpeedBox').classList.toggle('hot', c.jump || c.speed_m_s !== 0);
  $('tTouch').textContent = touchWhere(out.touch.on);
  $('tTouchBox').classList.toggle('hot', out.touch.on.length > 0);
  $('bspeed').textContent = `뇌 시간 ${(out.sim_ms / 1000).toFixed(1)} s · 실제의 ${(out.sim_ms / 1000 / Math.max(out.wall_s, 1e-3)).toFixed(2)} 배`
    + ` · 한 걸음 ${out.step_ms.toFixed(0)} ms(눈 그림 ${out.eyes_ms.toFixed(0)} · 앞단 ${out.ms.frontend} · 뇌 ${out.ms.brains})`;
}

function setLive(on) {
  $('blive').classList.toggle('on', on);
  $('blivet').textContent = on ? '연결됨' : '꺼짐';
  $('bon').classList.toggle('on', on);
  $('bon').textContent = on ? '뇌 연결 끄기' : '뇌 연결 켜기';
}

async function brainLoop() {                    // 눈 그림 → 뇌 서버(앞단 · 뇌 둘) → 진짜 배선 뇌 명령으로 한 걸음(10 ms)
  const dt = h.brain.dt_ms / 1000;
  while (brain?.on) {
    // 내 캐릭터는 실제 시간(10-18 요구자 "사용자 케릭터 움직임 너무 느려") · 시험용 사람만 뇌 시간
    if (play) stepPlay(1);
    const t0 = performance.now();
    const img = await grabEyes();
    const t1 = performance.now();
    const touch = contacts();
    brain.ws.send(pack({ type: 'step', tick: tick++, dial, contacts: touch, baseline: $('bbase').checked }, img));
    const out = await new Promise((res) => { brain.wait = res; });
    if (out) out.eyes_ms = t1 - t0;
    if (out) out.step_ms = performance.now() - t0;
    if (!out || out.type !== 'out') break;
    const c = out.cmd;
    if (c.jump && hop <= 0) hop = h.brain.motor.jump.hop_ms;
    fly.yaw += ((c.turn_deg_s * Math.PI) / 180) * dt;
    const f = fwd(fly.yaw);
    fly.pos[0] += f[0] * c.speed_m_s * dt;
    fly.pos[2] += f[2] * c.speed_m_s * dt;
    pushOut(fly.pos, h.fly_radius, [...h.colliders, { x: me.pos[0], z: me.pos[2], r: h.body.radius_m }]);
    hop = Math.max(0, hop - h.brain.dt_ms);   // 뛰어오름 높이(보이게만 · 뇌 시간 기준)
    const T = h.brain.motor.jump.hop_ms;
    fly.pos[1] = hop > 0 ? h.fly_body.hop_m * Math.sin((Math.PI * (T - hop)) / T) : 0;
    lastParts = out.parts; lastSpeed = c.speed_m_s;
    showBrain(out);
  }
}

function stepPlay(steps) {                      // 다가옴 시험 대본(10 ms 걸음) — steps 걸음만큼
  play.k = (play.k ?? 0) + steps;
  const tr = play.traj, k = Math.floor(play.k);
  const p = tr[Math.min(k, tr.length - 1)];
  place(nodes.person, p.pos, p.yaw);
  if (k >= tr.length + 100) { place(nodes.person, h.person_rest.pos, h.person_rest.yaw); play = null; }
}

function flyHome() {
  fly.pos = [...h.fly_pos]; fly.yaw = h.fly_yaw;
  if (brain?.ws?.readyState === 1) brain.ws.send(pack({ type: 'reset' }));
}

function setupBrain() {
  if (!h.brain) return;
  $('nobrain').style.display = 'none';
  $('brainbox').style.display = '';
  const dl = $('dial');
  dl.max = h.brain.dial.max; dl.value = h.brain.dial.default; dial = Number(dl.value);
  const showDial = () => {
    $('dialv').textContent = dial === 1 ? '×1 잰 그대로' : `×${dial} 손으로 올림`;
    $('dialv').classList.toggle('hand', dial !== 1);
  };
  showDial();
  dl.oninput = () => { dial = Number(dl.value); showDial(); };
  dl.onchange = () => canvas.focus();
  $('breset').onclick = flyHome;
  $('bon').onclick = () => {
    if (brain?.on) { brain.on = false; brain.ws.close(); brain = null; setLive(false); return; }
    const ws = new WebSocket(`ws://127.0.0.1:${h.brain.port}`);
    ws.binaryType = 'arraybuffer';
    brain = { ws, on: false, wait: null };
    ws.onmessage = (ev) => { const m = unpack(ev.data); if (m.type === 'reset.ok') return; brain?.wait?.(m); };
    ws.onclose = () => { brain?.wait?.(null); setLive(false); };
  $('bview').onchange = () => { lastTab = ''; canvas.focus(); };
  $('bbase').onchange = () => canvas.focus();
    ws.onopen = () => {                         // 켤 때마다 뇌 · 앞단을 처음 상태로(앞 연결의 흔적 없이)
      ws.send(pack({ type: 'reset' }));
      brain.on = true; setLive(true); brainLoop();
    };
  };
}

async function main() {
  h = await (await fetch(`/view/${NAME}.json`, { cache: 'no-store' })).json();
  $('name').textContent = `${NAME} · ${h.version}`;
  makeLin(h.eye_linear);
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
  eyeCam.fov = Math.PI / 2; eyeCam.minZ = h.fly_body.near_m; eyeCam.maxZ = 300;
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
  // 초파리 뇌 캐릭터 — 기본은 사람 몸(노랑 · 요구자 10-18 "초파리 뇌를 단 휴먼 케릭터") · char_body: fly 면 초파리 크기 몸 + 찾기 표지 · 눈 그림에서는 모두 뺀다
  const small = h.fly_body.kind === 'fly';
  flyNode = small ? flyBody(h.fly_body, h.view_marker) : avatar(body, 'flybrain', [0.95, 0.75, 0.25]);
  fly.pos = [...h.fly_pos]; fly.yaw = h.fly_yaw;
  place(flyNode, fly.pos, fly.yaw);
  const tag = label('초파리 뇌', '#f2c14e', flyNode, small ? h.view_marker.pole_m + 0.2 : 2.15);
  const ringD = small ? h.view_marker.ring_m : 2 * h.fly_radius;
  ring = B.MeshBuilder.CreateTorus('flyRing', { diameter: ringD, thickness: small ? 0.015 : 0.04, tessellation: 64 }, scene);
  ring.position.set(h.fly_pos[0], small ? 0.002 : 0.03, h.fly_pos[2]);
  const rm = new B.StandardMaterial('ringMat', scene);
  rm.emissiveColor = new B.Color3(0.95, 0.76, 0.3); rm.disableLighting = true;
  ring.material = rm;
  if (small) markers.push(ring, tag);   // 초파리 크기면 초파리 눈 시점에서 표지가 시야를 가림
  for (const m of [...flyNode.getChildMeshes(false), tag, ring]) notInEye.add(m);
  // 내 캐릭터(파랑) — 초파리 눈에 보인다
  meNode = avatar(body, 'me', [0.35, 0.55, 0.95]);
  meTag = label('나', '#6fa0ff', meNode, 2.15);
  me.pos = [...h.me_start.pos]; me.yaw = h.me_start.yaw;
  place(meNode, me.pos, me.yaw);

  $('cam').onchange = () => setMode(Number($('cam').value));
  $('appPlay').onclick = () => { play = { traj: h.approach[$('appSide').value], k: 0 }; canvas.focus(); };
  const home = () => { me.pos = [...h.me_start.pos]; me.yaw = h.me_start.yaw; me.pitch = 0; place(meNode, me.pos, me.yaw); };
  const setCollide = (on) => { collide = on; $('col').checked = on; };
  $('home').onclick = () => { home(); canvas.focus(); };
  $('col').onchange = () => { setCollide($('col').checked); canvas.focus(); };
  window.addEventListener('keydown', (e) => {
    if (e.target instanceof HTMLSelectElement || e.target instanceof HTMLInputElement) return;   // 고르기 상자 · 다이얼 조작 중에는 몸 키를 받지 않음
    if (['Digit1', 'Digit2', 'Digit3', 'Digit4'].includes(e.code)) setMode(Number(e.code.slice(-1)));
    if (e.code === 'KeyR') home();
    if (e.code === 'KeyC') setCollide(!collide);
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
  flyRigBones = small ? null : makeRig(flyNode);
  if (q.get('debug')) { window.__B = B; window.__fly = flyNode; window.__setMode = setMode; window.__cam = cam; window.__pose = (parts, speed = 0) => { lastParts = parts; lastSpeed = speed; window.__force = true; }; }
  setMode(2);
  setupBrain();
  await scene.whenReadyAsync();
  $('status').textContent = `에셋 ${Object.keys(containers).length} 실림 · 빠짐 ${missing.length} · 재질 묶음 ${bound} · 물체 ${h.poses.length}`;
  engine.runRenderLoop(() => {
    const dt = Math.min(0.05, engine.getDeltaTime() / 1000);
    stepMe(dt);                                 // 내 캐릭터는 늘 실제 시간
    if (!brain?.on && play) stepPlay(engine.getDeltaTime() / h.approach.dt_ms);   // 시험용 사람은 뇌가 켜지면 뇌 시간
    place(flyNode, fly.pos, fly.yaw);
    const live = brain?.on || window.__force;
    if (flyRigBones) stepPose(flyRigBones, live ? lastParts : null, live ? lastSpeed : 0, dt);
    ring.position.set(fly.pos[0], h.fly_body.kind === 'fly' ? 0.002 : 0.03, fly.pos[2]);
    stepCam();
    scene.render();
    const now = performance.now();
    if (!brain?.on && !eyeBusy && now - lastEye > h.eye_every_ms) { lastEye = now; updateEye(); }
  });
  window.addEventListener('resize', () => engine.resize());
}

main().catch((e) => { $('status').textContent = `오류: ${e}`; console.error(e); });
