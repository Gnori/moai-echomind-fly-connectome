// R-1177 렌더러(설계 문서 I 2.6) — 상태 없음: 세계 서버가 보낸 자세대로 그리고, 눈마다 큐브맵 6 면 밝기를 돌려준다.
// 통로 메시지 = [u32 머리 길이(little)][머리 JSON][본문]. 좌표: 오른손 · y 위(에코월드와 같음).
const B = window.BABYLON;
const q = new URLSearchParams(location.search);
const WS_URL = `ws://127.0.0.1:${q.get('ws')}`;
const WANT_GPU = q.get('gpu') === '1';

// KTX2 해독기는 로컬(에코월드 bj/engine.js 와 같은 파일 · 바깥 안 부름)
if (B.KhronosTextureContainer2?.URLConfig) {
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

const enc = new TextEncoder(), dec = new TextDecoder();
function pack(head, body) {
  const h = enc.encode(JSON.stringify(head));
  const n = body ? body.byteLength : 0;
  const buf = new ArrayBuffer(4 + h.length + n);
  new DataView(buf).setUint32(0, h.length, true);
  new Uint8Array(buf, 4, h.length).set(h);
  if (body) new Uint8Array(buf, 4 + h.length).set(new Uint8Array(body.buffer, body.byteOffset, n));
  return buf;
}
function unpack(buf) {
  const n = new DataView(buf).getUint32(0, true);
  return [JSON.parse(dec.decode(new Uint8Array(buf, 4, n))), buf.slice(4 + n)];
}

let engine, scene, cam, mainCam, rtt = null, ground = null, sun = null, hemi = null;
const nodes = {};          // 개체 id → TransformNode
const containers = {};     // 에셋 키 → AssetContainer

async function init() {
  const canvas = document.getElementById('c');
  if (WANT_GPU && B.WebGPUEngine && (await B.WebGPUEngine.IsSupportedAsync)) {
    try {
      const g = new B.WebGPUEngine(canvas, {
        antialias: false, setMaximumLimits: true, stencil: true, premultipliedAlpha: false,
        glslangOptions: { jsPath: '/babylon/glslang.js', wasmPath: '/babylon/glslang.wasm' },
        twgslOptions: { jsPath: '/babylon/twgsl.js', wasmPath: '/babylon/twgsl.wasm' },
      });
      await g.initAsync();
      engine = g;
    } catch (e) { console.warn('[eyeworld] WebGPU 실패 → WebGL2', e); }
  }
  if (!engine) engine = new B.Engine(canvas, false, { preserveDrawingBuffer: true, stencil: true, premultipliedAlpha: false }, true);
  scene = new B.Scene(engine);
  scene.useRightHandedSystem = true;
  scene.skipPointerMovePicking = true;
  cam = new B.UniversalCamera('eye', B.Vector3.Zero(), scene);
  cam.fovMode = B.Camera.FOVMODE_VERTICAL_FIXED;
  cam.fov = Math.PI / 2;                 // 정사각 면 · 90° = 큐브맵 한 면
  cam.minZ = 0.05; cam.maxZ = 1000;
  // 장면 카메라는 눈 카메라와 따로(10-17 F0-4): 같은 카메라면 Babylon 이 면 그리기 전에 변환 행렬을
  // 다시 넣지 않아 첫 면(왼눈 앞)이 직전 호출 마지막 면(오른눈 아래) 시점으로 그려졌다.
  mainCam = new B.UniversalCamera('main', B.Vector3.Zero(), scene);
  mainCam.minZ = 0.05; mainCam.maxZ = 1000;
  scene.activeCamera = mainCam;
  hemi = new B.HemisphericLight('hemi', new B.Vector3(0, 1, 0), scene);
  sun = new B.DirectionalLight('sun', new B.Vector3(-0.4, -1, -0.3), scene);
}

async function loadContainer(url) {
  if (B.LoadAssetContainerAsync) return await B.LoadAssetContainerAsync(url, scene);
  const i = url.lastIndexOf('/');
  return await B.SceneLoader.LoadAssetContainerAsync(url.slice(0, i + 1), url.slice(i + 1), scene);
}

async function loadScene(h) {
  const t0 = performance.now();
  const sky = h.sky?.color ?? [0.6, 0.75, 0.9];
  scene.clearColor = new B.Color4(sky[0], sky[1], sky[2], 1);
  hemi.intensity = h.light?.ambient ?? 0.55;
  if (h.light?.sun_dir) sun.direction = new B.Vector3(...h.light.sun_dir);
  if (ground) ground.dispose();
  const gs = h.ground?.size_m ?? 60;
  ground = B.MeshBuilder.CreateGround('ground', { width: gs, height: gs }, scene);
  const gm = new B.StandardMaterial('groundMat', scene);
  const gc = h.ground?.color ?? [0.3, 0.45, 0.2];
  gm.diffuseColor = new B.Color3(gc[0], gc[1], gc[2]);
  gm.specularColor = new B.Color3(0, 0, 0);
  if (h.ground?.texture_url) {               // 잔디 사진 텍스처(CC0) — 타일 크기(m)로 되풀이
    const tex = new B.Texture(h.ground.texture_url, scene);
    const tile = h.ground.tile_m ?? 1.4;
    tex.uScale = gs / tile; tex.vScale = gs / tile;
    gm.diffuseTexture = tex;
    gm.diffuseColor = new B.Color3(1, 1, 1);
  }
  ground.material = gm;
  const missing = [];
  for (const [key, url] of Object.entries(h.assets)) {
    try { containers[key] = await loadContainer(url); } catch (e) { missing.push(key); console.warn(e); }
  }
  for (const p of h.poses) {
    const c = containers[p.asset];
    if (!c) continue;
    const node = new B.TransformNode(`ent:${p.id}`, scene);
    const inst = c.instantiateModelsToScene((n) => `${p.id}:${n}`, false);
    for (const r of inst.rootNodes) r.parent = node;
    nodes[p.id] = node;
  }
  for (const [k, m] of (h.markers ?? []).entries()) {   // 방향 확인용 표지(빛나는 구슬 · 판정 아님)
    const s = B.MeshBuilder.CreateSphere(`marker:${k}`, { diameter: 2 * (m.radius ?? 0.2), segments: 16 }, scene);
    s.position.set(m.pos[0], m.pos[1], m.pos[2]);
    const mm = new B.StandardMaterial(`markerMat:${k}`, scene);
    const c = m.color ?? [1, 1, 1];
    mm.emissiveColor = new B.Color3(c[0], c[1], c[2]);
    mm.disableLighting = true;
    s.material = mm;
  }
  applyPoses(h.poses);
  await scene.whenReadyAsync();
  return { loaded: Object.keys(containers).length, missing, ms: Math.round(performance.now() - t0) };
}

function applyPoses(poses) {
  for (const p of poses) {
    const node = nodes[p.id];
    if (!node) continue;
    node.position.set(p.pos[0], p.pos[1], p.pos[2]);
    node.rotation.set(0, p.yaw, 0);      // 오른손 · y 축 회전 → 앞 = (sin yaw, 0, cos yaw)
    const s = p.scale ?? 1;
    node.scaling.set(s, s, s);
  }
}

async function render(h) {
  const t0 = performance.now();
  applyPoses(h.poses);
  scene.render();      // 뼈대 · 세계 행렬 갱신(안 하면 스킨 메시가 한 점으로 오그라들어 안 보임 — V1-a 10-17)
  const N = h.face_px;
  if (!rtt || rtt.getSize().width !== N) {
    if (rtt) rtt.dispose();
    rtt = new B.RenderTargetTexture('eyeRT', { width: N, height: N }, scene, false, true,
      B.Constants.TEXTURETYPE_UNSIGNED_BYTE);
    rtt.activeCamera = cam;
  }
  rtt.renderList = scene.meshes;
  const eyes = h.eyes;
  const out = new Uint8Array(eyes.length * 6 * N * N);
  let o = 0, readMs = 0;
  for (const eye of eyes) {
    const pos = new B.Vector3(...eye.pos);
    for (const f of eye.faces) {
      cam.position.copyFrom(pos);
      cam.upVector = new B.Vector3(...f.up);
      cam.setTarget(pos.add(new B.Vector3(...f.forward)));
      cam.getViewMatrix(true);
      rtt.render();
      const r0 = performance.now();
      const px = await rtt.readPixels();
      readMs += performance.now() - r0;
      for (let i = 0; i < N * N; i++) {   // 밝기 = 초록 가중(R1-R6 흡수 봉우리 · 선언) · 줄 0 = 아래
        out[o++] = Math.round(0.2 * px[4 * i] + 0.7 * px[4 * i + 1] + 0.1 * px[4 * i + 2]);
      }
    }
  }
  return [{ type: 'frames', tick: h.tick, n_eyes: eyes.length, face_px: N, dtype: 'u8', row0: 'bottom',
    faces: eyes[0]?.faces.map((f) => f.name) ?? [], render_ms: Math.round(performance.now() - t0),
    read_ms: Math.round(readMs) }, out];
}

function info(id) {                          // 진단 — 개체 노드 아래 메시들의 상태
  const node = nodes[id];
  const ms = node ? node.getChildMeshes(false) : [];
  return { id, has_node: !!node, meshes: ms.length, scene_meshes: scene.meshes.length,
    detail: ms.slice(0, 5).map((m) => ({ name: m.name, enabled: m.isEnabled(), visible: m.isVisible,
      verts: m.getTotalVertices(), skinned: !!m.skeleton,
      bbox: m.getBoundingInfo().boundingBox.minimumWorld.asArray().concat(m.getBoundingInfo().boundingBox.maximumWorld.asArray()) })),
    pos: node ? node.position.asArray() : null };
}

const ws = new WebSocket(WS_URL);
ws.binaryType = 'arraybuffer';
ws.onopen = async () => {
  try {
    await init();
    ws.send(pack({ type: 'ready', babylon: B.Engine.Version, backend: engine.isWebGPU ? 'webgpu' : 'webgl2',
      max_texture: engine.getCaps().maxTextureSize }));
  } catch (e) { ws.send(pack({ type: 'error', code: 'init', msg: String(e?.stack ?? e) })); }
};
ws.onmessage = async (ev) => {
  const [h] = unpack(ev.data);
  try {
    if (h.type === 'scene.load') ws.send(pack({ type: 'scene.ready', ...(await loadScene(h)) }));
    else if (h.type === 'render') { const [head, body] = await render(h); ws.send(pack(head, body)); }
    else if (h.type === 'info') ws.send(pack({ type: 'info', ...info(h.id) }));
    else if (h.type === 'bye') ws.close();
  } catch (e) { ws.send(pack({ type: 'error', code: h.type, msg: String(e?.stack ?? e) })); }
};
