// R-1177 F-008 보기 창(최소판) — 같은 장면을 사람이 둘러본다. 장면 · 자세는 /view/<이름>.json(파이썬이 만듦).
// 장면 짓는 순서는 render.js loadScene 과 같다(땅 · 빛 · 에셋 · 자세 · 좌표 오른손 · y 위). 렌더러(판정용)는 건드리지 않는다.
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
let engine, scene, cam, h;
const nodes = {};
let play = null;   // { traj, t0 }

async function loadContainer(url) {
  if (B.LoadAssetContainerAsync) return await B.LoadAssetContainerAsync(url, scene);
  const i = url.lastIndexOf('/');
  return await B.SceneLoader.LoadAssetContainerAsync(url.slice(0, i + 1), url.slice(i + 1), scene);
}

function setPose(id, p) {
  const n = nodes[id];
  if (!n) return;
  n.position.set(p.pos[0], p.pos[1], p.pos[2]);
  n.rotation.set(0, p.yaw, 0);
  const s = p.scale ?? 1;
  n.scaling.set(s, s, s);
}

function toEye() {
  const e = h.eye;
  cam.position.set(e[0], e[1], e[2]);
  cam.setTarget(new B.Vector3(e[0] + Math.sin(h.fly_yaw), e[1], e[2] + Math.cos(h.fly_yaw)));
}

function toTop() {
  cam.position.set(0.01, 28, -0.01);
  cam.setTarget(new B.Vector3(0, 0, 0));
}

async function main() {
  h = await (await fetch(`/view/${NAME}.json`, { cache: 'no-store' })).json();
  $('name').textContent = `${NAME} · ${h.version}`;
  const canvas = $('c');
  engine = new B.Engine(canvas, true, { stencil: true }, true);
  scene = new B.Scene(engine);
  scene.useRightHandedSystem = true;
  const sky = h.sky?.color ?? [0.6, 0.75, 0.9];
  scene.clearColor = new B.Color4(sky[0], sky[1], sky[2], 1);
  cam = new B.UniversalCamera('view', B.Vector3.Zero(), scene);
  cam.minZ = 0.05; cam.maxZ = 1000; cam.fov = 1.1;
  cam.speed = 0.12; cam.angularSensibility = 2500;
  cam.keysUp.push(87); cam.keysDown.push(83); cam.keysLeft.push(65); cam.keysRight.push(68);
  cam.attachControl(canvas, true);
  const hemi = new B.HemisphericLight('hemi', new B.Vector3(0, 1, 0), scene);
  hemi.intensity = h.light?.ambient ?? 0.55;
  const sun = new B.DirectionalLight('sun', new B.Vector3(-0.4, -1, -0.3), scene);
  if (h.light?.sun_dir) sun.direction = new B.Vector3(...h.light.sun_dir);
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
  const containers = {}, missing = [];
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
    setPose(p.id, p);
  }
  // 초파리 자리 표지(땅 위 노란 고리 · 몸 반지름) — 그림 판정에는 없는 보기 전용
  const ring = B.MeshBuilder.CreateTorus('flyRing', { diameter: 2 * h.fly_radius, thickness: 0.04, tessellation: 48 }, scene);
  ring.position.set(h.fly_pos[0], 0.03, h.fly_pos[2]);
  const rm = new B.StandardMaterial('ringMat', scene);
  rm.emissiveColor = new B.Color3(0.95, 0.76, 0.3); rm.disableLighting = true;
  ring.material = rm;
  toEye();
  $('eye').onclick = toEye;
  $('top').onclick = toTop;
  $('appL').onclick = () => { play = { traj: h.approach.left, t0: performance.now() }; };
  $('appR').onclick = () => { play = { traj: h.approach.right, t0: performance.now() }; };
  $('fly1').src = h.flyeye[0].url; $('cap1').textContent = h.flyeye[0].caption;
  $('fly2').src = h.flyeye[1].url; $('cap2').textContent = h.flyeye[1].caption;
  await scene.whenReadyAsync();
  $('status').textContent = `에셋 ${Object.keys(containers).length} 실림 · 빠짐 ${missing.length} · 물체 ${h.poses.length}`;
  engine.runRenderLoop(() => {
    if (play) {   // 대본과 같은 자세 차례(10 ms 걸음)를 실제 시간으로
      const k = Math.floor((performance.now() - play.t0) / h.approach.dt_ms);
      const tr = play.traj;
      setPose('person', tr[Math.min(k, tr.length - 1)]);
      if (k >= tr.length + 100) { setPose('person', h.person_rest); play = null; }
    }
    scene.render();
  });
  window.addEventListener('resize', () => engine.resize());
}

main().catch((e) => { $('status').textContent = `오류: ${e}`; console.error(e); });
