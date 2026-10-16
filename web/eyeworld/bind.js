// R-1177 — 에코월드 에셋 재질 묶기(렌더러 · 보기 창 같이 씀). 10-17: 식물 GLB 는 재질 이름만 들고 오고
// 텍스처는 색인으로 묶는다는 에코월드 자산 계약을 따른다(에코월드 파일은 읽기만 · 계약 = 그쪽 bj/foliage.js 머리 주석):
//   색인(foliage_index.json): 재질 이름 → { ext, maps[diff|nor|arm], dir, role?, tex_from? }
//   파일: <base><dir>/<tex_from 또는 재질 이름>_<맵>.<ext>
//   diff = 바탕색(잎은 알파 컷 0.5) · nor = 법선 · arm = 가림 · 거칠기 · 금속(초록 = 거칠기 · 파랑 = 금속 · 빨강 = 가림)
//   잎(role leaf 또는 이름 끝 _leaves · _twig)은 양면 · 양면 조명.
// 색인에 없는 재질은 그대로 둔다(지어내지 않음) — 묶지 못한 이름은 돌려준다(진단용).
const MAP_ROLE = { diff: 'albedo', nor: 'normal', arm: 'orm' };

export async function loadIndex(url) {
  const r = await fetch(url, { cache: 'no-store' });
  if (!r.ok) throw new Error(`색인 ${url} → ${r.status}`);
  return await r.json();
}

/** 컨테이너 재질을 색인으로 묶는다 → { bound: [...], unknown: [...] } */
export function bindContainer(B, scene, container, index, base, alphaCut = 0.5) {
  const bound = [], unknown = [];
  for (const mat of container.materials) {
    const ent = index[mat.name];
    if (!ent) { unknown.push(mat.name); continue; }
    const tex = (map, srgb) => {
      const file = `${base}${ent.dir}/${ent.tex_from || mat.name}_${map}.${ent.ext || 'jpg'}`;
      const t = new B.Texture(file, scene, false, false, B.Texture.TRILINEAR_SAMPLINGMODE);
      t.gammaSpace = srgb;
      return t;
    };
    for (const map of ent.maps || []) {
      const role = MAP_ROLE[String(map).toLowerCase()];
      if (role === 'albedo') {
        mat.albedoTexture = tex(map, true);
        mat.albedoTexture.hasAlpha = true;
        mat.useAlphaFromAlbedoTexture = true;
        mat.transparencyMode = B.Material.MATERIAL_ALPHATEST;
        mat.alphaCutOff = alphaCut;
      } else if (role === 'normal') {
        mat.bumpTexture = tex(map, false);
      } else if (role === 'orm') {
        mat.metallicTexture = tex(map, false);
        mat.useRoughnessFromMetallicTextureAlpha = false;
        mat.useRoughnessFromMetallicTextureGreen = true;
        mat.useMetallnessFromMetallicTextureBlue = true;
        mat.useAmbientOcclusionFromMetallicTextureRed = true;
        mat.metallic = 1;
        mat.roughness = 1;
      }
    }
    mat.backFaceCulling = false;
    const leaf = ent.role === 'leaf' || /_(leaves|twig)$/.test(mat.name);
    if (leaf) mat.twoSidedLighting = true;
    bound.push(mat.name);
  }
  return { bound, unknown };
}
