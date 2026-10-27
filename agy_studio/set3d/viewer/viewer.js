// AGY Studio 3D FPV Director — Zero-Blender 100% Procedural Three.js + Custom GLSL Shaders at 60 FPS.
// Builds the 1:1 room architecture, furniture, tableware, volumetric morning sunlight, and articulated 3D cast
// (Father, Rishi, Ria, Mother, Delivery Man) with FPV Walk/Fly + Orbit camera controls and 1-click Clean 3D PNG export.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const SENSOR_W = 36;
const ASPECT = { '16:9': 16 / 9, '9:16': 9 / 16, '1:1': 1, '4:3': 4 / 3, '2.39:1': 2.39 };
const RENDER_LONG_EDGE = 1920;

const state = {
  env: null,
  layout: null,
  sets: [],
  selected: null,
  activeCam: null,
  fly: false,
  keys: new Set(),
  dirty: false,
  standalone: typeof window !== 'undefined' && !!window.__INLINE_SET_LAYOUT__,
  fpsFrames: 0,
  fpsLastTime: performance.now(),
  animatedRigNodes: [],
};

// ------------------------------------------------------------------------------------------------- WebGL Renderer
const canvas = $('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.08;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 16 / 9, 0.05, 250);
camera.position.set(0.82, 1.32, 0.98);

const controls = new OrbitControls(camera, canvas);
controls.target.set(-0.15, 1.02, -0.20);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.maxPolarAngle = Math.PI * 0.98;

const setGroup = new THREE.Group();
const mqGroup = new THREE.Group();
const labelGroup = new THREE.Group();
const godRayGroup = new THREE.Group();
const lightGroup = new THREE.Group();
const helperGroup = new THREE.Group();
scene.add(setGroup, mqGroup, labelGroup, godRayGroup, lightGroup, helperGroup);

const grid = new THREE.GridHelper(20, 20, 0x3a4150, 0x262b36);
grid.position.y = 0.002;
grid.visible = false;
helperGroup.add(grid);

const axisLine = new THREE.Line(
  new THREE.BufferGeometry(),
  new THREE.LineDashedMaterial({ color: 0xf0a84a, dashSize: 0.2, gapSize: 0.1 })
);
axisLine.visible = false;
helperGroup.add(axisLine);

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let selectionBox = null;

function hexColor(c, fallback = '#9a9a9a') {
  return new THREE.Color(c || fallback);
}

// ------------------------------------------------------------------------------------------------- Custom GLSL Shaders (60 FPS Procedural Materials)
const COMMON_VERT_GLSL = `
  varying vec3 vWorldPos;
  varying vec3 vWorldNormal;
  varying vec2 vUv;
  void main() {
    vUv = uv;
    vec4 wp = modelMatrix * vec4(position, 1.0);
    vWorldPos = wp.xyz;
    vWorldNormal = normalize(mat3(modelMatrix) * normal);
    gl_Position = projectionMatrix * viewMatrix * wp;
  }
`;

// 1. Procedural Teak Wood Grain GLSL Shader
function createWoodShaderMaterial(hexStr, grainScale = 14.0) {
  const base = hexColor(hexStr, '#6e4526');
  return new THREE.ShaderMaterial({
    uniforms: {
      uBaseColor: { value: new THREE.Vector3(base.r, base.g, base.b) },
      uGrainScale: { value: grainScale },
      uSunDir: { value: new THREE.Vector3(-0.75, 0.55, 0.35).normalize() },
    },
    vertexShader: COMMON_VERT_GLSL,
    fragmentShader: `
      uniform vec3 uBaseColor;
      uniform float uGrainScale;
      uniform vec3 uSunDir;
      varying vec3 vWorldPos;
      varying vec3 vWorldNormal;
      varying vec2 vUv;

      float hash(vec3 p) {
        p = fract(p * 0.3183099 + 0.1);
        p *= 17.0;
        return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
      }
      float noise(vec3 x) {
        vec3 i = floor(x);
        vec3 f = fract(x);
        f = f * f * (3.0 - 2.0 * f);
        return mix(
          mix(mix(hash(i + vec3(0,0,0)), hash(i + vec3(1,0,0)), f.x),
              mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
          mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x),
              mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y), f.z);
      }
      void main() {
        vec3 p = vWorldPos * uGrainScale;
        float n = noise(p * vec3(1.0, 0.35, 1.0)) * 2.5;
        float ring = sin((p.x + p.z * 0.6) * 3.5 + n * 4.0) * 0.5 + 0.5;
        float fine = noise(p * 6.0) * 0.15;
        vec3 darkWood = uBaseColor * 0.68;
        vec3 lightWood = uBaseColor * 1.16;
        vec3 albedo = mix(darkWood, lightWood, ring) + fine * 0.06;

        vec3 N = normalize(vWorldNormal);
        float ndl = max(dot(N, uSunDir), 0.0);
        float hemi = 0.52 + 0.48 * N.y;
        vec3 viewDir = normalize(cameraPosition - vWorldPos);
        vec3 H = normalize(uSunDir + viewDir);
        float spec = pow(max(dot(N, H), 0.0), 28.0) * 0.22;

        vec3 col = albedo * (0.48 * hemi + 0.62 * ndl) + vec3(1.0, 0.95, 0.85) * spec;
        gl_FragColor = vec4(col, 1.0);
      }
    `,
  });
}

// 2. Procedural Brushed Anisotropic Stainless Steel GLSL Shader
function createSteelShaderMaterial(hexStr = '#cfd6dd') {
  const base = hexColor(hexStr, '#cfd6dd');
  return new THREE.ShaderMaterial({
    uniforms: {
      uBaseColor: { value: new THREE.Vector3(base.r, base.g, base.b) },
      uSunDir: { value: new THREE.Vector3(-0.75, 0.55, 0.35).normalize() },
    },
    vertexShader: COMMON_VERT_GLSL,
    fragmentShader: `
      uniform vec3 uBaseColor;
      uniform vec3 uSunDir;
      varying vec3 vWorldPos;
      varying vec3 vWorldNormal;
      varying vec2 vUv;

      void main() {
        vec3 N = normalize(vWorldNormal);
        vec3 V = normalize(cameraPosition - vWorldPos);
        vec3 R = reflect(-V, N);
        float brush = sin((vWorldPos.x * 160.0 + vWorldPos.z * 160.0 + vUv.y * 90.0)) * 0.04;
        float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.0);
        float envSky = smoothstep(-0.2, 0.8, R.y);
        vec3 envCol = mix(vec3(0.32, 0.26, 0.20), vec3(0.96, 0.95, 0.92), envSky);
        vec3 H = normalize(uSunDir + V);
        float spec = pow(max(dot(N, H), 0.0), 64.0) * 0.65;
        vec3 col = mix(uBaseColor + brush, envCol, 0.45 + 0.35 * fresnel) + vec3(1.0, 0.98, 0.92) * spec;
        gl_FragColor = vec4(col, 1.0);
      }
    `,
  });
}

// 3. Procedural Warm Indian Stucco Plaster Wall GLSL Shader
function createPlasterShaderMaterial(hexStr = '#e6dec8') {
  const base = hexColor(hexStr, '#e6dec8');
  return new THREE.ShaderMaterial({
    uniforms: {
      uBaseColor: { value: new THREE.Vector3(base.r, base.g, base.b) },
      uSunDir: { value: new THREE.Vector3(-0.72, 0.45, 0.35).normalize() },
    },
    vertexShader: COMMON_VERT_GLSL,
    fragmentShader: `
      uniform vec3 uBaseColor;
      uniform vec3 uSunDir;
      varying vec3 vWorldPos;
      varying vec3 vWorldNormal;

      float hash(vec2 p) {
        return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
      }
      void main() {
        vec3 N = normalize(vWorldNormal);
        float grain = (hash(floor(vWorldPos.xy * 120.0 + vWorldPos.zy * 90.0)) - 0.5) * 0.025;
        // Soft morning window light falloff from West (-X)
        float windowProximity = clamp(1.0 - (vWorldPos.x + 1.8) * 0.16, 0.68, 1.08);
        float ndl = max(dot(N, uSunDir), 0.0) * 0.35 + 0.72;
        vec3 col = (uBaseColor + grain) * ndl * windowProximity;
        gl_FragColor = vec4(col, 1.0);
      }
    `,
  });
}

// 4. Procedural Ceramic Floor Tile GLSL Shader
function createTileShaderMaterial(hexStr = '#d8d0c0') {
  const base = hexColor(hexStr, '#d8d0c0');
  return new THREE.ShaderMaterial({
    uniforms: {
      uBaseColor: { value: new THREE.Vector3(base.r, base.g, base.b) },
    },
    vertexShader: COMMON_VERT_GLSL,
    fragmentShader: `
      uniform vec3 uBaseColor;
      varying vec3 vWorldPos;
      void main() {
        vec2 tileUv = fract(vWorldPos.xz * 1.65);
        float grout = step(0.025, tileUv.x) * step(0.025, tileUv.y);
        float sunPatch = smoothstep(1.2, -1.4, vWorldPos.x) * 0.15;
        vec3 col = mix(uBaseColor * 0.72, uBaseColor + sunPatch, grout);
        gl_FragColor = vec4(col, 1.0);
      }
    `,
  });
}

// 5. Volumetric Morning Sunlight Shafts (God-Rays) GLSL Shader
function buildMorningGodRays() {
  while (godRayGroup.children.length) godRayGroup.remove(godRayGroup.children[0]);
  const rayMat = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
    uniforms: {
      uTime: { value: 0 },
    },
    vertexShader: COMMON_VERT_GLSL,
    fragmentShader: `
      uniform float uTime;
      varying vec2 vUv;
      varying vec3 vWorldPos;
      void main() {
        float fadeX = smoothstep(0.0, 0.18, vUv.x) * smoothstep(1.0, 0.15, vUv.x);
        float fadeY = smoothstep(0.0, 0.25, vUv.y) * smoothstep(1.0, 0.0, vUv.y);
        float streaks = 0.65 + 0.35 * sin(vUv.y * 22.0 + uTime * 0.6);
        float alpha = fadeX * fadeY * streaks * 0.095;
        gl_FragColor = vec4(1.0, 0.93, 0.76, alpha);
      }
    `,
  });
  for (let i = 0; i < 3; i++) {
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(2.6, 1.45), rayMat);
    plane.position.set(-0.75, 1.42 - i * 0.08, -0.35 + (i - 1) * 0.38);
    plane.rotation.y = THREE.MathUtils.degToRad(12 * (i - 1));
    plane.rotation.z = THREE.MathUtils.degToRad(-24);
    godRayGroup.add(plane);
  }
  state.godRayMat = rayMat;
}

// ------------------------------------------------------------------------------------------------- Smart Material Dispatcher
function materialForPrimitive(p) {
  const id = (p.id || '').toLowerCase();
  const note = (p.material_note || '').toLowerCase();
  if (id.includes('floor')) return createTileShaderMaterial(p.color);
  if (id.startsWith('wall_') || id.includes('ceiling')) return createPlasterShaderMaterial(p.color);
  if (
    id.includes('table') ||
    id.includes('chair') ||
    id.includes('door') ||
    id.includes('frame') ||
    id.includes('casserole') ||
    note.includes('wood') ||
    note.includes('teak')
  ) {
    return createWoodShaderMaterial(p.color);
  }
  if (id.includes('plate') || id.includes('glass') || id.includes('tumbler') || note.includes('steel')) {
    return createSteelShaderMaterial(p.color);
  }
  if (id.includes('window')) {
    return new THREE.MeshBasicMaterial({ color: hexColor('#fffdf6') });
  }
  return new THREE.MeshStandardMaterial({
    color: hexColor(p.color),
    roughness: 0.75,
    metalness: 0.05,
  });
}

function buildPrimitive(p) {
  let geo;
  const [sx, sy, sz] = p.size;
  if (p.kind === 'plane') geo = new THREE.BoxGeometry(sx, Math.max(0.01, Math.min(sy, 0.02)), sz);
  else if (p.kind === 'cylinder') geo = new THREE.CylinderGeometry(sx / 2, sx / 2, sy, 32);
  else if (p.kind === 'sphere') geo = new THREE.SphereGeometry(sx / 2, 24, 16);
  else geo = new THREE.BoxGeometry(sx, sy, sz);

  const mat = materialForPrimitive(p);
  const m = new THREE.Mesh(geo, mat);
  m.position.set(...p.position);
  m.rotation.y = THREE.MathUtils.degToRad(p.rotation_y_deg || 0);
  m.castShadow = !p.structural;
  m.receiveShadow = true;
  m.userData = { prim: p };
  return m;
}

// ------------------------------------------------------------------------------------------------- Floating 3D Name Tag Sprite
function makeLabelSprite(text, accentHex = '#f0a84a') {
  const c = document.createElement('canvas');
  c.width = 256;
  c.height = 64;
  const ctx = c.getContext('2d');
  ctx.fillStyle = 'rgba(15, 17, 21, 0.82)';
  ctx.beginPath();
  ctx.roundRect(8, 8, 240, 48, 24);
  ctx.fill();
  ctx.lineWidth = 3;
  ctx.strokeStyle = accentHex;
  ctx.stroke();
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 24px system-ui, -apple-system, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 128, 33);

  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(0.42, 0.105, 1);
  return sprite;
}

// ------------------------------------------------------------------------------------------------- Full Articulated 3D Cast Builder
function buildArticulatedCharacter(ch) {
  const cid = ch.character_id || ch.id || 'CHAR';
  const label = ch.label || cid;
  const pose = ch.pose || 'sitting';
  const h = ch.height_m || 1.68;

  const root = new THREE.Group();
  root.position.set(ch.position[0], 0, ch.position[2]);
  // In AGY set.json: facing_deg = 0 faces +Z (South); +90 faces +X (East); -90 faces -X (West).
  // In Three.js local space, +Z is forward when rotation.y = degToRad(facing_deg).
  root.rotation.y = THREE.MathUtils.degToRad(ch.facing_deg || 0);
  root.visible = ch.visible !== false;
  root.userData = { mq: ch, isCharacter: true };

  const skinMat = new THREE.MeshStandardMaterial({ color: hexColor(ch.skin_color || '#c68e68'), roughness: 0.65 });
  const hairMat = new THREE.MeshStandardMaterial({
    color: hexColor(cid === 'CHAR_FATHER' ? '#3a3a3c' : '#181616'),
    roughness: 0.55,
  });
  const topMat = new THREE.MeshStandardMaterial({ color: hexColor(ch.wardrobe_color || '#4a6fa5'), roughness: 0.8 });
  const bottomColors = {
    CHAR_FATHER: '#8c8275',
    CHAR_RISHI: '#222834',
    CHAR_RIA: '#2f3542',
    CHAR_MOTHER: '#6b6e65',
    CHAR_DELIVERY_MAN: '#2b3240',
  };
  const bottomMat = new THREE.MeshStandardMaterial({ color: hexColor(bottomColors[cid] || '#30343f'), roughness: 0.85 });
  const darkPropMat = new THREE.MeshStandardMaterial({ color: hexColor('#141619'), roughness: 0.3, metalness: 0.5 });
  const phoneScreenMat = new THREE.MeshBasicMaterial({ color: hexColor('#68b5ff') });
  const wireMat = new THREE.MeshBasicMaterial({ color: hexColor('#f5f7fa') });
  const paperMat = new THREE.MeshStandardMaterial({ color: hexColor('#f2efe6'), roughness: 0.9 });
  const steelMat = createSteelShaderMaterial('#d4dbe2');

  const upperBody = new THREE.Group();
  root.add(upperBody);

  if (pose === 'sitting') {
    const seatY = 0.46;
    const torsoH = 0.44;
    const torsoY = seatY + torsoH * 0.5;
    const shoulderY = seatY + torsoH - 0.04;
    const headY = shoulderY + 0.16;

    // Hips + Thighs + Calves + Feet (local +Z is forward)
    const hips = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.12, 0.26), bottomMat);
    hips.position.set(0, seatY, 0);
    root.add(hips);

    for (const sx of [-0.09, 0.09]) {
      const thigh = new THREE.Mesh(new THREE.CylinderGeometry(0.058, 0.052, 0.36, 16), bottomMat);
      thigh.rotation.x = Math.PI / 2;
      thigh.position.set(sx, seatY + 0.01, 0.16);
      const calf = new THREE.Mesh(new THREE.CylinderGeometry(0.048, 0.042, 0.40, 16), skinMat);
      calf.position.set(sx, 0.23, 0.32);
      const foot = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.05, 0.18), skinMat);
      foot.position.set(sx, 0.03, 0.36);
      root.add(thigh, calf, foot);
    }

    // Torso, Neck, Head
    const torso = new THREE.Mesh(new THREE.BoxGeometry(0.34, torsoH, 0.21), topMat);
    torso.position.set(0, torsoY, 0);
    torso.castShadow = true;
    const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.048, 0.09, 16), skinMat);
    neck.position.set(0, shoulderY + 0.04, 0);
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.105, 24, 20), skinMat);
    head.scale.set(0.92, 1.12, 1.0);
    head.position.set(0, headY, 0.01);
    head.castShadow = true;
    upperBody.add(torso, neck, head);

    // Hair & Facial Hair
    const hairCap = new THREE.Mesh(new THREE.SphereGeometry(0.108, 20, 16), hairMat);
    hairCap.scale.set(0.96, 0.95, 1.05);
    hairCap.position.set(0, headY + 0.025, -0.015);
    upperBody.add(hairCap);

    if (cid === 'CHAR_FATHER') {
      // Moustache + Eyeglasses + Sipping Steel Chai Glass (Right Hand) + Newspaper (Left Hand)
      const moustache = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.02, 0.025), hairMat);
      moustache.position.set(0, headY - 0.038, 0.102);
      const glassBridge = new THREE.Mesh(new THREE.BoxGeometry(0.13, 0.015, 0.02), darkPropMat);
      glassBridge.position.set(0, headY + 0.01, 0.104);
      upperBody.add(moustache, glassBridge);

      // Right arm (-X in local coords when facing +Z) raised to sip chai tumbler
      const armR = new THREE.Mesh(new THREE.CylinderGeometry(0.042, 0.038, 0.26, 14), topMat);
      armR.position.set(-0.20, shoulderY - 0.06, 0.10);
      armR.rotation.set(THREE.MathUtils.degToRad(-55), 0, THREE.MathUtils.degToRad(-15));
      const forearmR = new THREE.Mesh(new THREE.CylinderGeometry(0.036, 0.032, 0.24, 14), skinMat);
      forearmR.position.set(-0.12, shoulderY + 0.03, 0.22);
      forearmR.rotation.set(THREE.MathUtils.degToRad(-25), 0, THREE.MathUtils.degToRad(28));
      const chaiGlass = new THREE.Mesh(new THREE.CylinderGeometry(0.034, 0.028, 0.105, 20), steelMat);
      chaiGlass.position.set(-0.04, headY - 0.05, 0.17);
      chaiGlass.rotation.x = THREE.MathUtils.degToRad(22);

      // Left arm (+X in local coords) holding open newspaper over the table
      const armL = new THREE.Mesh(new THREE.CylinderGeometry(0.042, 0.038, 0.26, 14), topMat);
      armL.position.set(0.20, shoulderY - 0.08, 0.10);
      armL.rotation.set(THREE.MathUtils.degToRad(-45), 0, THREE.MathUtils.degToRad(12));
      const forearmL = new THREE.Mesh(new THREE.CylinderGeometry(0.036, 0.032, 0.24, 14), skinMat);
      forearmL.position.set(0.16, shoulderY - 0.12, 0.26);
      forearmL.rotation.set(THREE.MathUtils.degToRad(-75), 0, THREE.MathUtils.degToRad(-20));
      const newspaper = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.23, 0.008), paperMat);
      newspaper.position.set(0.08, shoulderY - 0.06, 0.34);
      newspaper.rotation.set(THREE.MathUtils.degToRad(-28), THREE.MathUtils.degToRad(-12), 0);

      upperBody.add(armR, forearmR, chaiGlass, armL, forearmL, newspaper);
    } else if (cid === 'CHAR_RISHI') {
      // Rishi: White wired earphones + looking down at black smartphone in left hand + eating roti with right hand
      head.rotation.x = THREE.MathUtils.degToRad(16); // Tilted down toward phone
      for (const sx of [-0.095, 0.095]) {
        const earbud = new THREE.Mesh(new THREE.SphereGeometry(0.014, 10, 10), wireMat);
        earbud.position.set(sx, headY - 0.01, 0.01);
        const cord = new THREE.Mesh(new THREE.CylinderGeometry(0.004, 0.004, 0.28, 8), wireMat);
        cord.position.set(sx * 0.55, shoulderY - 0.02, 0.13);
        cord.rotation.set(THREE.MathUtils.degToRad(25), 0, sx > 0 ? 0.18 : -0.18);
        upperBody.add(earbud, cord);
      }
      // Left arm (+X) holding black smartphone over table
      const armL = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.036, 0.25, 14), topMat);
      armL.position.set(0.20, shoulderY - 0.09, 0.08);
      armL.rotation.x = THREE.MathUtils.degToRad(-45);
      const forearmL = new THREE.Mesh(new THREE.CylinderGeometry(0.034, 0.03, 0.22, 14), skinMat);
      forearmL.position.set(0.11, shoulderY - 0.12, 0.24);
      forearmL.rotation.set(THREE.MathUtils.degToRad(-80), 0, THREE.MathUtils.degToRad(-28));
      const phone = new THREE.Mesh(new THREE.BoxGeometry(0.075, 0.14, 0.01), darkPropMat);
      phone.position.set(0.05, shoulderY - 0.09, 0.31);
      phone.rotation.x = THREE.MathUtils.degToRad(-40);

      // Right arm (-X) holding piece of roti over steel plate
      const armR = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.036, 0.25, 14), topMat);
      armR.position.set(-0.20, shoulderY - 0.09, 0.08);
      armR.rotation.x = THREE.MathUtils.degToRad(-45);
      const forearmR = new THREE.Mesh(new THREE.CylinderGeometry(0.034, 0.03, 0.22, 14), skinMat);
      forearmR.position.set(-0.11, shoulderY - 0.13, 0.24);
      forearmR.rotation.set(THREE.MathUtils.degToRad(-78), 0, THREE.MathUtils.degToRad(22));
      const rotiBit = new THREE.Mesh(
        new THREE.CylinderGeometry(0.035, 0.035, 0.008, 12),
        new THREE.MeshStandardMaterial({ color: hexColor('#d8b278'), roughness: 0.9 })
      );
      rotiBit.position.set(-0.05, shoulderY - 0.11, 0.30);

      upperBody.add(armL, forearmL, phone, armR, forearmR, rotiBit);
    } else if (cid === 'CHAR_RIA') {
      // Ria: Ponytail + right hand near poha plate + left hand secretly holding smartphone under the table edge
      const ponytail = new THREE.Mesh(new THREE.SphereGeometry(0.07, 16, 14), hairMat);
      ponytail.scale.set(0.75, 1.35, 0.85);
      ponytail.position.set(0, headY - 0.02, -0.13);
      ponytail.rotation.x = THREE.MathUtils.degToRad(28);

      // Right arm (-X) resting on table near yellow poha plate
      const armR = new THREE.Mesh(new THREE.CylinderGeometry(0.036, 0.032, 0.23, 14), topMat);
      armR.position.set(-0.18, shoulderY - 0.09, 0.07);
      armR.rotation.x = THREE.MathUtils.degToRad(-45);
      const forearmR = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.028, 0.21, 14), skinMat);
      forearmR.position.set(-0.12, shoulderY - 0.13, 0.22);
      forearmR.rotation.set(THREE.MathUtils.degToRad(-82), 0, THREE.MathUtils.degToRad(18));

      // Left arm (+X) reaching down secretly under the table edge with glowing smartphone
      const armL = new THREE.Mesh(new THREE.CylinderGeometry(0.036, 0.032, 0.24, 14), topMat);
      armL.position.set(0.18, shoulderY - 0.11, 0.04);
      armL.rotation.x = THREE.MathUtils.degToRad(-18);
      const forearmL = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.028, 0.21, 14), skinMat);
      forearmL.position.set(0.14, seatY + 0.14, 0.18);
      forearmL.rotation.set(THREE.MathUtils.degToRad(-68), 0, THREE.MathUtils.degToRad(-18));
      const riaPhone = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.13, 0.01), darkPropMat);
      riaPhone.position.set(0.08, seatY + 0.13, 0.26);
      riaPhone.rotation.x = THREE.MathUtils.degToRad(-65);
      const riaScreen = new THREE.Mesh(new THREE.PlaneGeometry(0.06, 0.11), phoneScreenMat);
      riaScreen.position.set(0, 0, 0.006);
      riaPhone.add(riaScreen);

      upperBody.add(ponytail, armR, forearmR, armL, forearmL, riaPhone);
    }
  } else {
    // Standing characters: Mother (CHAR_MOTHER) & Delivery Man (CHAR_DELIVERY_MAN)
    const shoulderY = h - 0.30;
    const headY = h - 0.12;
    for (const sx of [-0.085, 0.085]) {
      const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.048, 0.78, 14), bottomMat);
      leg.position.set(sx, 0.42, 0);
      const foot = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.05, 0.20), skinMat);
      foot.position.set(sx, 0.03, 0.05);
      root.add(leg, foot);
    }
    const tunicLen = cid === 'CHAR_MOTHER' ? 0.68 : 0.48;
    const tunicY = cid === 'CHAR_MOTHER' ? h - 0.60 : h - 0.52;
    const torso = new THREE.Mesh(new THREE.BoxGeometry(0.33, tunicLen, 0.21), topMat);
    torso.position.set(0, tunicY, 0);
    torso.castShadow = true;
    const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.044, 0.046, 0.09, 14), skinMat);
    neck.position.set(0, shoulderY + 0.04, 0);
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.102, 20, 16), skinMat);
    head.scale.set(0.92, 1.12, 1.0);
    head.position.set(0, headY, 0.01);
    const hair = new THREE.Mesh(new THREE.SphereGeometry(0.105, 18, 14), hairMat);
    hair.position.set(0, headY + 0.02, -0.01);
    upperBody.add(torso, neck, head, hair);

    if (cid === 'CHAR_MOTHER') {
      const bun = new THREE.Mesh(new THREE.SphereGeometry(0.055, 14, 12), hairMat);
      bun.position.set(0, headY - 0.02, -0.11);
      const dupattaMat = new THREE.MeshStandardMaterial({ color: hexColor('#54483e'), roughness: 0.85 });
      const dupatta = new THREE.Mesh(new THREE.BoxGeometry(0.36, 0.30, 0.23), dupattaMat);
      dupatta.position.set(0, shoulderY - 0.12, 0.01);
      upperBody.add(bun, dupatta);
    } else {
      const parcelMat = createWoodShaderMaterial('#b08252', 8.0);
      const parcel = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.18, 0.24), parcelMat);
      parcel.position.set(0, shoulderY - 0.22, 0.22);
      upperBody.add(parcel);
    }
  }

  // Register subtle 60 FPS breathing node
  state.animatedRigNodes.push({ group: upperBody, phase: Math.random() * Math.PI * 2 });

  // Floating 3D Name Tag
  const shortName = label.split('(')[0].trim();
  const sprite = makeLabelSprite(shortName);
  sprite.position.set(root.position.x, pose === 'sitting' ? 1.42 : h + 0.18, root.position.z);
  labelGroup.add(sprite);

  return root;
}

// ------------------------------------------------------------------------------------------------- Scene Lighting
function buildLight(l) {
  const col = hexColor(l.color, '#fff5e1');
  let light;
  if (l.kind === 'sun') {
    light = new THREE.DirectionalLight(col, Math.min(l.intensity, 3.2));
    light.position.set(...l.position);
    light.target.position.set(...(l.target || [0, 0.8, 0]));
    light.castShadow = true;
    light.shadow.mapSize.set(2048, 2048);
    const s = 6;
    Object.assign(light.shadow.camera, { left: -s, right: s, top: s, bottom: -s, near: 0.2, far: 30 });
    lightGroup.add(light.target);
  } else if (l.kind === 'spot') {
    light = new THREE.SpotLight(col, Math.min(l.intensity, 4), 25, Math.PI / 4, 0.4, 1.2);
    light.position.set(...l.position);
    light.target.position.set(...(l.target || [0, 0.8, 0]));
    light.castShadow = true;
    lightGroup.add(light.target);
  } else if (l.kind === 'ambient') {
    light = new THREE.AmbientLight(col, Math.min(l.intensity, 1.4));
  } else {
    light = new THREE.PointLight(col, Math.min(l.intensity * 0.04, 2.2), 18, 1.4);
    light.position.set(...l.position);
    light.castShadow = true;
  }
  return light;
}

function getLayoutCharacters(L) {
  if (Array.isArray(L.characters) && L.characters.length) return L.characters;
  if (Array.isArray(L.mannequins) && L.mannequins.length) return L.mannequins;
  return [];
}

function rebuildScene() {
  for (const g of [setGroup, mqGroup, labelGroup, lightGroup]) {
    while (g.children.length) g.remove(g.children[0]);
  }
  state.animatedRigNodes = [];

  const L = state.layout;
  scene.background = hexColor(L.sky_color || '#181c24');
  lightGroup.add(new THREE.HemisphereLight(0xfff8eb, 0x3d352e, 0.85));

  for (const p of L.primitives || []) setGroup.add(buildPrimitive(p));
  for (const ch of getLayoutCharacters(L)) mqGroup.add(buildArticulatedCharacter(ch));
  for (const l of L.lights || []) lightGroup.add(buildLight(l));
  buildMorningGodRays();

  const ax = L.axis_of_action;
  if (ax && ax.length >= 2) {
    axisLine.geometry.dispose();
    axisLine.geometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(ax[0][0], 0.02, ax[0][2]),
      new THREE.Vector3(ax[1][0], 0.02, ax[1][2]),
    ]);
    axisLine.computeLineDistances();
    axisLine.visible = $('showAxis').checked;
  } else {
    axisLine.visible = false;
  }

  mqGroup.visible = $('showMq').checked;
  labelGroup.visible = $('showLabels').checked && $('showMq').checked;
  godRayGroup.visible = $('showGodRays').checked;
  grid.visible = $('showGrid').checked;
  $('setNotes').textContent = L.notes || '';
  select(null);
}

// ------------------------------------------------------------------------------------------------- Camera & Live View-Space Frustum Math
function focalToFov(focal, aspect) {
  const hfov = 2 * Math.atan(SENSOR_W / (2 * focal));
  return THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(hfov / 2) / aspect));
}

function applyLens() {
  const focal = +$('focal').value;
  const aspect = ASPECT[$('aspect').value];
  camera.fov = focalToFov(focal, aspect);
  camera.updateProjectionMatrix();
  $('focalOut').textContent = `${focal} mm`;
  document.querySelectorAll('.lens-presets button').forEach((b) => {
    b.classList.toggle('active', +b.dataset.lens === focal);
  });
  layoutFrame();
}

function layoutFrame() {
  const stage = $('stage');
  const W = stage.clientWidth;
  const H = stage.clientHeight;
  const aspect = ASPECT[$('aspect').value];
  let fw, fh;
  if (W / H > aspect) {
    fh = H;
    fw = H * aspect;
  } else {
    fw = W;
    fh = W / aspect;
  }
  const fi = $('frameInner');
  fi.style.width = `${fw}px`;
  fi.style.height = `${fh}px`;
  $('frameLabel').textContent = `${$('aspect').value} · ${$('focal').value} mm · ${state.activeCam || 'FREE CAM'}`;
  const vf = focalToFov(+$('focal').value, aspect);
  camera.fov =
    W / H > aspect
      ? vf
      : THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(THREE.MathUtils.degToRad(vf) / 2) * (H / Math.max(fh, 1))));
  camera.updateProjectionMatrix();
}

function cameraDescriptor() {
  const dir = new THREE.Vector3();
  camera.getWorldDirection(dir);
  const dist = state.fly ? 1.8 : camera.position.distanceTo(controls.target);
  const look = camera.position.clone().add(dir.multiplyScalar(dist));
  return {
    position: [+camera.position.x.toFixed(3), +camera.position.y.toFixed(3), +camera.position.z.toFixed(3)],
    look_at: [+look.x.toFixed(3), +look.y.toFixed(3), +look.z.toFixed(3)],
    focal_mm: +$('focal').value,
    roll_deg: +$('roll').value,
    aspect_ratio: $('aspect').value,
  };
}

function computeLiveFrustumSummary() {
  if (!state.layout) return 'View-Space 3D Frustum: —';
  const chars = getLayoutCharacters(state.layout);
  if (!chars.length) return 'View-Space 3D Frustum: no characters';
  const c = cameraDescriptor();
  const [cx, , cz] = c.position;
  const [tx, , tz] = c.look_at;
  let vx = tx - cx;
  let vz = tz - cz;
  const vLen = Math.hypot(vx, vz) || 1;
  vx /= vLen;
  vz /= vLen;
  const rx = -vz;
  const rz = vx;

  const items = [];
  for (const ch of chars) {
    const [px, , pz] = ch.position;
    const dx = px - cx;
    const dz = pz - cz;
    const depth = dx * vx + dz * vz;
    if (depth <= 0.12 || depth > 3.4) continue;
    const horiz = dx * rx + dz * rz;
    const normX = horiz / Math.max(depth, 0.35);
    const hZone = normX < -0.15 ? 'LEFT' : normX > 0.15 ? 'RIGHT' : 'CENTER';
    const dZone = depth < 1.15 ? 'OTS FG' : depth < 2.2 ? 'MID' : 'BG';
    const shortName = (ch.label || ch.character_id || '').split('(')[0].trim();
    items.push({ normX, text: `${shortName} [${dZone} ${hZone}]` });
  }
  items.sort((a, b) => a.normX - b.normX);
  return items.length
    ? `3D Frustum Lock (L → R):  ${items.map((i) => i.text).join('   →   ')}`
    : '3D Frustum Lock: no characters in immediate frustum';
}

function applyCamera(c) {
  $('focal').value = c.focal_mm || 35;
  $('aspect').value = c.aspect_ratio || '16:9';
  $('roll').value = c.roll_deg || 0;
  $('rollOut').textContent = `${c.roll_deg || 0}°`;
  camera.position.set(...c.position);
  const lookAt = c.look_at || c.target || [0, 1.0, 0];
  controls.target.set(...lookAt);
  camera.lookAt(controls.target);
  $('camH').value = c.position[1].toFixed(2);
  applyLens();
  controls.update();
}

function updateCamInfo() {
  const c = cameraDescriptor();
  const d = Math.hypot(c.look_at[0] - c.position[0], c.look_at[1] - c.position[1], c.look_at[2] - c.position[2]);
  $('camInfo').textContent =
    `pos [${c.position.map((v) => v.toFixed(2)).join(', ')}] → look [${c.look_at.map((v) => v.toFixed(2)).join(', ')}]\n` +
    `height ${c.position[1].toFixed(2)} m · dist ${d.toFixed(2)} m · lens ${c.focal_mm}mm`;
  if (document.activeElement !== $('camH')) $('camH').value = c.position[1].toFixed(2);
  $('frustumHud').textContent = computeLiveFrustumSummary();
}

// ------------------------------------------------------------------------------------------------- Selection & Character/Prop Editing
function select(obj) {
  if (selectionBox) {
    scene.remove(selectionBox);
    selectionBox = null;
  }
  state.selected = obj;
  if (!obj) {
    $('selId').textContent = 'none';
    return;
  }
  selectionBox = new THREE.BoxHelper(obj, 0xf0a84a);
  scene.add(selectionBox);

  if (obj.userData.isCharacter) {
    const ch = obj.userData.mq;
    $('selId').textContent = `${ch.character_id || ch.id} — ${ch.label}`;
    [$('px').value, $('py').value, $('pz').value] = ch.position;
    [$('sx').value, $('sy').value, $('sz').value] = [0.4, ch.height_m || 1.68, 0.4];
    $('ry').value = ch.facing_deg || 0;
  } else if (obj.userData.prim) {
    const p = obj.userData.prim;
    $('selId').textContent = `${p.id} — ${p.label || p.kind}`;
    [$('px').value, $('py').value, $('pz').value] = p.position;
    [$('sx').value, $('sy').value, $('sz').value] = p.size;
    $('ry').value = p.rotation_y_deg || 0;
  }
}

canvas.addEventListener('pointerdown', (e) => {
  pointer.down = [e.clientX, e.clientY];
});
canvas.addEventListener('pointerup', (e) => {
  if (!pointer.down || Math.hypot(e.clientX - pointer.down[0], e.clientY - pointer.down[1]) > 5) return;
  const r = canvas.getBoundingClientRect();
  pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);

  // Check characters first, then room primitives
  const charHits = raycaster.intersectObjects(mqGroup.children, true);
  if (charHits.length) {
    let curr = charHits[0].object;
    while (curr && !curr.userData.isCharacter) curr = curr.parent;
    if (curr) {
      select(curr);
      return;
    }
  }
  const hit = raycaster.intersectObjects(setGroup.children, false)[0];
  select(hit ? hit.object : null);
});

$('applyEdit').onclick = () => {
  const m = state.selected;
  if (!m) return;
  if (m.userData.isCharacter) {
    const ch = m.userData.mq;
    ch.position = [+$('px').value, +$('py').value, +$('pz').value];
    ch.facing_deg = +$('ry').value;
    state.dirty = true;
    rebuildScene();
    status(`Moved character ${ch.label} — click "Save Set Layout" to persist`);
  } else if (m.userData.prim) {
    const p = m.userData.prim;
    p.position = [+$('px').value, +$('py').value, +$('pz').value];
    p.size = [Math.max(0.01, +$('sx').value), Math.max(0.01, +$('sy').value), Math.max(0.01, +$('sz').value)];
    p.rotation_y_deg = +$('ry').value;
    state.dirty = true;
    rebuildScene();
    status(`Updated block ${p.id} — click "Save Set Layout" to persist`);
  }
};

$('dupBlock').onclick = () => {
  const m = state.selected;
  if (!m || !m.userData.prim) return;
  const p = JSON.parse(JSON.stringify(m.userData.prim));
  let n = 2;
  while (state.layout.primitives.some((q) => q.id === `${m.userData.prim.id}_${n}`)) n++;
  p.id = `${m.userData.prim.id}_${n}`;
  p.position[0] += p.size[0] + 0.15;
  p.structural = false;
  state.layout.primitives.push(p);
  state.dirty = true;
  rebuildScene();
};

$('delBlock').onclick = () => {
  const m = state.selected;
  if (!m || !m.userData.prim) return;
  state.layout.primitives = state.layout.primitives.filter((q) => q !== m.userData.prim);
  state.dirty = true;
  rebuildScene();
};

$('saveSet').onclick = async () => {
  if (state.standalone) {
    status('Standalone mode: layout updated in memory');
    return;
  }
  const r = await api(`/api/set/${state.env}`, 'PUT', state.layout);
  if (r.ok) {
    state.dirty = false;
    status(`Set saved (v${r.version})`);
  } else {
    status(`Save failed: ${r.error}`);
  }
};

// ------------------------------------------------------------------------------------------------- Camera List (Safe DOM creation without innerHTML)
async function refreshCameras() {
  let cams = state.layout.cameras || [];
  if (!state.standalone && state.env) {
    const fetched = await api(`/api/set/${state.env}/cameras`);
    if (Array.isArray(fetched)) {
      cams = fetched;
      state.layout.cameras = cams;
    }
  }
  const ul = $('camList');
  ul.replaceChildren();
  for (const c of cams) {
    const li = document.createElement('li');
    if (c.name === state.activeCam) li.classList.add('active');

    const nameSpan = document.createElement('span');
    nameSpan.className = 'name';
    nameSpan.textContent = c.name;
    li.appendChild(nameSpan);

    const descSpan = document.createElement('span');
    descSpan.className = 'desc';
    descSpan.textContent = `${c.focal_mm || 35}mm · ${c.aspect_ratio || '16:9'}${c.note ? ' · ' + c.note : ''}`;
    li.appendChild(descSpan);

    if (c.locked) {
      const lockTag = document.createElement('span');
      lockTag.className = 'tag lock';
      lockTag.textContent = 'locked';
      li.appendChild(lockTag);
    }
    if (c.last_render) {
      const okTag = document.createElement('span');
      okTag.className = 'tag ok';
      okTag.textContent = 'rendered';
      li.appendChild(okTag);
    }
    if (!c.locked && !state.standalone) {
      const delBtn = document.createElement('button');
      delBtn.type = 'button';
      delBtn.title = 'Delete camera';
      delBtn.textContent = '✕';
      delBtn.onclick = (ev) => {
        ev.stopPropagation();
        deleteCamera(c.name);
      };
      li.appendChild(delBtn);
    }

    li.onclick = () => {
      state.activeCam = c.name;
      $('camName').value = c.name;
      $('camNote').value = c.note || '';
      applyCamera(c);
      refreshCameras();
      status(`Cut to camera ${c.name} (${c.focal_mm || 35}mm)`);
    };
    ul.appendChild(li);
  }
}

async function saveCamera() {
  const name = $('camName').value.trim() || `CAM_${(state.layout.cameras || []).length + 1}`;
  $('camName').value = name;
  const cam = { name, note: $('camNote').value.trim(), ...cameraDescriptor() };
  if (state.standalone) {
    const idx = state.layout.cameras.findIndex((x) => x.name === name);
    if (idx >= 0) state.layout.cameras[idx] = cam;
    else state.layout.cameras.push(cam);
    state.activeCam = name;
    await refreshCameras();
    status(`Saved camera ${name} in browser`);
    return cam;
  }
  const r = await api(`/api/set/${state.env}/cameras`, 'POST', cam);
  if (r.error) {
    status(`Camera not saved: ${r.error}`);
    return null;
  }
  state.activeCam = name;
  await refreshCameras();
  status(`Camera ${name} saved to set.json`);
  return r;
}

async function deleteCamera(name) {
  const r = await api(`/api/set/${state.env}/cameras/${name}`, 'DELETE');
  if (r.error) status(r.error);
  else {
    if (state.activeCam === name) state.activeCam = null;
    refreshCameras();
  }
}
$('saveCam').onclick = saveCamera;

// ------------------------------------------------------------------------------------------------- Clean Offscreen 3D Render (Hides Labels/Grid/Helpers Automatically)
const depthMat = new THREE.MeshDepthMaterial({ depthPacking: THREE.BasicDepthPacking });

function renderCleanBuffers(camSpec) {
  const aspect = ASPECT[camSpec.aspect_ratio] || 16 / 9;
  const W = aspect >= 1 ? RENDER_LONG_EDGE : Math.round(RENDER_LONG_EDGE * aspect);
  const H = aspect >= 1 ? Math.round(RENDER_LONG_EDGE / aspect) : RENDER_LONG_EDGE;
  const rc = new THREE.PerspectiveCamera(focalToFov(camSpec.focal_mm, aspect), aspect, 0.05, 250);
  rc.position.set(...camSpec.position);
  rc.lookAt(new THREE.Vector3(...camSpec.look_at));
  rc.rotateZ(THREE.MathUtils.degToRad(camSpec.roll_deg || 0));
  rc.updateMatrixWorld();

  const target = new THREE.WebGLRenderTarget(W, H, { samples: 4 });
  const prevHelper = helperGroup.visible;
  const prevLabels = labelGroup.visible;
  const prevGodRays = godRayGroup.visible;
  helperGroup.visible = false;
  labelGroup.visible = false;
  godRayGroup.visible = false;
  if (selectionBox) selectionBox.visible = false;

  const capturePass = (overrideMat) => {
    scene.overrideMaterial = overrideMat;
    renderer.setRenderTarget(target);
    renderer.render(scene, rc);
    const buf = new Uint8Array(W * H * 4);
    renderer.readRenderTargetPixels(target, 0, 0, W, H, buf);
    renderer.setRenderTarget(null);
    scene.overrideMaterial = null;
    return buf;
  };

  const colorBuf = capturePass(null);
  const depthBuf = capturePass(depthMat);

  helperGroup.visible = prevHelper;
  labelGroup.visible = prevLabels;
  godRayGroup.visible = prevGodRays;
  if (selectionBox) selectionBox.visible = true;
  target.dispose();

  return { colorBuf, depthBuf, W, H };
}

function pixelsToDataUrl(buf, W, H, isDepth) {
  const c = document.createElement('canvas');
  c.width = W;
  c.height = H;
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(W, H);
  for (let y = 0; y < H; y++) {
    const src = (H - 1 - y) * W * 4;
    const dst = y * W * 4;
    for (let x = 0; x < W * 4; x += 4) {
      if (isDepth) {
        const v = 255 - buf[src + x];
        img.data[dst + x] = img.data[dst + x + 1] = img.data[dst + x + 2] = v;
        img.data[dst + x + 3] = 255;
      } else {
        img.data[dst + x] = buf[src + x];
        img.data[dst + x + 1] = buf[src + x + 1];
        img.data[dst + x + 2] = buf[src + x + 2];
        img.data[dst + x + 3] = 255;
      }
    }
  }
  ctx.putImageData(img, 0, 0);
  return c.toDataURL('image/png');
}

$('downloadCleanPng').onclick = () => {
  const camSpec = cameraDescriptor();
  const camName = ($('camName').value.trim() || state.activeCam || 'FPV_SHOT').replace(/[^a-zA-Z0-9_-]/g, '_');
  const { colorBuf, W, H } = renderCleanBuffers(camSpec);
  const dataUrl = pixelsToDataUrl(colorBuf, W, H, false);
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = `${camName}_3d_clean.png`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  status(`Downloaded clean 3D plate ${camName}_3d_clean.png (${W}x${H}, zero wireframes/labels)`);
};

async function renderReference() {
  const cam = await saveCamera();
  if (!cam) return;
  if (state.standalone) {
    $('downloadCleanPng').click();
    return;
  }
  const btn = $('renderCam');
  btn.disabled = true;
  status('Rendering clean 1920p 3D blockout + depth pass to AGY Studio…');
  try {
    const { colorBuf, depthBuf, W, H } = renderCleanBuffers(cam);
    const body = {
      blockout_png_b64: pixelsToDataUrl(colorBuf, W, H, false).split(',')[1],
      depth_png_b64: pixelsToDataUrl(depthBuf, W, H, true).split(',')[1],
      width: W,
      height: H,
      renderer: `three.js r${THREE.REVISION} procedural-glsl`,
      visible_mannequins: $('showMq').checked
        ? mqGroup.children.filter((g) => g.visible).map((g) => g.userData.mq.character_id || g.userData.mq.id)
        : [],
    };
    const r = await api(`/api/set/${state.env}/render/${cam.name}`, 'POST', body);
    if (r.error) status(`Render failed: ${r.error}`);
    else {
      status(`Rendered ${cam.name} → ${r.render_dir}`);
      refreshCameras();
    }
  } finally {
    btn.disabled = false;
  }
}
$('renderCam').onclick = renderReference;

// ------------------------------------------------------------------------------------------------- FPV Walk / Fly Mode + Orbit Mode
function setFly(on) {
  state.fly = on;
  $('flyMode').checked = on;
  $('btnFlyMode').classList.toggle('active', on);
  $('btnOrbitMode').classList.toggle('active', !on);
  controls.enabled = !on;
  if (on) {
    const d = new THREE.Vector3();
    camera.getWorldDirection(d);
    controls.target.copy(camera.position).add(d.multiplyScalar(2.0));
  }
  status(
    on
      ? 'FPV Walk/Fly Mode active: WASD move · Q/E down/up · Drag mouse to look · Press F for Orbit'
      : 'Cinema Orbit Mode active: Drag to orbit · Right-drag to pan · Scroll to dolly'
  );
}
$('btnOrbitMode').onclick = () => setFly(false);
$('btnFlyMode').onclick = () => setFly(true);

let lookDrag = null;
const euler = new THREE.Euler(0, 0, 0, 'YXZ');
canvas.addEventListener('pointerdown', (e) => {
  if (state.fly && e.button === 0) lookDrag = [e.clientX, e.clientY];
});
window.addEventListener('pointerup', () => {
  lookDrag = null;
});
canvas.addEventListener('pointermove', (e) => {
  if (!state.fly || !lookDrag) return;
  euler.setFromQuaternion(camera.quaternion);
  euler.y -= (e.clientX - lookDrag[0]) * 0.0035;
  euler.x -= (e.clientY - lookDrag[1]) * 0.0035;
  euler.x = Math.max(-1.45, Math.min(1.45, euler.x));
  camera.quaternion.setFromEuler(euler);
  lookDrag = [e.clientX, e.clientY];
});

window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key === 'f' || e.key === 'F') {
    setFly(!state.fly);
    return;
  }
  if (e.key === ' ') {
    e.preventDefault();
    saveCamera();
    return;
  }
  state.keys.add(e.key.toLowerCase());
});
window.addEventListener('keyup', (e) => state.keys.delete(e.key.toLowerCase()));

function flyStep(dt) {
  if (!state.fly) return;
  const speed = (state.keys.has('shift') ? 3.6 : 1.45) * dt;
  const fwd = new THREE.Vector3();
  const right = new THREE.Vector3();
  camera.getWorldDirection(fwd);
  right.crossVectors(fwd, camera.up).normalize();
  if (state.keys.has('w') || state.keys.has('arrowup')) camera.position.addScaledVector(fwd, speed);
  if (state.keys.has('s') || state.keys.has('arrowdown')) camera.position.addScaledVector(fwd, -speed);
  if (state.keys.has('a') || state.keys.has('arrowleft')) camera.position.addScaledVector(right, -speed);
  if (state.keys.has('d') || state.keys.has('arrowright')) camera.position.addScaledVector(right, speed);
  if (state.keys.has('e')) camera.position.y += speed;
  if (state.keys.has('q')) camera.position.y = Math.max(0.2, camera.position.y - speed);
}

// ------------------------------------------------------------------------------------------------- UI Wiring
$('focal').oninput = applyLens;
document.querySelectorAll('.lens-presets button').forEach((btn) => {
  btn.onclick = () => {
    $('focal').value = btn.dataset.lens;
    applyLens();
  };
});
$('aspect').onchange = applyLens;
$('roll').oninput = () => {
  $('rollOut').textContent = `${$('roll').value}°`;
};
$('camH').onchange = () => {
  const dy = +$('camH').value - camera.position.y;
  camera.position.y += dy;
  if (!state.fly) controls.target.y += dy;
};
$('showMq').onchange = (e) => {
  mqGroup.visible = e.target.checked;
  labelGroup.visible = e.target.checked && $('showLabels').checked;
};
$('showLabels').onchange = (e) => {
  labelGroup.visible = e.target.checked && $('showMq').checked;
};
$('showGodRays').onchange = (e) => {
  godRayGroup.visible = e.target.checked;
};
$('showGrid').onchange = (e) => {
  grid.visible = e.target.checked;
};
$('showAxis').onchange = (e) => {
  axisLine.visible = e.target.checked && !!(state.layout && state.layout.axis_of_action);
};
$('showStill').onchange = (e) => {
  const im = $('stillOverlay');
  im.style.display = e.target.checked ? 'block' : 'none';
  if (e.target.checked && !state.standalone) im.src = `/api/set/${state.env}/still?${Date.now()}`;
};
$('setSelect').onchange = (e) => loadSet(e.target.value);

function status(t) {
  $('status').textContent = t;
}

async function api(url, method = 'GET', body) {
  const r = await fetch(url, {
    method,
    headers: body ? { 'content-type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  try {
    return await r.json();
  } catch {
    return { error: `${r.status} ${r.statusText}` };
  }
}

async function loadSet(env) {
  state.env = env;
  state.activeCam = null;
  if (!state.standalone) history.replaceState(null, '', `?env=${env}`);
  const L = state.standalone ? window.__INLINE_SET_LAYOUT__ : await api(`/api/set/${env}`);
  if (L.error) {
    status(L.error);
    return;
  }
  state.layout = L;
  rebuildScene();
  await refreshCameras();
  const preferred =
    (L.cameras || []).find((c) => c.name === 'OTS_RIA_TO_FATHER_RISHI') || (L.cameras || [])[0];
  if (preferred) {
    state.activeCam = preferred.name;
    applyCamera(preferred);
    $('camName').value = preferred.name;
    $('camNote').value = preferred.note || '';
  }
  refreshCameras();
  const charCount = getLayoutCharacters(L).length;
  status(`${env} · ${L.primitives.length} procedural blocks · ${charCount} articulated 3D characters · ${L.cameras.length} cameras`);
}

async function boot() {
  if (state.standalone) {
    const L = window.__INLINE_SET_LAYOUT__;
    const sel = $('setSelect');
    sel.replaceChildren();
    const o = document.createElement('option');
    o.value = L.environment_id;
    o.textContent = `${L.environment_id} — ${L.title || 'Standalone 3D FPV'}`;
    sel.appendChild(o);
    await loadSet(L.environment_id);
    return;
  }
  const info = await api('/api/sets');
  state.sets = info.sets || [];
  state.projectAspect = info.aspect_ratio;
  const sel = $('setSelect');
  sel.replaceChildren();
  for (const s of state.sets) {
    const o = document.createElement('option');
    o.value = s.env_id;
    o.textContent = `${s.env_id} (${s.cameras} cams)`;
    sel.appendChild(o);
  }
  if (!state.sets.length) {
    status('No 3D sets yet — run: set3d build <ENV_ID>');
    return;
  }
  const want = new URLSearchParams(location.search).get('env');
  const env = state.sets.find((s) => s.env_id === (want || '').toUpperCase())?.env_id || state.sets[0].env_id;
  sel.value = env;
  await loadSet(env);
}

// ------------------------------------------------------------------------------------------------- 60 FPS Render Loop
function resize() {
  const stage = $('stage');
  const W = stage.clientWidth;
  const H = stage.clientHeight;
  renderer.setSize(W, H, false);
  camera.aspect = W / Math.max(H, 1);
  layoutFrame();
}
window.addEventListener('resize', resize);
resize();

let last = performance.now();
function tick(now) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;

  // FPS counter update
  state.fpsFrames++;
  if (now - state.fpsLastTime >= 600) {
    const fps = Math.min(60, Math.round((state.fpsFrames * 1000) / (now - state.fpsLastTime)));
    $('fpsBadge').textContent = `${fps} FPS · GLSL`;
    state.fpsFrames = 0;
    state.fpsLastTime = now;
  }

  // Subtle 60 FPS procedural character breathing & volumetric sunlight shimmer
  const t = now * 0.001;
  if (state.godRayMat) state.godRayMat.uniforms.uTime.value = t;
  if ($('idleAnim') && $('idleAnim').checked) {
    for (const node of state.animatedRigNodes) {
      node.group.position.y = Math.sin(t * 2.1 + node.phase) * 0.004;
      node.group.rotation.z = Math.sin(t * 1.1 + node.phase) * 0.006;
    }
  }

  flyStep(dt);
  if (!state.fly) controls.update();
  if (selectionBox) selectionBox.update();
  renderer.render(scene, camera);
  updateCamInfo();
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
boot();
