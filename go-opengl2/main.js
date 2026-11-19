// 用 WebGL2 对 30-3.png 做重光照：
//   1. 加载 30-3.png
//   2. POST 到 gen_normal_dav2.py 的 /depth 接口，拿到深度图
//   3. 把"原图 + 深度图"交给 shader，鼠标移动 = 移动光源
//
// 运行方式见 index.html 同目录下的说明（需要起一个本地静态服务器，不能直接双击打开）

const IMAGE_URL = './25-3.png';
const DEPTH_SERVER = 'http://localhost:7860'; // python tools/gen_normal_dav2.py --serve
const MODEL_SIZE = 'small';                   // small | base | large

// ── 着色器 ──────────────────────────────────────────────────────
const vertexSrc = /*glsl*/`#version 300 es
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUv;
out vec2 vUv;
void main() {
    vUv = aUv;
    gl_Position = vec4(aPos, 0.0, 1.0);
}`;

const fragmentSrc = /*glsl*/`#version 300 es
precision highp float;
in vec2 vUv;

uniform sampler2D uImage;      // 原图
uniform sampler2D uDepth;      // 深度图（白=近，黑=远）
uniform vec3      uLightPos;   // 光源位置（UV 空间 xy + 高度 z）
uniform float     uIntensity;  // 光强
uniform float     uSpecular;   // 高光强度
uniform float     uDepthScale; // 深度对法线的影响强度
uniform float     uAmbient;    // 环境光
uniform vec2      uResolution; // 画布像素尺寸
uniform int       uShadowMode; // 0=无 1=衰减 2=硬阴影 3=RayMarch软 4=RayMarch+抖动

out vec4 fragColor;

float getDepth(vec2 uv) {
    return texture(uDepth, uv).r;
}

void main() {
    // 一个像素在 UV 空间里的大小
    vec2 px = 1.0 / uResolution;

    // 1. 采样上下左右四个邻居的深度
    float dL = getDepth(vUv + vec2(-px.x, 0.0));
    float dR = getDepth(vUv + vec2(+px.x, 0.0));
    float dU = getDepth(vUv + vec2(0.0, +px.y));
    float dD = getDepth(vUv + vec2(0.0, -px.y));

    // 2. 中心差分求偏导（斜率）
    float dzdx = (dR - dL) / (2.0 * px.x);
    float dzdy = (dU - dD) / (2.0 * px.y);

    // 3. 曲面两条切线，叉积得到法线
    vec3 Tx = vec3(1.0, 0.0, dzdx * uDepthScale);
    vec3 Ty = vec3(0.0, 1.0, dzdy * uDepthScale);
    vec3 normal = normalize(cross(Tx, Ty));

    // 当前像素的 3D 位置、指向光源的方向
    float depth   = getDepth(vUv);
    vec3  pos     = vec3(vUv, depth);
    vec3  toLight = normalize(uLightPos - pos);

    // 4. Wrapped Lambert 漫反射
    float wrap    = 0.3;
    float diffuse = clamp((dot(normal, toLight) + wrap) / (1.0 + wrap), 0.0, 1.0);

    // 5. Blinn-Phong 高光（正交相机，视线固定朝 +Z）
    vec3  viewDir = vec3(0.0, 0.0, 1.0);
    vec3  halfVec = normalize(toLight + viewDir);
    float spec    = pow(max(dot(normal, halfVec), 0.0), 32.0) * uSpecular;

    // 6. 阴影
    vec2  dir2D  = uLightPos.xy - vUv;
    float dist   = length(dir2D);
    vec2  dir2Dn = dist > 1e-6 ? dir2D / dist : vec2(0.0);
    float shadow = 1.0;

    if (uShadowMode == 0) {
        shadow = 1.0;
    } else if (uShadowMode == 1) {
        // 距离平方衰减
        float attenuation = 1.0 / (1.0 + 4.0 * dist * dist);
        shadow = mix(0.3, 1.0, attenuation);
    } else if (uShadowMode == 2) {
        // 硬阴影：只采样一个点
        float pDepth = getDepth(vUv + dir2Dn * px * 4.0);
        shadow = (pDepth < depth - 0.02) ? 0.5 : 1.0;
    } else if (uShadowMode == 3) {
        // Ray March 软阴影
        vec2 stepV = dir2Dn * px * 2.0;
        vec2 cur   = vUv + stepV;
        for (int i = 0; i < 16; i++) {
            if (length(cur - vUv) >= dist) break;
            if (getDepth(cur) < depth - 0.02) shadow -= 0.04;
            cur += stepV;
        }
        shadow = clamp(shadow, 0.5, 1.0);
    } else {
        // Ray March + 抖动
        float jitter = fract(sin(dot(vUv, vec2(12.9898, 78.233))) * 43758.5453);
        vec2  stepV  = dir2Dn * px * 2.0;
        vec2  cur    = vUv + stepV * (0.5 + jitter * 0.5);
        for (int i = 0; i < 16; i++) {
            if (length(cur - vUv) >= dist) break;
            if (getDepth(cur) < depth - 0.02) shadow -= 0.04;
            cur += stepV;
        }
        shadow = clamp(shadow, 0.5, 1.0);
    }

    // 7. 合成。画布默认是"预乘 alpha"，所以 rgb 要乘 a，透明处才不会漏色
    vec4  color = texture(uImage, vUv);
    float light = uAmbient + (diffuse + spec) * uIntensity * shadow;
    fragColor   = vec4(color.rgb * light * color.a, color.a);
}`;

// ── 主流程 ──────────────────────────────────────────────────────
async function main() {
  const canvas = document.getElementById('c');
  const status = document.getElementById('status');
  const gl = canvas.getContext('webgl2');
  if (!gl) { status.textContent = '浏览器不支持 WebGL2'; return; }

  try {
    // 1. 加载原图。先拿到 Blob：既能做纹理，也能原样 POST 给服务器
    status.textContent = '加载图片…';
    const imageBlob = await (await fetch(IMAGE_URL)).blob();
    // imageOrientation: 'flipY'：图片第 0 行在顶部，纹理 v=0 在底部，要翻转。
    // 注意：对 ImageBitmap 来说，gl.pixelStorei(UNPACK_FLIP_Y_WEBGL) 不生效，必须在这里翻
    const imageBitmap = await createImageBitmap(imageBlob, { imageOrientation: 'flipY' });

    // 画布像素尺寸 = 图片尺寸，CSS 里再缩小显示
    canvas.width = imageBitmap.width;
    canvas.height = imageBitmap.height;

    // 2. 请求深度图
    status.textContent = `请求 ${DEPTH_SERVER}/depth …（首次要加载模型，可能要等一会）`;
    const t0 = performance.now();
    const depthBitmap = await fetchDepth(imageBlob);
    const sec = ((performance.now() - t0) / 1000).toFixed(2);
    status.textContent = `深度图已获取（${sec}s，${depthBitmap.width}×${depthBitmap.height}）· 移动鼠标改变光源`;

    // 3. 初始化 GL
    const program = newProgram(gl, vertexSrc, fragmentSrc);
    const vao = newFullscreenQuad(gl);
    const imageTex = newTexture(gl, imageBitmap);
    const depthTex = newTexture(gl, depthBitmap);

    const u = {};
    for (const name of ['uImage', 'uDepth', 'uLightPos', 'uIntensity', 'uSpecular',
                        'uDepthScale', 'uAmbient', 'uResolution', 'uShadowMode']) {
      u[name] = gl.getUniformLocation(program, name);
    }

    // 4. 光照参数：滑块和鼠标
    const params = { intensity: 1.5, height: 1.5, ambient: 0.2, depthScale: 0.08, specular: 0.4, shadowMode: 3 };
    const light = { x: 0.5, y: 0.5 };

    bindSlider('sIntensity', 'intensity', params);
    bindSlider('sHeight', 'height', params);
    bindSlider('sAmbient', 'ambient', params);
    bindSlider('sDepthScale', 'depthScale', params);
    bindSlider('sSpecular', 'specular', params);
    document.getElementById('sShadow').addEventListener('change', e => {
      params.shadowMode = parseInt(e.target.value);
    });

    canvas.addEventListener('mousemove', e => {
      const r = canvas.getBoundingClientRect();
      light.x = (e.clientX - r.left) / r.width;
      light.y = 1.0 - (e.clientY - r.top) / r.height; // UV 原点在左下，要翻转 y
    });

    // 5. 渲染循环
    function frame() {
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);

      gl.useProgram(program);
      gl.uniform3f(u.uLightPos, light.x, light.y, params.height);
      gl.uniform2f(u.uResolution, canvas.width, canvas.height);
      gl.uniform1f(u.uIntensity, params.intensity);
      gl.uniform1f(u.uSpecular, params.specular);
      gl.uniform1f(u.uDepthScale, params.depthScale);
      gl.uniform1f(u.uAmbient, params.ambient);
      gl.uniform1i(u.uShadowMode, params.shadowMode);

      // 纹理单元 0 放原图，1 放深度图；sampler 里填的是"单元号"
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, imageTex);
      gl.uniform1i(u.uImage, 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, depthTex);
      gl.uniform1i(u.uDepth, 1);

      gl.bindVertexArray(vao);
      gl.drawArrays(gl.TRIANGLES, 0, 6); // 全屏四边形 = 2 个三角形 = 6 个顶点

      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);

  } catch (err) {
    console.error(err);
    status.textContent = '✗ ' + err.message;
  }
}

// ── 请求深度图：POST {server}/depth，字段 image + size ─────────────
// 响应：{ "depth": "<base64 png>" } 或 { "error": "..." }
async function fetchDepth(imageBlob) {
  const form = new FormData();
  form.append('image', imageBlob, '30-3.png');
  form.append('size', MODEL_SIZE);

  let res;
  try {
    res = await fetch(`${DEPTH_SERVER}/depth`, { method: 'POST', body: form });
  } catch (e) {
    throw new Error(`连不上 ${DEPTH_SERVER}，深度服务启动了吗？（python tools/gen_normal_dav2.py --serve）`);
  }
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);

  const data = await res.json();
  if (data.error) throw new Error('服务端错误: ' + data.error);

  // base64 → Blob → ImageBitmap
  const bytes = Uint8Array.from(atob(data.depth), c => c.charCodeAt(0));
  return createImageBitmap(new Blob([bytes], { type: 'image/png' }), { imageOrientation: 'flipY' });
}

// ── 全屏四边形：每个顶点 x, y, u, v，v 向上（左下角 uv=(0,0)） ────────
function newFullscreenQuad(gl) {
  const vertices = new Float32Array([
    -1, -1, 0, 0,
     1, -1, 1, 0,
     1,  1, 1, 1,
    -1, -1, 0, 0,
     1,  1, 1, 1,
    -1,  1, 0, 1,
  ]);

  const vao = gl.createVertexArray();
  const vbo = gl.createBuffer();
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

  const stride = 4 * 4; // 每个顶点 4 个 float
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, stride, 0);     // aPos：前 2 个
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(1, 2, gl.FLOAT, false, stride, 2 * 4); // aUv：后 2 个，偏移 8 字节
  gl.enableVertexAttribArray(1);
  return vao;
}

// ── 上传纹理 ─────────────────────────────────────────────────────
function newTexture(gl, bitmap) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return tex;
}

function bindSlider(id, key, params) {
  const slider = document.getElementById(id);
  const label = document.getElementById(id + 'Val');
  slider.addEventListener('input', () => {
    params[key] = parseFloat(slider.value);
    label.textContent = slider.value;
  });
}

// ── 编译两个着色器并链接成一个程序 ────────────────────────────────
function newProgram(gl, vertexSource, fragmentSource) {
  const vs = compileShader(gl, vertexSource, gl.VERTEX_SHADER);
  const fs = compileShader(gl, fragmentSource, gl.FRAGMENT_SHADER);

  const program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error('着色器程序链接失败: ' + gl.getProgramInfoLog(program));
  }

  gl.deleteShader(vs);
  gl.deleteShader(fs);
  return program;
}

function compileShader(gl, source, type) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error('着色器编译失败: ' + gl.getShaderInfoLog(shader));
  }
  return shader;
}

main();
