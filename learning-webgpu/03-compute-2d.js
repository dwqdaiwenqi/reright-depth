// ─────────────────────────────────────────────────────────
// 为什么图像处理要用 @workgroup_size(8, 8) 而不是 @workgroup_size(64)?
//
// 图片是二维的，每个像素有 x 坐标和 y 坐标。
//
// 用 @workgroup_size(64)（一维）：
//   线程编号是 0, 1, 2, 3 ... 65535（一条线）
//   你要自己算：x = id.x % width，y = id.x / width（麻烦且容易出错）
//
// 用 @workgroup_size(8, 8)（二维）：
//   线程天然就是 (x=0..7, y=0..7) 的网格
//   id.x 直接就是像素列，id.y 直接就是像素行，一一对应，更自然
//
// 视觉理解：
//
//   图片（256×256 像素）被切成 32×32 个工作组，每组负责 8×8 的小方块：
//
//   ┌──────────────────────────────────┐
//   │ [组0,0] [组1,0] [组2,0] ...     │
//   │ [组0,1] [组1,1] [组2,1] ...     │  每个 [组x,y] 内部有
//   │ [组0,2] [组1,2] [组2,2] ...     │  8×8 = 64 个线程
//   │  ...                            │  每个线程处理1个像素
//   └──────────────────────────────────┘
//
//   dispatchWorkgroups(32, 32) × workgroup_size(8, 8) = 65536 个线程
//   正好覆盖 256×256 = 65536 个像素，完美对齐，没有浪费
//
// ─────────────────────────────────────────────────────────
// WebGL 为什么不用管 workgroup_size？
//
// WebGL fragment shader 里你完全感知不到线程，GPU 驱动自动替你处理了。
// 但 GPU 内部其实也有 workgroup，驱动偷偷选了一个尺寸，比如 8×4。
//
// 重要：8×4 不是整个屏幕的大小！！
// 8×4 是"一个工作组"负责的像素方块，整个屏幕被切成很多这样的方块：
//
//   1920×1080 屏幕，驱动选 workgroup_size = 8×4（32线程=1个subgroup）：
//
//   x方向：1920 / 8 = 240 个工作组
//   y方向：1080 / 4 = 270 个工作组
//   总工作组数 = 240 × 270 = 64800 个
//   总线程数   = 64800 × 32 = 2,073,600 = 1920×1080 ✓ 正好覆盖所有像素
//
//   ┌──────────────────────────────────────────────┐
//   │[8×4][8×4][8×4][8×4]...(240个)               │ ↑
//   │[8×4][8×4][8×4][8×4]...                      │ 270行
//   │[8×4][8×4][8×4][8×4]...                      │ ↓
//   └──────────────────────────────────────────────┘
//        1920像素宽
//
// WebGPU compute 只是把这个控制权交给了你，让你自己填这个数字。
// WebGL 是驱动黑盒帮你填，你看不到也改不了。
//
// ─────────────────────────────────────────────────────────
// 为什么驱动选 8×4，而不是 32×1 或 1×32？
// 这和你的 Apple Silicon subgroupSize = 32 直接相关。
//
// 联系：8 × 4 = 32 = 正好一个 subgroup，全部满载，零浪费
//
//   GPU 硬件每次必须以 32 线程为一批调度（subgroup）
//   只要线程总数 = 32 的倍数，就没有空转浪费
//   所以 32×1 / 16×2 / 8×4 / 4×8 在利用率上都一样
//
// 但为什么选接近正方形的 8×4，而不是细长的 32×1？
// 原因是：缓存局部性
//
//   图片在内存里是一行一行存的：
//   [行0的像素0][行0的像素1]...[行0的像素255][行1的像素0]...
//
//   32×1（一行32个像素）：
//     线程只读了1行，只利用了水平方向的缓存
//     ← 上下相邻行的像素完全没有预取
//
//   8×4（接近正方形的小方块）：
//     同时覆盖4行×8列，水平和垂直方向都相邻
//     ← GPU 缓存预取同时命中横向和纵向，缓存利用率更高
//
//   ┌────────────┐  ← 32×1：只有这一行，缓存命中率低
//   │████████████│
//   └────────────┘
//
//   ┌────────┐  ← 8×4：正方形小方块，两个方向都利用缓存
//   │████████│
//   │████████│
//   │████████│
//   │████████│
//   └────────┘
//
// 所以你自己写 WebGPU 图像处理时：
//   @workgroup_size(8, 8)  = 64线程 = 2个subgroup，正方形 ✓ 常用
//   @workgroup_size(16,16) = 256线程 = 8个subgroup，正方形 ✓ 更常用
//   @workgroup_size(64, 1) = 64线程 = 2个subgroup，细长  ✗ 缓存差
//   线程数相同，但正方形比细长形缓存命中率更高
// ─────────────────────────────────────────────────────────

function log(msg, type = 'info') {
  const el = document.getElementById('log');
  const d  = document.createElement('div');
  d.className = type; d.textContent = msg;
  el.appendChild(d);
}

// ── 图片尺寸（必须是 workgroup_size 的倍数，否则边缘需要越界检查）──
const W = 256;
const H = 256;

// ── 在 inputCanvas 上画一张彩色渐变图（模拟"原始图片"）──────────────
const inputCanvas = document.getElementById('inputCanvas');
const ctx2d = inputCanvas.getContext('2d');

// 生成彩色渐变：X方向控制红色，Y方向控制绿色，斜向控制蓝色
const imageData = ctx2d.createImageData(W, H);
for (let y = 0; y < H; y++) {
  for (let x = 0; x < W; x++) {
    const i = (y * W + x) * 4;
    imageData.data[i + 0] = x;                    // R：左→右 红色渐变
    imageData.data[i + 1] = y;                    // G：上→下 绿色渐变
    imageData.data[i + 2] = 255 - (x + y) / 2;   // B：斜向蓝色渐变
    imageData.data[i + 3] = 255;                  // A：不透明
  }
}
ctx2d.putImageData(imageData, 0, 0);

// 把画好的图片数据取出来，准备上传给 GPU
// imageData.data 是 Uint8ClampedArray，格式：[R,G,B,A, R,G,B,A, ...]
const inputPixels = new Uint8Array(imageData.data.buffer);

// ── WebGPU 初始化 ────────────────────────────────────────────────────
if (!navigator.gpu) {
  log('你的浏览器不支持 WebGPU，请用 Chrome 113+', 'warn');
  throw new Error('no webgpu');
}
const adapter = await navigator.gpu.requestAdapter();
const device  = await adapter.requestDevice();
log('✓ WebGPU 初始化完成');


// ─────────────────────────────────────────────────────────
// Compute Shader（WGSL）
// ─────────────────────────────────────────────────────────
const shaderCode = /* wgsl */ `

// 图片宽高（通过 uniform buffer 传入，避免硬编码）
struct Params {
  width:  u32,
  height: u32,
}
@group(0) @binding(0) var<uniform>             params    : Params;
@group(0) @binding(1) var<storage, read>       inputBuf  : array<u32>; // 输入像素（每个u32=1个RGBA像素）
@group(0) @binding(2) var<storage, read_write> outputBuf : array<u32>; // 输出像素

// ── 核心：二维 workgroup_size ───────────────────────────────
// @workgroup_size(8, 8)：
//   每个工作组是 8×8 的线程网格，共 64 个线程
//   对比一维 @workgroup_size(64)：线程数一样，但坐标天然是二维的
//
//   对应 JS 侧：dispatchWorkgroups(W/8, H/8, 1)
//     第1个参数 = x方向派多少组（256/8 = 32组）
//     第2个参数 = y方向派多少组（256/8 = 32组）
//     第3个参数 = z方向（不用，写1）
//
//   总线程数 = 32×32组 × 8×8线程/组 = 1024 × 64 = 65536
//            = 256×256 图片像素数，完美覆盖
@compute @workgroup_size(8, 8)
fn main(
  // global_invocation_id 现在是真正的 (x, y) 坐标：
  //   .x = 当前线程负责的像素列（0 ~ width-1）
  //   .y = 当前线程负责的像素行（0 ~ height-1）
  //
  // 和一维的区别：
  //   一维 @workgroup_size(64)：只有 id.x，要手动算 y = id.x / width
  //   二维 @workgroup_size(8,8)：id.x 和 id.y 直接就是像素坐标，直观
  @builtin(global_invocation_id) id: vec3<u32>
) {
  let x = id.x;
  let y = id.y;

  // 越界检查（图片尺寸不是 workgroup_size 倍数时，边缘会有多余线程）
  // 本例 256 正好是 8 的倍数，这行实际不会触发，但保留是好习惯
  if (x >= params.width || y >= params.height) { return; }

  // ── 关键：二维坐标 → 一维数组下标 ──────────────────────────
  // 内存里的像素数组是一维的（一行一行存储），需要手动换算：
  //   第 y 行的起点 = y * width
  //   再加上列偏移 x
  //   就像二维数组 arr[y][x] → 一维 arr[y * width + x]
  let idx = y * params.width + x;

  // ── 解包像素（u32 → RGBA 四个分量）────────────────────────
  // inputBuf 里每个 u32 存了一个像素，字节顺序：R在低位，A在高位
  // 内存：[R][G][B][A] → u32 = R | G<<8 | B<<16 | A<<24
  let px = inputBuf[idx];
  let r = f32( px        & 0xFFu) / 255.0; // 取低8位 = R
  let g = f32((px >>  8) & 0xFFu) / 255.0; // 取第9~16位 = G
  let b = f32((px >> 16) & 0xFFu) / 255.0; // 取第17~24位 = B

  // ── 灰度公式：不是简单平均，而是按人眼敏感度加权 ──────────
  // 人眼对绿色最敏感（0.587），对蓝色最不敏感（0.114）
  // 这个公式来自 ITU-R BT.601 标准
  let gray = u32((0.299 * r + 0.587 * g + 0.114 * b) * 255.0);

  // ── 重新打包：灰度图 R=G=B=gray，A保持255 ─────────────────
  outputBuf[idx] = gray | (gray << 8u) | (gray << 16u) | (0xFFu << 24u);
}
`;

const shaderModule = device.createShaderModule({ code: shaderCode });


// ── 按钮点击：执行 GPU 计算 ──────────────────────────────────────────
document.getElementById('btn').onclick = async () => {
  document.getElementById('log').innerHTML = '';
  document.getElementById('outputCanvas').style.filter = 'brightness(0.3)';
  log('▶ 开始 GPU 灰度转换...', 'info');

  const t0 = performance.now();
  const byteSize = W * H * 4; // 256×256 像素 × 4字节/像素 = 262144 字节

  // ── 创建 GPU Buffer ──────────────────────────────────────────────
  // 1. Uniform Buffer：传图片宽高给 shader
  //    uniform 是只读的小数据（不超过64KB），适合传参数
  //    （对比 storage：可读写，适合大数据）
  const paramsBuffer = device.createBuffer({
    size:  16,   // Struct { width: u32, height: u32 } = 8字节，但 uniform 要求最少16字节对齐
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([W, H, 0, 0]));

  // 2. 输入 buffer：上传原图像素数据
  const inputBuffer = device.createBuffer({
    size:  byteSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(inputBuffer, 0, inputPixels);

  // 3. 输出 buffer：GPU 写入灰度结果
  const outputBuffer = device.createBuffer({
    size:  byteSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // 4. 回读 buffer：把 GPU 结果读回 CPU
  const readbackBuffer = device.createBuffer({
    size:  byteSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  log(`✓ 创建 4 个 GPU Buffer（共 ${(byteSize * 3 / 1024).toFixed(0)} KB 显存）`);

  // ── 创建计算管线 ─────────────────────────────────────────────────
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: shaderModule, entryPoint: 'main' },
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramsBuffer  } }, // 宽高参数
      { binding: 1, resource: { buffer: inputBuffer   } }, // 原图像素
      { binding: 2, resource: { buffer: outputBuffer  } }, // 灰度结果
    ],
  });

  // ── 录制并提交命令 ───────────────────────────────────────────────
  const encoder = device.createCommandEncoder();
  const pass    = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);

  // ── 关键：二维 dispatch ────────────────────────────────────────
  // 一维例子用的是 dispatchWorkgroups(N)
  // 二维图像用 dispatchWorkgroups(x方向组数, y方向组数)
  //
  // 256 × 256 图片，@workgroup_size(8, 8)：
  //   x方向：256 / 8 = 32 个工作组
  //   y方向：256 / 8 = 32 个工作组
  //   总工作组 = 32 × 32 = 1024 个，每组 64 线程，共 65536 线程
  //   正好 = 256 × 256 像素数，一线程处理一像素，完美覆盖
  const wgX = Math.ceil(W / 8); // x方向工作组数 = 32
  const wgY = Math.ceil(H / 8); // y方向工作组数 = 32
  pass.dispatchWorkgroups(wgX, wgY);
  log(`✓ dispatchWorkgroups(${wgX}, ${wgY}) → ${wgX * wgY} 个工作组 × 64 线程 = ${wgX * wgY * 64} 线程`);

  pass.end();

  encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, byteSize);
  device.queue.submit([encoder.finish()]);

  // ── 读回结果并显示 ───────────────────────────────────────────────
  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const result = new Uint8Array(readbackBuffer.getMappedRange());

  // 把 GPU 输出的像素数据写到 outputCanvas 上显示
  const outputCanvas = document.getElementById('outputCanvas');
  const outCtx = outputCanvas.getContext('2d');
  const outImageData = outCtx.createImageData(W, H);
  outImageData.data.set(result);
  outCtx.putImageData(outImageData, 0, 0);
  outputCanvas.style.filter = ''; // 取消暗化

  readbackBuffer.unmap();

  const ms = (performance.now() - t0).toFixed(2);
  log(`✓ GPU 灰度转换完成，耗时 ${ms}ms（处理了 ${W * H} 个像素）`, 'ok');
};
