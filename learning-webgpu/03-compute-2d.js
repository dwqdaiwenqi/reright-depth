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


// ─────────────────────────────────────────────────────────────────
// Pipeline 1：灰度化 Shader
// 对标 ONNX 里的 Cast + Mul + Add（预处理标准化）
// ─────────────────────────────────────────────────────────────────
const grayscaleShader = device.createShaderModule({ code: /* wgsl */ `
struct Params { width: u32, height: u32 }
@group(0) @binding(0) var<uniform>             params    : Params;
@group(0) @binding(1) var<storage, read>       inputBuf  : array<u32>;
@group(0) @binding(2) var<storage, read_write> outputBuf : array<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let x = id.x;
  let y = id.y;
  if (x >= params.width || y >= params.height) { return; }
  let idx = y * params.width + x; // 二维坐标 → 一维下标

  let px = inputBuf[idx];
  let r = f32( px        & 0xFFu) / 255.0;
  let g = f32((px >>  8u) & 0xFFu) / 255.0;
  let b = f32((px >> 16u) & 0xFFu) / 255.0;

  // ITU-R BT.601 灰度公式，按人眼敏感度加权
  let gray = u32((0.299 * r + 0.587 * g + 0.114 * b) * 255.0);
  outputBuf[idx] = gray | (gray << 8u) | (gray << 16u) | (0xFFu << 24u);
}
`});

// ─────────────────────────────────────────────────────────────────
// Pipeline 2：增亮 Shader
// 对标 ONNX 里的 Mul（把激活值乘以一个系数）
//
// 关键：这个 shader 的 inputBuf = Pipeline1 的 outputBuf
//       数据从没有离开过 GPU 显存，不需要读回 CPU 再上传
//       这就是 ONNX 里 45 个算子能高效串联的原因
// ─────────────────────────────────────────────────────────────────
const brightnessShader = device.createShaderModule({ code: /* wgsl */ `
struct Params { width: u32, height: u32 }
@group(0) @binding(0) var<uniform>             params    : Params;
@group(0) @binding(1) var<storage, read>       inputBuf  : array<u32>; // ← Pipeline1 的输出
@group(0) @binding(2) var<storage, read_write> outputBuf : array<u32>; // ← 最终结果

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let x = id.x;
  let y = id.y;
  if (x >= params.width || y >= params.height) { return; }
  let idx = y * params.width + x;

  let px = inputBuf[idx];
  let r = f32( px        & 0xFFu);
  let g = f32((px >>  8u) & 0xFFu);
  let b = f32((px >> 16u) & 0xFFu);

  // 每个通道乘以增亮系数 1.8，超过 255 的截断（clamp）
  let factor = 1.8;
  let rr = min(u32(r * factor), 255u);
  let gg = min(u32(g * factor), 255u);
  let bb = min(u32(b * factor), 255u);

  outputBuf[idx] = rr | (gg << 8u) | (bb << 16u) | (0xFFu << 24u);
}
`});


// ── 按钮点击：串联执行两个 Pipeline ────────────────────────────────
document.getElementById('btn').onclick = async () => {
  document.getElementById('log').innerHTML = '';
  document.getElementById('midCanvas').style.filter    = 'brightness(0.3)';
  document.getElementById('outputCanvas').style.filter = 'brightness(0.3)';
  log('▶ 开始串联 GPU 处理...', 'info');

  const t0 = performance.now();
  const byteSize = W * H * 4;
  const wgX = Math.ceil(W / 8);
  const wgY = Math.ceil(H / 8);

  // ── Uniform Buffer（两个 pipeline 共用同一份宽高参数）──────────
  const paramsBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([W, H, 0, 0]));

  // ── Buffer 链：体现 Pipeline 间数据传递的核心 ───────────────────
  //
  //   inputBuffer  →[Pipeline1]→  midBuffer  →[Pipeline2]→  outputBuffer
  //                               ↑ 这个 buffer 同时是 P1 的输出和 P2 的输入
  //                               usage 必须同时包含 STORAGE（两个 pipeline 都要读写它）
  //
  // inputBuffer：原图，CPU → GPU，只写一次
  const inputBuffer = device.createBuffer({
    size:  byteSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(inputBuffer, 0, inputPixels);

  // midBuffer：Pipeline1 输出 / Pipeline2 输入，纯 GPU 内部中转
  // 对标 ONNX 里那个 128MB 的大 Buffer，存中间激活值
  // 注意：COPY_SRC 是为了后面把中间结果读回 CPU 显示，生产环境里不需要
  const midBuffer = device.createBuffer({
    size:  byteSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    //     ↑ STORAGE：Pipeline1写入 + Pipeline2读取都需要它
  });

  // outputBuffer：Pipeline2 最终输出
  const outputBuffer = device.createBuffer({
    size:  byteSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // readbackBuffer：GPU → CPU 读回，只用于在 canvas 上显示
  const midReadback = device.createBuffer({
    size:  byteSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const outReadback = device.createBuffer({
    size:  byteSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  log(`✓ 创建 6 个 GPU Buffer：input → mid → output + 2个回读`);

  // ── 创建两个 Pipeline ─────────────────────────────────────────
  const pipeline1 = device.createComputePipeline({
    layout: 'auto',
    compute: { module: grayscaleShader, entryPoint: 'main' },
    // label 对应 Inspector 里看到的名字，方便调试
    // （ONNX 里就是用 label: 'Cast' / 'Mul' / 'Add' 来标记的）
    label: 'Grayscale',
  });

  const pipeline2 = device.createComputePipeline({
    layout: 'auto',
    compute: { module: brightnessShader, entryPoint: 'main' },
    label: 'Brightness',
  });

  // ── BindGroup：把 Buffer 绑到对应的 @binding 插槽 ─────────────
  // Pipeline1 的 BindGroup：input → mid
  const bindGroup1 = device.createBindGroup({
    layout: pipeline1.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramsBuffer } },
      { binding: 1, resource: { buffer: inputBuffer  } }, // 读：原图
      { binding: 2, resource: { buffer: midBuffer    } }, // 写：灰度结果
    ],
  });

  // Pipeline2 的 BindGroup：mid → output
  // 关键：binding(1) 绑的是 midBuffer，和 Pipeline1 的 binding(2) 是同一个 Buffer
  // 这就是"Pipeline1 的输出 = Pipeline2 的输入"在代码里的体现
  const bindGroup2 = device.createBindGroup({
    layout: pipeline2.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramsBuffer } },
      { binding: 1, resource: { buffer: midBuffer    } }, // 读：Pipeline1 写入的灰度结果
      { binding: 2, resource: { buffer: outputBuffer } }, // 写：增亮结果
    ],
  });

  // ── 录制命令：一个 CommandEncoder 录制两个 ComputePass ──────────
  // 对应 Inspector 里 Record 时看到的那个命令列表：
  //   ComputePass (label: Grayscale)   ← Pipeline1
  //     setPipeline / setBindGroup / dispatchWorkgroups
  //   ComputePass (label: Brightness)  ← Pipeline2
  //     setPipeline / setBindGroup / dispatchWorkgroups
  //   copyBufferToBuffer (mid → midReadback)
  //   copyBufferToBuffer (output → outReadback)
  const encoder = device.createCommandEncoder();

  // ── Pass 1：灰度化 ─────────────────────────────────────────────
  const pass1 = encoder.beginComputePass({ label: 'Grayscale' });
  pass1.setPipeline(pipeline1);
  pass1.setBindGroup(0, bindGroup1);
  pass1.dispatchWorkgroups(wgX, wgY); // input → mid
  pass1.end();
  log(`✓ Pass1 Grayscale dispatched`);

  // ── Pass 2：增亮 ───────────────────────────────────────────────
  // GPU 保证 Pass1 执行完毕后才开始 Pass2（因为 Pass2 依赖 Pass1 的输出）
  // 不需要任何显式同步，WebGPU 的 submit 顺序就保证了这一点
  const pass2 = encoder.beginComputePass({ label: 'Brightness' });
  pass2.setPipeline(pipeline2);
  pass2.setBindGroup(0, bindGroup2);
  pass2.dispatchWorkgroups(wgX, wgY); // mid → output
  pass2.end();
  log(`✓ Pass2 Brightness dispatched`);

  // 把两个中间结果都拷贝出来，以便在 canvas 上分别显示
  encoder.copyBufferToBuffer(midBuffer,    0, midReadback, 0, byteSize);
  encoder.copyBufferToBuffer(outputBuffer, 0, outReadback, 0, byteSize);

  // 一次 submit 提交所有命令，GPU 按顺序执行
  device.queue.submit([encoder.finish()]);

  // ── 读回并显示中间结果（灰度）─────────────────────────────────
  await midReadback.mapAsync(GPUMapMode.READ);
  const midResult = new Uint8Array(midReadback.getMappedRange());
  const midCanvas = document.getElementById('midCanvas');
  const midCtx    = midCanvas.getContext('2d');
  const midImgData = midCtx.createImageData(W, H);
  midImgData.data.set(midResult);
  midCtx.putImageData(midImgData, 0, 0);
  midCanvas.style.filter = '';
  midReadback.unmap();

  // ── 读回并显示最终结果（增亮）────────────────────────────────
  await outReadback.mapAsync(GPUMapMode.READ);
  const outResult = new Uint8Array(outReadback.getMappedRange());
  const outputCanvas = document.getElementById('outputCanvas');
  const outCtx       = outputCanvas.getContext('2d');
  const outImgData   = outCtx.createImageData(W, H);
  outImgData.data.set(outResult);
  outCtx.putImageData(outImgData, 0, 0);
  outputCanvas.style.filter = '';
  outReadback.unmap();

  const ms = (performance.now() - t0).toFixed(2);
  log(`✓ 串联处理完成，耗时 ${ms}ms`, 'ok');
  log(`  midBuffer 全程留在 GPU 显存，没有经过 CPU`, 'info');
};

// ── 空的 rAF 循环，让 Inspector Immediate 模式能抓帧 ─────────────────
// Inspector 的 Immediate 模式靠 requestAnimationFrame 来判断"一帧"的边界
// 没有 rAF 循环它就不知道何时是一帧的开始和结束，抓不到任何东西
// 这个循环本身什么都不做，只是让 Inspector 能工作
// 按钮点击触发的 GPU 命令会在某一帧的 rAF 回调期间或之后 submit，
// Inspector 会把它归入最近的那一帧
(function rafLoop() { requestAnimationFrame(rafLoop); })();
