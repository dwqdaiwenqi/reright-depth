// ─────────────────────────────────────────────────────────
// 计算管线 vs 渲染管线的最大区别：
//
// 渲染管线：输入顶点 → 输出画面（你能看到的像素）
// 计算管线：输入数据 → 输出数据（Buffer，看不见的显存）
//
// 计算管线没有顶点、没有片段、不画任何东西
// 只是让 GPU 的几千个核心同时跑同一段程序处理不同数据
// ─────────────────────────────────────────────────────────

function log(msg, type = 'info') {
  const el = document.getElementById('log');
  const d  = document.createElement('div');
  d.className = type; d.textContent = msg;
  el.appendChild(d);
}

// 显示输入数组
const INPUT_DATA = new Float32Array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]);
const inputEl = document.getElementById('inputNums');
INPUT_DATA.forEach(v => {
  const s = document.createElement('span');
  s.className = 'num';
  s.textContent = v;
  inputEl.appendChild(s);
});

// 检查 WebGPU 支持
if (!navigator.gpu) {
  log('你的浏览器不支持 WebGPU，请用 Chrome 113+', 'warn');
  throw new Error('no webgpu');
}

const adapter = await navigator.gpu.requestAdapter();
const device  = await adapter.requestDevice();
log('✓ WebGPU 初始化完成');


// ─────────────────────────────────────────────────────────
// Compute Shader（WGSL）
//
// 核心概念：workgroup（工作组）
//   GPU 把任务分成很多 workgroup 并行执行
//   @workgroup_size(N) 表示每个 workgroup 有 N 个线程
//   @builtin(global_invocation_id).x 是当前线程的全局编号
//
// 类比：
//   你有 16 道数学题
//   不是一道一道算，而是叫 16 个人同时每人算一道
//   每个人通过"我是第几个人"知道该算哪道题
// ─────────────────────────────────────────────────────────

const shaderCode = /* wgsl */ `

// 声明两个 storage buffer（可读写的显存区域）
// @group(0) @binding(0) 是"插槽"编号，CPU 侧绑定数据时用
@group(0) @binding(0) var<storage, read>       inputBuf  : array<f32>;
@group(0) @binding(1) var<storage, read_write> outputBuf : array<f32>;

// @compute 标记这是一个 compute shader（没有顶点/片段，只处理数据）
//
// @workgroup_size(32) 的意思：
//   每个 workgroup（工作组）里有 32 个线程同时启动并行跑这个函数
//   不是"每次启动32个"，而是"每个工作组固定是32个线程的编队"
//
//   JS 里 dispatchWorkgroups(N) 决定启动几个工作组：
//     总线程数 = N（工作组数） × 32（每组线程数）
//     例：dispatchWorkgroups(4) → 4组 × 32线程 = 128个线程同时跑
//
// 为什么写 32（你 Apple Silicon 的 subgroupSize）：
//   GPU 硬件每次必须以 32 线程为一批来调度（这是硬件固定的，改不了）
//   写 32  → 每组正好1批，32线程全部干活 ✓
//   写 64  → 每组2批，64线程全部干活 ✓
//   写 200 → 每组需要7批（7×32=224），但只有200个线程
//            第7批只来了8个，剩24个线程位置空着 ✗ 浪费
//   规则：必须是 32 的倍数，否则最后一批有空位浪费
//
// 为什么不能远超数据量：
//   workgroup_size(512)，数据只有16个元素：
//     启动 512 个线程，线程0~15读数据干活
//     线程16~511：执行 if (i >= 16) return，什么都没干
//     GPU 为这512个线程全部分配了寄存器、全部调度了，只有16个有产出
//     利用率 3%，就像雇了512个工人搬16块砖，496人站着看
//     结果不会错，但 GPU 资源白白浪费了
@compute @workgroup_size(32)
fn main(
  // global_invocation_id：当前线程在整个任务里的全局坐标（x, y, z三维）
  // 这里数据是一维数组，只用 .x
  // 例：dispatchWorkgroups(1) + workgroup_size(32) → 线程编号 0,1,2,...,31
  @builtin(global_invocation_id) id: vec3<u32>
) {
  let i = id.x;   // 当前线程负责处理第 i 个元素，每个线程的 i 都不同

  // 越界检查：启动的线程数量可能多于数组长度
  // 例：数据16个，workgroup_size(32) → 启动32线程 → 线程16~31没有对应数据
  // 没有这行检查，线程16~31会越界写入非法显存，导致崩溃或数据错误
  if (i >= arrayLength(&inputBuf)) { return; }

  // 核心计算：每个线程独立处理自己负责的那个元素，互不干扰
  // 线程0：outputBuf[0] = inputBuf[0] * 2.0
  // 线程1：outputBuf[1] = inputBuf[1] * 2.0
  // 线程2：outputBuf[2] = inputBuf[2] * 2.0  ...同时并行
  outputBuf[i] = inputBuf[i] * 2.0;
}
`;

const shaderModule = device.createShaderModule({ code: shaderCode });


document.getElementById('btn').onclick = async () => {
  document.getElementById('log').innerHTML = '';
  document.getElementById('outputNums').innerHTML = '';
  log('▶ 开始 GPU 计算...', 'info');

  const t0 = performance.now();

  // ───────────────────────────────────────────────────────
  // 第一步：在 GPU 显存里创建 Buffer
  //
  // CPU 内存 ≠ GPU 显存，数据需要显式上传
  // Buffer 是显存里的一段连续内存，相当于 GPU 版的 ArrayBuffer
  // ───────────────────────────────────────────────────────

  const byteSize = INPUT_DATA.byteLength;  // 16个 float32 = 64字节

  // 输入 buffer：CPU 写入，GPU 只读
  const inputBuffer = device.createBuffer({
    size:  byteSize,
    usage: GPUBufferUsage.STORAGE |      // 可以被 shader 读取（storage buffer）
           GPUBufferUsage.COPY_DST,      // 可以从 CPU 拷贝数据进来
  });

  // 输出 buffer：GPU 写入结果
  const outputBuffer = device.createBuffer({
    size:  byteSize,
    usage: GPUBufferUsage.STORAGE |      // shader 写入结果
           GPUBufferUsage.COPY_SRC,      // 可以把结果拷贝到 CPU 可读的 buffer
  });

  // 中转 buffer：用于把 GPU 结果读回 CPU
  // GPU 的 STORAGE buffer 不能直接被 CPU 读，需要先拷贝到 MAP_READ buffer
  const readbackBuffer = device.createBuffer({
    size:  byteSize,
    usage: GPUBufferUsage.COPY_DST |     // 接收从 outputBuffer 拷贝过来的数据
           GPUBufferUsage.MAP_READ,      // 允许 CPU 读取（map 到 CPU 内存）
  });

  log(`✓ 创建了 3 个 GPU Buffer（共 ${byteSize * 3} 字节显存）`);

  // 把 CPU 数据上传到 GPU inputBuffer
  device.queue.writeBuffer(inputBuffer, 0, INPUT_DATA);
  log('✓ 输入数据已上传到 GPU 显存');


  // ───────────────────────────────────────────────────────
  // 第二步：创建计算管线
  // ───────────────────────────────────────────────────────

  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module:     shaderModule,
      entryPoint: 'main',
    },
  });

  // BindGroup：把 CPU 侧的 Buffer 绑定到 shader 里声明的插槽
  // shader 里 @binding(0) → inputBuffer
  // shader 里 @binding(1) → outputBuffer
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuffer  } },
      { binding: 1, resource: { buffer: outputBuffer } },
    ],
  });

  log('✓ 计算管线和 BindGroup 创建完成');


  // ───────────────────────────────────────────────────────
  // 第三步：录制并提交命令
  // ───────────────────────────────────────────────────────

  const encoder = device.createCommandEncoder();

  // 开始一个 compute pass（对应渲染管线里的 render pass）
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);       // 绑定数据

  // dispatchWorkgroups(N)：启动 N 个 workgroup
  // 每个 workgroup 有 32 个线程（对应 shader 里的 @workgroup_size(32)）
  // 总线程数 = N × 32，必须 >= 数据长度才能覆盖所有元素
  //
  // 数据16个，workgroup_size=32：
  //   Math.ceil(16 / 32) = 1 个 workgroup
  //   启动 32 个线程，线程0~15处理数据，线程16~31越界return
  //
  // 如果数据是100个：
  //   Math.ceil(100 / 32) = 4 个 workgroup → 128个线程 → 线程100~127越界return
  const workgroupCount = Math.ceil(INPUT_DATA.length / 32);
  pass.dispatchWorkgroups(workgroupCount);
  pass.end();

  // 计算完成后，把结果从 outputBuffer 拷贝到 readbackBuffer
  encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, byteSize);

  // 提交所有命令，GPU 开始执行
  device.queue.submit([encoder.finish()]);
  log('✓ 命令已提交 GPU，等待执行完毕...');


  // ───────────────────────────────────────────────────────
  // 第四步：把结果读回 CPU
  //
  // GPU 是异步的，submit 之后 GPU 才开始跑
  // mapAsync 会等待 GPU 完成 + buffer 可读
  // ───────────────────────────────────────────────────────

  await readbackBuffer.mapAsync(GPUMapMode.READ);

  // getMappedRange 拿到 CPU 可读的 ArrayBuffer
  const resultArray = new Float32Array(readbackBuffer.getMappedRange());
  const ms = (performance.now() - t0).toFixed(2);
  log(`✓ GPU 计算完成，耗时 ${ms}ms`, 'ok');

  // 显示结果
  const outputEl = document.getElementById('outputNums');
  resultArray.forEach(v => {
    const s = document.createElement('span');
    s.className = 'num result';
    s.textContent = v;
    outputEl.appendChild(s);
  });

  // 验证结果正确性
  let correct = true;
  for (let i = 0; i < INPUT_DATA.length; i++) {
    if (resultArray[i] !== INPUT_DATA[i] * 2) { correct = false; break; }
  }
  log(correct ? '✓ 结果验证正确（每个元素 × 2）' : '✗ 结果有误', correct ? 'ok' : 'warn');

  // 用完之后 unmap，归还给 GPU
  readbackBuffer.unmap();
};
