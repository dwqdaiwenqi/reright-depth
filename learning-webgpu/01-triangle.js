// ─────────────────────────────────────────────────────────
// 第一步：初始化 WebGPU
// WebGL 用 canvas.getContext('webgl')
// WebGPU 用 navigator.gpu 这套新 API
// ─────────────────────────────────────────────────────────

// 检查浏览器是否支持 WebGPU
if (!navigator.gpu) {
  document.getElementById('msg').textContent = '你的浏览器不支持 WebGPU，请用 Chrome 113+';
  throw new Error('WebGPU not supported');
}

// adapter = 代表你的 GPU 硬件
const adapter = await navigator.gpu.requestAdapter();
// device  = 用来创建所有 WebGPU 资源的核心对象（相当于 WebGL 的 gl）
const device  = await adapter.requestDevice();

// 拿到 canvas 的 WebGPU 绘图上下文
const canvas  = document.getElementById('c');
const context = canvas.getContext('webgpu');

// 配置 context：告诉它用什么像素格式
const format = navigator.gpu.getPreferredCanvasFormat();
context.configure({ device, format });

const info = adapter.info ?? await adapter.requestAdapterInfo?.() ?? {};
document.getElementById('msg').textContent = `GPU: ${info.description || info.device || '已连接'}`;


// ─────────────────────────────────────────────────────────
// 第二步：写 Shader（WGSL 语言）
//
// WebGL 用 GLSL，WebGPU 用 WGSL（W3C 标准的新语言）
// 语法有点像 Rust，但概念和 GLSL 一样
// ─────────────────────────────────────────────────────────

const shaderCode = /* wgsl */ `

// ── Vertex Shader ──────────────────────────────────────
// @vertex 相当于 WebGL 的 vertex shader
//
// @location(0) 对标 WebGL 的 layout(location=0) in vec2 a_position;
// 这里从顶点 buffer 里逐顶点读取 xy 坐标（对标 vertexAttribPointer）
// -> @builtin(position) 输出裁剪空间坐标（和 GLSL 的 gl_Position 一样）

@vertex
fn vs_main(@location(0) pos: vec2f) -> @builtin(position) vec4f {
  // 从 buffer 读来的 xy，补上 Z=0 W=1
  return vec4f(pos, 0.0, 1.0);
}


// ── Fragment Shader ────────────────────────────────────
// @fragment 相当于 WebGL 的 fragment shader
// @location(0) 表示输出到第 0 号颜色附件（就是 canvas）

@fragment
fn fs_main() -> @location(0) vec4f {
  // 返回紫色：RGBA = (0.6, 0.4, 1.0, 1.0)
  return vec4f(0.6, 0.4, 1.0, 1.0);
}
`;

// 把 WGSL 代码编译成 GPU 可用的 ShaderModule
const shaderModule = device.createShaderModule({ code: shaderCode });


// ─────────────────────────────────────────────────────────
// 第三步：创建顶点 Buffer（对标 WebGL 的 VBO）
//
// WebGL：
//   const buf = gl.createBuffer()
//   gl.bindBuffer(gl.ARRAY_BUFFER, buf)
//   gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW)
//
// WebGPU：不需要 bind，创建时直接上传数据，用 usage 声明用途
// ─────────────────────────────────────────────────────────

// 三角形三个顶点的 XY 坐标（裁剪空间，范围 -1 到 1）
// 和 WebGL 一样用 Float32Array 打包成字节
const vertices = new Float32Array([
   0.0,  0.5,   // 顶部中心
  -0.5, -0.5,   // 左下
   0.5, -0.5,   // 右下
]);

// 对标 WebGL 的 gl.createBuffer() + gl.bufferData()
// VERTEX    → 可作为顶点输入（对标 ARRAY_BUFFER）
// COPY_DST  → 允许从 CPU 写入（对标 STATIC_DRAW 上传路径）
const vertexBuffer = device.createBuffer({
  size:  vertices.byteLength,
  usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});
// 把 CPU 数据写入 GPU buffer（对标 gl.bufferData）
device.queue.writeBuffer(vertexBuffer, 0, vertices);


// ─────────────────────────────────────────────────────────
// 第四步：创建 Render Pipeline（渲染管线）
//
// WebGL：你需要手动 createProgram / attachShader / linkProgram
//        再用 vertexAttribPointer + enableVertexAttribArray 描述顶点格式
//        这些信息记录在 VAO 里
//
// WebGPU：全部写在 createRenderPipeline 的 vertex.buffers 里
//         没有单独的 VAO 对象，格式信息烘进 Pipeline
// ─────────────────────────────────────────────────────────

const pipeline = device.createRenderPipeline({
  layout: 'auto',   // 让 WebGPU 自动推断 bind group layout

  // 顶点阶段配置
  vertex: {
    module:     shaderModule,
    entryPoint: 'vs_main',   // 对应 shader 里 @vertex fn vs_main

    // 对标 WebGL 的 vertexAttribPointer + enableVertexAttribArray
    // 以及 VAO 里记录的顶点格式信息
    buffers: [{
      arrayStride: 2 * 4,       // 每个顶点占多少字节（2个float × 4字节）
                                // 对标 vertexAttribPointer 的 stride
      attributes: [{
        shaderLocation: 0,      // 对标 layout(location=0) / attrib index
        offset: 0,              // 对标 vertexAttribPointer 的 offset
        format: 'float32x2',    // 对标 vertexAttribPointer 的 type=FLOAT, size=2
      }],
    }],
  },

  // 片段阶段配置
  fragment: {
    module:     shaderModule,
    entryPoint: 'fs_main',   // 对应 shader 里 @fragment fn fs_main
    targets: [{ format }],   // 输出格式要和 canvas 一致
  },

  // 图元类型：三角形列表（每 3 个顶点画一个三角形）
  primitive: { topology: 'triangle-list' },
});


// ─────────────────────────────────────────────────────────
// 第五步：每帧渲染
//
// WebGL：gl.clear() + gl.drawArrays()
// WebGPU：录制命令（Command Buffer）→ 提交给 GPU
//
// WebGPU 的命令是先"录制"后"提交"，而不是立即执行
// 这样 GPU 可以批量优化执行顺序
// ─────────────────────────────────────────────────────────

function render() {
  // CommandEncoder：命令录制器，相当于一个待执行任务的清单
  const encoder = device.createCommandEncoder();

  // RenderPassEncoder：开始一次渲染操作
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      // 渲染目标：canvas 当前帧的纹理
      view:       context.getCurrentTexture().createView(),
      // loadOp: 'clear' → 渲染前先清空画布（相当于 gl.clear）
      loadOp:     'clear',
      clearValue: { r: 0.06, g: 0.06, b: 0.1, a: 1 },  // 深色背景
      // storeOp: 'store' → 渲染结果保存到纹理
      storeOp:    'store',
    }],
  });

  pass.setPipeline(pipeline);
  // 绑定顶点 buffer，对标 WebGL 的：
  //   gl.bindVertexArray(vao)  或
  //   gl.bindBuffer(gl.ARRAY_BUFFER, buf)
  // 第一个参数 0 = slot 0，对应 pipeline.vertex.buffers[0]
  pass.setVertexBuffer(0, vertexBuffer);
  pass.draw(3);                // 画 3 个顶点 = 1 个三角形
  pass.end();                  // 结束录制

  // 把录制好的命令提交给 GPU 执行
  device.queue.submit([encoder.finish()]);
}

// 渲染循环
function loop() {
  render();
  requestAnimationFrame(loop);
}
loop();
