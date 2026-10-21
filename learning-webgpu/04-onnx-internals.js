// ═════════════════════════════════════════════════════════════════
// 你在 WebGPU Inspector 里看到的那些东西是什么？
//
// 当你点击"加载 Depth Anything V2"时，Transformers.js 加载了一个
// ONNX 模型，然后用 WebGPU 来运行它。
//
// ── 运行 AI 模型 = 在 GPU 上执行一次神经网络的前向传播（Forward Pass）──
//
// 神经网络模型
//   = 一组数学运算的有序组合（计算图）
//   = 每个运算 → 一个 Compute Shader + Pipeline
//   = 45个运算串联执行 → 得到结果
//
// 就像一条流水线工厂：
//
//   原料（图片）
//     → 工序1  Cast    清洗原料（整数转浮点）
//     → 工序2  Mul     切成标准尺寸（归一化 ÷255）
//     → 工序3  Add     去掉杂质（减均值）
//     → 工序4~42      核心加工（卷积/注意力，神经网络主体）
//     → 工序43~45     精加工（Resize/Add）
//     → 成品（深度图）
//
//   每道工序 = 1个 Pipeline
//   工序之间通过 Buffer 传递中间产品
//   全程在 GPU 显存里流转，不经过 CPU
//
// 从 WebGPU 角度描述"跑一次 AI 模型"就是：
//   按顺序 dispatch 45 个 Compute Shader
//   每个 Shader 对显存里的数据做一次数学变换
//   最后一个 Shader 的 outputData buffer 就是推理结果
//
// ── 45个 Pipeline 按神经网络层次分组 ──────────────────────────────
//
//   预处理（前几个）：把图片变成模型能处理的格式
//     Cast → Mul → Mul → Add
//
//   特征提取（中间大部分）：神经网络"理解"图片内容的核心
//     Conv → Add → Relu          ← 卷积层：提取边缘/纹理特征
//     Conv → Add → Relu          ← 这些重复组合就是卷积神经网络
//     TransposeShared             ← 调整数据维度
//     ReduceMeanShared            ← 注意力机制（Transformer）
//     Pow → Sqrt → Div            ← Layer Normalization（归一化）
//     Erf → Add → Mul             ← GELU 激活函数
//
//   输出处理（最后几个）：把神经网络输出变成深度图
//     Resize → ConvTranspose2D → Add   ← 上采样回原始尺寸
//
// ── 每个算子 = 一个 Compute Pipeline + 一个 Shader Module ──────────
// 所以：Compute Pipelines 45 = Shader Modules 45，一一对应
// ═════════════════════════════════════════════════════════════════


// ─────────────────────────────────────────────────────────────────
// 算子 1：Cast（类型转换）
//
// 在神经网络里的作用：预处理第一步，把原始图片像素格式转换
// 输入：整数像素 array<vec4<i32>>（值范围 0~255）
// 输出：浮点像素 array<vec4<f32>>（值范围 0.0~255.0，还没归一化）
//
// 为什么要转：后续所有神经网络计算都用浮点数，整数不能直接参与
// ─────────────────────────────────────────────────────────────────
const castShader = /* wgsl */ `
enable f16;        // 开启半精度浮点（16位），省显存省带宽，AI推理精度够用
enable subgroups;  // 开启线程组内通信（subgroup），后面的归约算子会用到
                   // 这两行是 ONNX runtime 的统一模板，所有 shader 都加，不管用不用

struct Uniforms { output_size: u32 }
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

// ── vec4 是性能优化：一次处理4个值 ──────────────────────────────────
// 不用 array<f32>（每次处理1个），而用 array<vec4<f32>>（每次处理4个）
// GPU 内存带宽是瓶颈，打包读取带宽利用率提升4倍
// 效果：1个线程处理4个像素分量，总线程数 = 像素数 / 4
@group(0) @binding(0) var<storage, read>       input  : array<vec4<i32>>; // 整数像素输入
@group(0) @binding(1) var<storage, read_write> output : array<vec4<f32>>; // 浮点像素输出

// ── @workgroup_size(64, 1, 1) ────────────────────────────────────
// 等价于 @workgroup_size(64)，ONNX 代码生成器统一写三个维度
// 图像处理你会写 @workgroup_size(8,8)，数组处理写 @workgroup_size(64) 就够
@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(global_invocation_id)   global_id   : vec3<u32>, // 全局线程编号，最常用
  @builtin(workgroup_id)           workgroup_id : vec3<u32>, // 工作组编号（这里没用）
  @builtin(local_invocation_index) local_idx   : u32,        // 组内线程编号（这里没用）
  @builtin(local_invocation_id)    local_id    : vec3<u32>   // 组内xyz坐标（这里没用）
  // ONNX 模板统一声明这4个，不管用不用都带着，省得每个shader单独判断
) {
  let global_idx = global_id.x;
  if (global_idx >= uniforms.output_size) { return; } // 越界检查

  // 核心逻辑只有一行：vec4<i32> → vec4<f32>，WGSL 内置类型转换
  // [255, 128, 64, 255](i32) → [255.0, 128.0, 64.0, 255.0](f32)
  // 注意：值的大小没变，只是类型变了，归一化（÷255）由下一个 Mul 做
  output[global_idx] = vec4<f32>(input[global_idx]);
}
`;


// ─────────────────────────────────────────────────────────────────
// 算子 2：Mul（逐元素相乘）
//
// 在神经网络里的作用：预处理第二步，归一化
// 输入A：Cast 输出的浮点像素 [255.0, 128.0, 64.0, ...]
// 输入B：模型权重里存的归一化系数 [1/255, 1/255, 1/255, ...]（加载模型时上传）
// 输出：归一化后的像素 [1.0, 0.50, 0.25, ...]（范围 0~1）
//
// 为什么要归一化：神经网络训练时输入就是 0~1 范围，推理时必须一致
// ─────────────────────────────────────────────────────────────────
const mulShader = /* wgsl */ `
enable f16;
enable subgroups;

struct Uniforms { vec_size: u32 }
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

// 注意：这里有两个输入 buffer，不是一个
// aData = 上一个算子的输出（动态数据，每张图片不同）
// bData = 模型权重（静态常量，加载模型时就上传到 GPU，之后不变）
@group(0) @binding(0) var<storage, read>       aData      : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       bData      : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> outputData : array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(local_invocation_index) local_idx: u32,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let global_idx = global_id.x;
  if (global_idx >= uniforms.vec_size) { return; }

  // vec4 × vec4 = 4个float同时相乘，一条指令干4份工作
  // [255.0, 128.0, 64.0, 255.0] × [1/255, 1/255, 1/255, 1/255]
  // = [1.0, 0.502, 0.251, 1.0]
  outputData[global_idx] = aData[global_idx] * bData[global_idx];
}
`;


// ─────────────────────────────────────────────────────────────────
// 算子 3：Add（带 Broadcasting 的逐元素相加）
//
// 在神经网络里的作用：预处理第三步，减去均值（标准化）
// 输入A：归一化后的像素 [1.0, 0.50, 0.25, ...]
// 输入B：每个通道的均值（形状和A不同，需要 Broadcasting 扩展）
// 输出：减去均值后的像素，分布更集中，有利于神经网络训练和推理
//
// Broadcasting 是什么？
//   A 的形状：[1, 3, 224, 224]（一张图，3通道，224×224像素）
//   B 的形状：[1, 3,   1,   1]（每个通道只有1个均值）
//   B 自动"广播"扩展，通道1的均值加到通道1所有像素，以此类推
//
// 简单 Mul 形状相同，直接 aData[i]*bData[i] 即可
// 带 Broadcasting 的 Add 形状不同，需要：
//   1. 把线性下标 i 转回 [n,c,h,w] 四维坐标（o2i_outputData 函数）
//   2. 用坐标查 A 和 B 各自对应位置（broadcastedIndicesTo*Offset 函数）
//   3. 取值相加
// 那一大坨函数就是在做这个坐标换算
// ─────────────────────────────────────────────────────────────────


// ─────────────────────────────────────────────────────────────────
// 最后一个算子 45：Add（输出层 bias 加法）
//
// 在神经网络里的作用：最后一层的偏置加法，输出最终深度值
// 输入A：神经网络预测的深度值（前44个算子计算的结果）
// 输入B：模型学到的偏置常量（只有1个标量值）
// 输出：深度图（每个像素一个 f32 深度值）
//
// 注意这里和普通 Add 不同：
//   outputData[i] = aData[i] + vec4<f32>(bData[0].x)
//                                        ↑ 只取第一个元素的x分量
//   vec4<f32>(scalar) = 把标量扩展成4个相同值的 vec4
//   相当于：每个像素的深度值都加上同一个常量 bias
//
// 这个 outputData buffer 的内容就是最终深度图：
//   Transformers.js 内部 mapAsync 读回 CPU
//   → output.depth.data
//   → 你在 index.html 里用 putImageData 画到 canvas 显示
// ─────────────────────────────────────────────────────────────────
const lastAddShader = /* wgsl */ `
enable f16;
enable subgroups;

struct Uniforms {
  vec_size: u32,
  // 下面这些 shape/strides 是 Broadcasting 用的，简单标量加法用不到
  // 但 ONNX 的 Add 模板统一带着，不管这个 Add 是不是 Broadcasting
  aData_shape:        vec4<u32>,
  aData_strides:      vec4<u32>,
  bData_shape:        vec4<u32>,
  bData_strides:      vec4<u32>,
  outputData_shape:   vec4<u32>,
  outputData_strides: vec4<u32>,
}
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

@group(0) @binding(0) var<storage, read>       aData      : array<vec4<f32>>; // 深度预测值
@group(0) @binding(1) var<storage, read>       bData      : array<vec4<f32>>; // 偏置常量
@group(0) @binding(2) var<storage, read_write> outputData : array<vec4<f32>>; // 最终深度图

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(local_invocation_index) local_idx: u32,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let global_idx = global_id.x;
  if (global_idx >= uniforms.vec_size) { return; }

  // bData[0].x = 取第一个 vec4 的 x 分量 = 整个 bData 里唯一的那个 bias 值
  // vec4<f32>(scalar) = 把这一个数复制4份，凑成 vec4，和 aData 的 vec4 对齐相加
  outputData[global_idx] = aData[global_idx] + vec4<f32>(bData[0].x);
  // 这个 outputData 就是最终输出，内容是深度图每个像素的 f32 深度值
}
`;


// ─────────────────────────────────────────────────────────────────
// Buffer 大小的意义（Inspector 里 Buffers 334 个）
//
// { "size": 134217728, "usage": "COPY_SRC | COPY_DST | STORAGE" }
//   134217728 字节 = 128 MB
//   这是中间激活值 buffer（各层计算的中间结果）
//   STORAGE = 可被 compute shader 读写（对标 02-compute 里的 STORAGE）
//   COPY_SRC | COPY_DST = 可在 GPU buffer 之间互相拷贝
//   类比工厂：这是流水线上传递"半成品"的传送带
//
// { "size": 128, "usage": "COPY_SRC | COPY_DST | UNIFORM" }
//   128 字节 = 极小，存的是传给 shader 的参数（宽高、步长、形状等）
//   就是 shader 里 struct Uniforms { vec_size, shape, strides... } 的内容
//   UNIFORM = 只读小数据，适合传参数（对标 03-compute-2d 里的 paramsBuffer）
//   类比工厂：这是每道工序的"工艺参数说明卡"
//
// 为什么有 334 个 Buffer？
//   每个算子需要：输入 buffer + 输出 buffer + uniform buffer
//   + 模型权重（每层的卷积核、偏置等常量，加载模型时上传，之后不变）
//   权重数量最多，所以 buffer 总数远超 45（pipeline数量）
// ─────────────────────────────────────────────────────────────────


// ─────────────────────────────────────────────────────────────────
// enable f16 和 enable subgroups 是什么
//
// enable f16：
//   开启半精度浮点（16位 vs 普通 float 32位）
//   显存减半、带宽减半、Apple Silicon 有专用 f16 运算单元速度翻倍
//   这就是 Transformers.js 默认 dtype:'fp16' 比 'fp32' 快的原因
//   AI 推理对精度要求不高，f16 够用
//
// enable subgroups：
//   开启 subgroup 线程间通信（你的 Apple Silicon subgroupSize=32）
//   同一个 subgroup 的32个线程可以互相读取对方的寄存器，不需要走显存
//   用于：
//     ReduceMeanShared → subgroupAdd() 求和（softmax 分母、LayerNorm）
//     Pow/Sqrt/Div     → 配合归约算法
//   不开启就要通过 var<workgroup> shared memory 间接通信，慢一点
//
// 你自己写的 shader 不需要这两个，因为：
//   - 只用标准 f32，没有 f16
//   - 每个线程独立处理，不需要线程间通信
// ─────────────────────────────────────────────────────────────────
