// main.go 的 JS（WebGL2）版本，结构、函数名、顺序和 main.go 一一对应
// 运行：用浏览器打开同目录下的 index.html

const vertexSrc = `#version 300 es
layout(location = 0) in vec2 aPos;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
}`;

const fragmentSrc = `#version 300 es
precision highp float;
uniform vec3 uColor; // 颜色由 CPU 传进来，每画一个三角形换一次
out vec4 fragColor;
void main() {
    fragColor = vec4(uColor, 1.0);
}`;

function main() {
  // ── 1. 创建窗口 ───────────────────────────────
  const canvas = document.getElementById('c');
  canvas.width = 640;
  canvas.height = 480;

  const gl = canvas.getContext('webgl2');
  console.log('WebGL 版本:', gl.getParameter(gl.VERSION));

  // ── 2. 编译着色器，链接成程序 ──────────────────
  const program = newProgram(gl, vertexSrc, fragmentSrc);

  // ── 3. 准备 3 个三角形 ──────────────────────────
  // 坐标范围 -1 ~ 1，原点在窗口中心，x 向右，y 向上
  //
  // VBO 和 VAO 的区别：
  //   VBO（Vertex Buffer Object）= GPU 里的一块内存，存"顶点数据本身"（一堆 float）
  //   VAO（Vertex Array Object） = 一份"读数据的配置"，记录：用哪个 VBO、
  //                                每个顶点几个 float、从哪开始读、对应 shader 的哪个 location
  //   类比：VBO 是一张存了数据的 Excel 表格，VAO 是"这张表怎么读"的说明书。
  //   画的时候只要 bindVertexArray(某个 vao)，说明书和它指向的数据就都生效了。
  //   这里每个三角形一个 VBO + 一个 VAO，画哪个就绑定哪个。
  const triangles = [
    newTriangle(gl, [-0.6, 0.4, -0.85, -0.4, -0.35, -0.4], [1.0, 0.3, 0.3]), // 左：红
    newTriangle(gl, [0.0, 0.4, -0.25, -0.4, 0.25, -0.4], [0.3, 1.0, 0.3]),   // 中：绿
    newTriangle(gl, [0.6, 0.4, 0.35, -0.4, 0.85, -0.4], [0.3, 0.5, 1.0]),    // 右：蓝
  ];
  const uColor = gl.getUniformLocation(program, 'uColor'); // 找到 shader 里 uColor 的位置

  // ── 4. 渲染循环 ───────────────────────────────
  function frame() {
    gl.clearColor(0.1, 0.1, 0.15, 1); // 背景色
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(program);
    for (const t of triangles) {
      gl.uniform3f(uColor, t.color[0], t.color[1], t.color[2]); // 设置颜色
      gl.bindVertexArray(t.vao);                                // 切换到这个三角形的"说明书"
      gl.drawArrays(gl.TRIANGLES, 0, 3);                        // 画 3 个顶点组成的三角形
    }

    requestAnimationFrame(frame); // 浏览器自动换缓冲、处理事件，所以没有 SwapBuffers / PollEvents
  }
  requestAnimationFrame(frame);
}

// 创建一个三角形：上传顶点数据到 VBO，并用 VAO 记录读取方式
function newTriangle(gl, vertices, color) {
  const vao = gl.createVertexArray(); // VAO：记录"数据怎么读"
  const vbo = gl.createBuffer();      // VBO：GPU 里存顶点数据的内存

  gl.bindVertexArray(vao); // 先绑定 VAO：之后的配置都会记录进这个 VAO
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);

  // 告诉 GPU：location=0 的 aPos，每个顶点读 2 个 float，每个顶点占 2*4 字节
  // 这条配置连同当前绑定的 VBO，一起被记录进 VAO
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 2 * 4, 0);
  gl.enableVertexAttribArray(0);

  return { vao, color };
}

// 编译两个着色器并链接成一个程序
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

  gl.deleteShader(vs); // 链接完就不需要了
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
