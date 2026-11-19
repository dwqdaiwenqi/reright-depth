package main

import (
	"fmt"
	"log"
	"runtime"

	"github.com/go-gl/gl/v4.1-core/gl"
	"github.com/go-gl/glfw/v3.3/glfw"
)

func init() {
	runtime.LockOSThread() // glfw 必须在主线程
}

// 顶点着色器：每个顶点运行一次，决定顶点在屏幕上的位置
// 注意：Go 字符串末尾要加 "\x00"，因为 OpenGL 是 C 接口，C 字符串以 0 结尾
const vertexSrc = `#version 410 core
layout(location = 0) in vec2 aPos;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
}` + "\x00"

// 片元着色器：每个像素运行一次，决定像素颜色
const fragmentSrc = `#version 410 core
uniform vec3 uColor; // 颜色由 CPU 传进来，每画一个三角形换一次
out vec4 fragColor;
void main() {
    fragColor = vec4(uColor, 1.0);
}` + "\x00"

func main() {

	// Go	Node.js
	// defer 某操作	finally { 某操作 }
	// 写在函数中间，登记即可	要把后面所有代码包进 try {}
	// Go 的好处是不用多包一层 try，一行 defer 就搞定，代码不用缩进。
	err := glfw.Init()
	if err != nil {
		log.Fatal(err)
	}
	defer glfw.Terminate()

	// ── 1. 创建窗口 ───────────────────────────────
	// 向系统申请 OpenGL 4.1 core。macOS 最高只支持 4.1，并且必须开 ForwardCompatible
	glfw.WindowHint(glfw.ContextVersionMajor, 4)
	glfw.WindowHint(glfw.ContextVersionMinor, 1)
	glfw.WindowHint(glfw.OpenGLProfile, glfw.OpenGLCoreProfile)
	glfw.WindowHint(glfw.OpenGLForwardCompatible, glfw.True)

	window, err := glfw.CreateWindow(640, 480, "Triangle", nil, nil)
	if err != nil {
		log.Fatal(err)
	}

	// 把这个窗口的 OpenGL 上下文设为"当前"，之后的 gl.xxx 都作用在它上面
	// 相当于 JS：const gl = canvas.getContext('webgl2')
	window.MakeContextCurrent()

	// 加载 OpenGL 函数，必须在 MakeContextCurrent 之后
	if err := gl.Init(); err != nil {
		log.Fatal(err)
	}
	fmt.Println("OpenGL 版本:", gl.GoStr(gl.GetString(gl.VERSION)))

	// ── 2. 编译着色器，链接成程序 ──────────────────
	program := newProgram(vertexSrc, fragmentSrc)

	// ── 3. 准备 3 个三角形 ──────────────────────────
	// 坐标范围 -1 ~ 1，原点在窗口中心，x 向右，y 向上
	//
	// VBO 和 VAO 的区别：
	//   VBO（Vertex Buffer Object）= GPU 里的一块内存，存"顶点数据本身"（一堆 float）
	//   VAO（Vertex Array Object） = 一份"读数据的配置"，记录：用哪个 VBO、
	//                                每个顶点几个 float、从哪开始读、对应 shader 的哪个 location
	//   类比：VBO 是一张存了数据的 Excel 表格，VAO 是"这张表怎么读"的说明书。
	//   画的时候只要 BindVertexArray(某个 vao)，说明书和它指向的数据就都生效了。
	//   这里每个三角形一个 VBO + 一个 VAO，画哪个就绑定哪个。
	triangles := []triangle{
		newTriangle([]float32{-0.6, 0.4, -0.85, -0.4, -0.35, -0.4}, [3]float32{1.0, 0.3, 0.3}), // 左：红
		newTriangle([]float32{0.0, 0.4, -0.25, -0.4, 0.25, -0.4}, [3]float32{0.3, 1.0, 0.3}),   // 中：绿
		newTriangle([]float32{0.6, 0.4, 0.35, -0.4, 0.85, -0.4}, [3]float32{0.3, 0.5, 1.0}),    // 右：蓝
	}
	uColor := gl.GetUniformLocation(program, gl.Str("uColor\x00")) // 找到 shader 里 uColor 的位置

	// ── 4. 渲染循环 ───────────────────────────────
	// 等价于 JS
	// while (!window.shouldClose()) {
	//   glfw.pollEvents()
	// }
	for !window.ShouldClose() {
		gl.ClearColor(0.1, 0.1, 0.15, 1) // 背景色
		gl.Clear(gl.COLOR_BUFFER_BIT)

		gl.UseProgram(program)
		for _, t := range triangles {
			gl.Uniform3f(uColor, t.color[0], t.color[1], t.color[2]) // 设置颜色
			gl.BindVertexArray(t.vao)                                // 切换到这个三角形的"说明书"
			gl.DrawArrays(gl.TRIANGLES, 0, 3)                        // 画 3 个顶点组成的三角形
		}

		window.SwapBuffers() // 把画好的内容显示到窗口（双缓冲）
		glfw.PollEvents()
	}
}

type triangle struct {
	vao   uint32
	color [3]float32
}

// 创建一个三角形：上传顶点数据到 VBO，并用 VAO 记录读取方式
func newTriangle(vertices []float32, color [3]float32) triangle {
	var vao, vbo uint32
	// JS	Go
	// 对象、数组	自动按引用，你看不到指针	要自己决定传值还是传指针
	// 基础类型（数字、字符串）	按值传	按值传
	// 想让函数改外面的变量	传对象，改属性	传指针
	// 所以不是"Go 要你多写 &"，而是 Go 把 JS 里被隐藏的"引用"显式交给你来选。
	gl.GenVertexArrays(1, &vao) // VAO：记录"数据怎么读"
	gl.GenBuffers(1, &vbo)      // VBO：GPU 里存顶点数据的内存

	gl.BindVertexArray(vao) // 先绑定 VAO：之后的配置都会记录进这个 VAO
	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
	// 第 2 个参数是字节数：每个 float32 占 4 字节
	gl.BufferData(gl.ARRAY_BUFFER, len(vertices)*4, gl.Ptr(vertices), gl.STATIC_DRAW)

	// 告诉 GPU：location=0 的 aPos，每个顶点读 2 个 float，每个顶点占 2*4 字节
	// 这条配置连同当前绑定的 VBO，一起被记录进 VAO
	gl.VertexAttribPointer(0, 2, gl.FLOAT, false, 2*4, gl.PtrOffset(0))
	gl.EnableVertexAttribArray(0)

	return triangle{vao: vao, color: color}
}

// 编译两个着色器并链接成一个程序
func newProgram(vertexSource, fragmentSource string) uint32 {
	vs := compileShader(vertexSource, gl.VERTEX_SHADER)
	fs := compileShader(fragmentSource, gl.FRAGMENT_SHADER)

	program := gl.CreateProgram()
	gl.AttachShader(program, vs)
	gl.AttachShader(program, fs)
	gl.LinkProgram(program)

	var status int32
	gl.GetProgramiv(program, gl.LINK_STATUS, &status)
	if status == gl.FALSE {
		log.Fatal("着色器程序链接失败")
	}

	gl.DeleteShader(vs) // 链接完就不需要了
	gl.DeleteShader(fs)
	return program
}

func compileShader(source string, kind uint32) uint32 {
	shader := gl.CreateShader(kind)
	csources, free := gl.Strs(source)
	gl.ShaderSource(shader, 1, csources, nil)
	free()
	gl.CompileShader(shader)

	var status int32
	gl.GetShaderiv(shader, gl.COMPILE_STATUS, &status)
	if status == gl.FALSE {
		var length int32
		gl.GetShaderiv(shader, gl.INFO_LOG_LENGTH, &length)
		msg := make([]byte, length+1)
		gl.GetShaderInfoLog(shader, length, nil, &msg[0])
		log.Fatalf("着色器编译失败: %s", msg)
	}
	return shader
}
