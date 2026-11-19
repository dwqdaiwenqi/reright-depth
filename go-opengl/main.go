// 最小示例：用 Go + OpenGL 画一个彩色三角形
// 运行：在 go-opengl 目录下执行  go run .
package main

import (
	"fmt"
	"log"
	"runtime"

	"github.com/go-gl/gl/v4.1-core/gl"
	"github.com/go-gl/glfw/v3.3/glfw"
)

// 顶点着色器：每个顶点运行一次，决定顶点位置
const vertSrc = `#version 410 core
layout(location = 0) in vec2 aPos;    // 位置
layout(location = 1) in vec3 aColor;  // 颜色
out vec3 vColor;
void main() {
    vColor = aColor;
    gl_Position = vec4(aPos, 0.0, 1.0);
}` + "\x00"

// 片元着色器：每个像素运行一次，决定像素颜色
const fragSrc = `#version 410 core
in vec3 vColor;
out vec4 fragColor;
void main() {
    fragColor = vec4(vColor, 1.0);
}` + "\x00"

func init() {
	// OpenGL 只能在创建窗口的那个线程调用，把 main 钉在主线程上
	runtime.LockOSThread()
}

func main() {
	// 1. 初始化 GLFW（负责创建窗口）并请求 OpenGL 4.1 core
	if err := glfw.Init(); err != nil {
		log.Fatal(err)
	}
	defer glfw.Terminate()
	glfw.WindowHint(glfw.ContextVersionMajor, 4)
	glfw.WindowHint(glfw.ContextVersionMinor, 1)
	glfw.WindowHint(glfw.OpenGLProfile, glfw.OpenGLCoreProfile)
	glfw.WindowHint(glfw.OpenGLForwardCompatible, glfw.True) // macOS 必须

	win, err := glfw.CreateWindow(640, 480, "Triangle", nil, nil)
	if err != nil {
		log.Fatal(err)
	}
	win.MakeContextCurrent()

	// 2. 加载 OpenGL 函数指针
	if err := gl.Init(); err != nil {
		log.Fatal(err)
	}
	fmt.Println("OpenGL", gl.GoStr(gl.GetString(gl.VERSION)))

	// 3. 编译并链接着色器
	prog := link(compile(vertSrc, gl.VERTEX_SHADER), compile(fragSrc, gl.FRAGMENT_SHADER))

	// 4. 顶点数据：每个顶点 x, y, r, g, b
	vertices := []float32{
		0.0, 0.6, 1, 0, 0, // 上：红
		-0.6, -0.6, 0, 1, 0, // 左下：绿
		0.6, -0.6, 0, 0, 1, // 右下：蓝
	}
	var vao, vbo uint32
	gl.GenVertexArrays(1, &vao)
	gl.GenBuffers(1, &vbo)
	gl.BindVertexArray(vao)
	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
	gl.BufferData(gl.ARRAY_BUFFER, len(vertices)*4, gl.Ptr(vertices), gl.STATIC_DRAW)
	stride := int32(5 * 4) // 每个顶点 5 个 float32
	gl.VertexAttribPointer(0, 2, gl.FLOAT, false, stride, gl.PtrOffset(0))
	gl.EnableVertexAttribArray(0)
	gl.VertexAttribPointer(1, 3, gl.FLOAT, false, stride, gl.PtrOffset(2*4))
	gl.EnableVertexAttribArray(1)

	// 5. 渲染循环
	for !win.ShouldClose() {
		gl.ClearColor(0.1, 0.1, 0.15, 1)
		gl.Clear(gl.COLOR_BUFFER_BIT)

		gl.UseProgram(prog)
		gl.BindVertexArray(vao)
		gl.DrawArrays(gl.TRIANGLES, 0, 3)

		win.SwapBuffers()
		glfw.PollEvents()
	}
}

func compile(src string, kind uint32) uint32 {
	s := gl.CreateShader(kind)
	cs, free := gl.Strs(src)
	gl.ShaderSource(s, 1, cs, nil)
	free()
	gl.CompileShader(s)
	var ok int32
	gl.GetShaderiv(s, gl.COMPILE_STATUS, &ok)
	if ok == gl.FALSE {
		var n int32
		gl.GetShaderiv(s, gl.INFO_LOG_LENGTH, &n)
		buf := make([]byte, n+1)
		gl.GetShaderInfoLog(s, n, nil, &buf[0])
		log.Fatalf("shader 编译失败: %s", buf)
	}
	return s
}

func link(vs, fs uint32) uint32 {
	p := gl.CreateProgram()
	gl.AttachShader(p, vs)
	gl.AttachShader(p, fs)
	gl.LinkProgram(p)
	var ok int32
	gl.GetProgramiv(p, gl.LINK_STATUS, &ok)
	if ok == gl.FALSE {
		log.Fatal("program 链接失败")
	}
	return p
}
