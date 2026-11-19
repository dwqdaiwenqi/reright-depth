package main

import (
	_ "embed" // go:embed 需要
	"image"
	"image/draw"
	"log"
	"strings"

	"github.com/go-gl/gl/v4.1-core/gl"
)

// ── 着色器 ──────────────────────────────────────────────────────
// 源码放在单独的 shader.vert / shader.frag 里（编辑器能给 GLSL 上色）。
// //go:embed 会在【编译时】把文件内容塞进下面的字符串变量，
// 所以运行时不依赖这两个文件，go build 出来的仍然是单个可执行文件。
//
// 和 main.js 的区别：#version 300 es → 410 core，去掉 precision
//
//go:embed shader.vert
var vertexSrc string

//go:embed shader.frag
var fragmentSrc string

// ── renderer：把"程序、几何、纹理、uniform 位置"收在一起 ──────────
type renderer struct {
	program  uint32
	vao      uint32
	imageTex uint32
	depthTex uint32
	imgW     int
	imgH     int
	u        struct { // 匿名结构体：各个 uniform 的位置
		image, depth, lightPos, resolution, intensity, specular int32
		depthScale, ambient, shadowMode, showDepth              int32
	}
}

// 深度图还没回来时先用这张"平的"顶替：全 0 → 法线朝正前方，画面是均匀打光
var flatDepth = &depthMap{width: 1, height: 1, data: []float32{0}}

func newRenderer(img image.Image) *renderer {
	r := &renderer{
		program: newProgram(vertexSrc, fragmentSrc),
		vao:     newFullscreenQuad(),
		imgW:    img.Bounds().Dx(),
		imgH:    img.Bounds().Dy(),
	}
	r.imageTex = newTexture(img)
	r.depthTex = newFloatTexture(flatDepth)

	// JS：gl.getUniformLocation(program, name)
	loc := func(name string) int32 { return gl.GetUniformLocation(r.program, gl.Str(name+"\x00")) }
	r.u.image, r.u.depth = loc("uImage"), loc("uDepth")
	r.u.lightPos, r.u.resolution = loc("uLightPos"), loc("uResolution")
	r.u.intensity, r.u.specular = loc("uIntensity"), loc("uSpecular")
	r.u.depthScale, r.u.ambient = loc("uDepthScale"), loc("uAmbient")
	r.u.shadowMode, r.u.showDepth = loc("uShadowMode"), loc("uShowDepth")

	// 透明处透出背景：配合 shader 输出的预乘 alpha
	gl.Enable(gl.BLEND)
	gl.BlendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA)
	return r
}

// 深度图回来之后，换掉占位的那张
func (r *renderer) setDepth(d *depthMap) {
	gl.DeleteTextures(1, &r.depthTex) // 释放旧纹理，GPU 内存不会被 Go 的垃圾回收管，要手动释放
	r.depthTex = newFloatTexture(d)
}

// JS：function frame() { ... }
func (r *renderer) draw(s *state, fbW, fbH int) {
	// 视网膜屏上，帧缓冲的像素数 ≠ 窗口坐标数，viewport 要用帧缓冲尺寸
	gl.Viewport(0, 0, int32(fbW), int32(fbH))
	gl.ClearColor(0.125, 0.125, 0.18, 1) // 背景色
	gl.Clear(gl.COLOR_BUFFER_BIT)

	gl.UseProgram(r.program)
	gl.Uniform3f(r.u.lightPos, s.lightX, s.lightY, s.height)
	gl.Uniform2f(r.u.resolution, float32(r.imgW), float32(r.imgH)) // 用图片尺寸，和 JS 版（画布=图片尺寸）一致
	gl.Uniform1f(r.u.intensity, s.intensity)
	gl.Uniform1f(r.u.specular, s.specular)
	gl.Uniform1f(r.u.depthScale, s.depthScale)
	gl.Uniform1f(r.u.ambient, s.ambient)
	gl.Uniform1i(r.u.shadowMode, s.shadowMode)
	gl.Uniform1i(r.u.showDepth, s.showDepth)

	// 纹理单元 0 放原图，1 放深度图；sampler 里填的是"单元号"
	gl.ActiveTexture(gl.TEXTURE0)
	gl.BindTexture(gl.TEXTURE_2D, r.imageTex)
	gl.Uniform1i(r.u.image, 0)
	gl.ActiveTexture(gl.TEXTURE1)
	gl.BindTexture(gl.TEXTURE_2D, r.depthTex)
	gl.Uniform1i(r.u.depth, 1)

	gl.BindVertexArray(r.vao)
	gl.DrawArrays(gl.TRIANGLES, 0, 6) // 全屏四边形 = 2 个三角形 = 6 个顶点
}

// ── 全屏四边形：每个顶点 x, y, u, v，v 向上（左下角 uv=(0,0)）────────
func newFullscreenQuad() uint32 {
	vertices := []float32{
		-1, -1, 0, 0,
		1, -1, 1, 0,
		1, 1, 1, 1,
		-1, -1, 0, 0,
		1, 1, 1, 1,
		-1, 1, 0, 1,
	}
	var vao, vbo uint32
	gl.GenVertexArrays(1, &vao)
	gl.GenBuffers(1, &vbo)
	gl.BindVertexArray(vao)
	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
	gl.BufferData(gl.ARRAY_BUFFER, len(vertices)*4, gl.Ptr(vertices), gl.STATIC_DRAW)

	stride := int32(4 * 4)                                                 // 每个顶点 4 个 float
	gl.VertexAttribPointer(0, 2, gl.FLOAT, false, stride, gl.PtrOffset(0)) // aPos：前 2 个
	gl.EnableVertexAttribArray(0)
	gl.VertexAttribPointer(1, 2, gl.FLOAT, false, stride, gl.PtrOffset(2*4)) // aUv：后 2 个，偏移 8 字节
	gl.EnableVertexAttribArray(1)
	return vao
}

// ── 纹理参数：线性过滤 + 边缘夹紧（和 JS 版一样）────────────────────
func setTexParams() {
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
}

// ── 上传原图（RGBA8，每通道 0~255）─────────────────────────────────
// 对应 JS 的 newTexture。JS 靠 createImageBitmap 的 imageOrientation:'flipY' 翻转，这里自己翻行
func newTexture(img image.Image) uint32 {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()

	// 用 NRGBA（不预乘 alpha）：shader 里是自己 rgb*a 的，如果这里已经预乘过就会乘两次
	nrgba := image.NewNRGBA(image.Rect(0, 0, w, h))
	draw.Draw(nrgba, nrgba.Bounds(), img, b.Min, draw.Src)

	flipped := make([]uint8, len(nrgba.Pix))
	for y := 0; y < h; y++ {
		copy(flipped[(h-1-y)*nrgba.Stride:(h-y)*nrgba.Stride], nrgba.Pix[y*nrgba.Stride:(y+1)*nrgba.Stride])
	}

	var tex uint32
	gl.GenTextures(1, &tex)
	gl.BindTexture(gl.TEXTURE_2D, tex)
	gl.PixelStorei(gl.UNPACK_ALIGNMENT, 1)
	gl.TexImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, int32(w), int32(h), 0, gl.RGBA, gl.UNSIGNED_BYTE, gl.Ptr(flipped))
	setTexParams()
	return tex
}

// ── 上传单通道浮点纹理（深度图）────────────────────────────────────
// 对应 JS 的 newFloatTexture。桌面 OpenGL 4.1 的 R32F 天然支持线性过滤，
// 不用像浏览器那样检测 OES_texture_float_linear，也不用退到 R16F
func newFloatTexture(d *depthMap) uint32 {
	var tex uint32
	gl.GenTextures(1, &tex)
	gl.BindTexture(gl.TEXTURE_2D, tex)
	gl.TexImage2D(gl.TEXTURE_2D, 0, gl.R32F, int32(d.width), int32(d.height), 0, gl.RED, gl.FLOAT, gl.Ptr(d.data))
	setTexParams()
	return tex
}

// ── 编译两个着色器并链接成一个程序 ────────────────────────────────
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
		var length int32
		gl.GetProgramiv(program, gl.INFO_LOG_LENGTH, &length)
		msg := make([]byte, length+1)
		gl.GetProgramInfoLog(program, length, nil, &msg[0])
		log.Fatalf("着色器程序链接失败: %s", msg)
	}

	gl.DeleteShader(vs)
	gl.DeleteShader(fs)
	return program
}

func compileShader(source string, kind uint32) uint32 {
	// OpenGL 是 C 接口，字符串要以 \x00 结尾；这里统一补上，shader 源码里不用管
	if !strings.HasSuffix(source, "\x00") {
		source += "\x00"
	}
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
