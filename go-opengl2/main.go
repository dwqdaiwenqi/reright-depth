// Go + OpenGL 4.1 版重光照，和 main.js（WebGL2）一一对应。
//
//  1. 加载 30-3.png，立刻打开窗口
//  2. 【后台协程】POST 到 gen_normal_dav2.py 的 /depth（format=float），拿到 float32 深度
//  3. 深度图回来后换上去；鼠标移动 = 移动光源
//
// 文件分工：
//
//	main.go    入口：参数、窗口、启动协程、主循环
//	depth.go   请求并解析深度图（HTTP）
//	gfx.go     OpenGL：着色器、纹理、渲染器
//	input.go   键盘鼠标、光照参数
//	shader.*   GLSL 源码
//
// 运行（先启动深度服务：python3 tools/gen_normal_dav2.py --serve --port 7860）：
//
//	go run .
//	go run . -image 25-3.png -server http://localhost:7860 -size small
package main

import (
	"bytes"
	"flag"
	"fmt"
	"image"
	_ "image/jpeg" // 只为了"注册 JPEG 解码器"（init 副作用），代码里不直接调用
	_ "image/png"  // 同上，PNG
	"log"
	"os"
	"runtime"
	"time"

	"github.com/go-gl/gl/v4.1-core/gl"
	"github.com/go-gl/glfw/v3.3/glfw"
)

func init() {
	runtime.LockOSThread() // glfw / OpenGL 必须在主线程
}

// 协程把结果交给主线程的"信封"
type depthResult struct {
	depth *depthMap
	err   error
	took  time.Duration
}

func main() {
	imagePath := flag.String("image", "30-3.png", "输入图片路径")
	server := flag.String("server", "http://localhost:7860", "深度服务地址")
	size := flag.String("size", "small", "模型大小：small | base | large")
	flag.Parse()

	// ── 1. 加载原图 ────────────────────────────────
	imageBytes, err := os.ReadFile(*imagePath)
	if err != nil {
		log.Fatalf("读取图片失败: %v", err)
	}
	img, _, err := image.Decode(bytes.NewReader(imageBytes))
	if err != nil {
		log.Fatalf("解码图片失败: %v", err)
	}

	// ── 2. 窗口 + OpenGL 上下文 ────────────────────
	if err := glfw.Init(); err != nil {
		log.Fatal(err)
	}
	defer glfw.Terminate()

	glfw.WindowHint(glfw.ContextVersionMajor, 4)
	glfw.WindowHint(glfw.ContextVersionMinor, 1)
	glfw.WindowHint(glfw.OpenGLProfile, glfw.OpenGLCoreProfile)
	glfw.WindowHint(glfw.OpenGLForwardCompatible, glfw.True)

	// 窗口按图片宽高比缩放，最长边不超过 900（对应 CSS 里把画布缩小显示）
	winW, winH := img.Bounds().Dx(), img.Bounds().Dy()
	if m := max(winW, winH); m > 900 {
		winW, winH = winW*900/m, winH*900/m
	}
	window, err := glfw.CreateWindow(winW, winH, "Relight", nil, nil)
	if err != nil {
		log.Fatal(err)
	}
	window.MakeContextCurrent()
	glfw.SwapInterval(1) // 垂直同步，对应浏览器的 requestAnimationFrame 节奏

	if err := gl.Init(); err != nil {
		log.Fatal(err)
	}
	fmt.Println("OpenGL 版本:", gl.GoStr(gl.GetString(gl.VERSION)))

	r := newRenderer(img)
	s := newState()
	s.bind(window)
	printHelp()

	// ── 3. 后台协程请求深度图 ──────────────────────
	// Node.js：
	//   fetchDepth(...).then(depth => { ... })      // 不阻塞，结果回来再处理
	// Go：
	//   go func() { ... }()                         // 开一个协程（goroutine），里面可以放心写"阻塞"代码
	//   通过 channel（通道）把结果送回来            // 相当于 Promise 的 resolve
	//
	// 重要：OpenGL 只能在主线程调用，所以协程里【只做网络请求】，不碰 gl.xxx；
	//       拿到数据后交给主线程，由主线程去上传纹理。
	resultCh := make(chan depthResult, 1) // 容量 1：协程发完就能走，不会卡在"没人收"上
	go func() {
		fmt.Printf("POST %s/depth  size=%s format=float ...\n", *server, *size)
		t0 := time.Now()
		d, err := fetchDepth(*server, *imagePath, imageBytes, *size)
		resultCh <- depthResult{depth: d, err: err, took: time.Since(t0)} // 发送：相当于 resolve(...)
	}()

	s.status = "请求深度图中…"
	window.SetTitle(s.title())

	// ── 4. 主循环 ──────────────────────────────────
	// JS：function frame() { ...; requestAnimationFrame(frame) }
	for !window.ShouldClose() {
		// select + default：看一眼通道里有没有结果，没有就【不等】，继续画
		// 相当于 JS 里"每一帧检查 promise 是否已完成"
		select {
		case res := <-resultCh:
			if res.err != nil {
				fmt.Println("✗", res.err)
				s.status = "深度图失败，见终端"
			} else {
				r.setDepth(res.depth)
				fmt.Printf("深度图已获取（%.2fs，%d×%d）\n", res.took.Seconds(), res.depth.width, res.depth.height)
				s.status = fmt.Sprintf("深度图 %.1fs", res.took.Seconds())
			}
			window.SetTitle(s.title())
		default:
		}

		fbW, fbH := window.GetFramebufferSize()
		r.draw(s, fbW, fbH)

		window.SwapBuffers()
		glfw.PollEvents()
	}
}
