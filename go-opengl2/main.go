// Go + OpenGL 4.1 版重光照，和 main.js（WebGL2）一一对应。
// 窗口和控制面板用 Dear ImGui（cimgui-go），滑块可以直接拖动。
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
//	ui.go      光照参数、ImGui 控制面板、鼠标控制光源
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

	"github.com/AllenDang/cimgui-go/backend"
	"github.com/AllenDang/cimgui-go/backend/glfwbackend"
	"github.com/AllenDang/cimgui-go/imgui"
	_ "github.com/AllenDang/cimgui-go/impl/glfw"    // ImGui 的 GLFW 平台层（副作用导入）
	_ "github.com/AllenDang/cimgui-go/impl/opengl3" // ImGui 的 OpenGL3 渲染层（副作用导入）
	"github.com/go-gl/gl/v4.1-core/gl"
)

func init() {
	runtime.LockOSThread() // GLFW / OpenGL 必须在主线程
}

// 协程把结果交给主线程的"信封"
type depthResult struct {
	depth *depthMap
	err   error
	took  time.Duration
}

func main() {
	imagePath := flag.String("image", "25-3.png", "输入图片路径")
	server := flag.String("server", "http://localhost:8888", "深度服务地址")
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

	// 图像区域按图片宽高比缩放，最长边不超过 900；窗口右边再留出控制面板
	imgW, imgH := img.Bounds().Dx(), img.Bounds().Dy()
	if m := max(imgW, imgH); m > 900 {
		imgW, imgH = imgW*900/m, imgH*900/m
	}

	s := newState()

	// ── 2. 后台协程请求深度图 ──────────────────────
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

	// ── 3. 窗口 ────────────────────────────────────
	// 窗口、OpenGL 上下文、事件循环都由 cimgui-go 的 GLFW 后端管理，
	// 我们只需要提供"每一帧要做什么"的函数
	b, err := backend.CreateBackend(glfwbackend.NewGLFWBackend())
	if err != nil {
		log.Fatal(err)
	}
	b.SetBgColor(imgui.NewVec4(0.125, 0.125, 0.18, 1)) // 背景色（后端每帧开头清屏用）
	b.CreateWindow("Relight", imgW+panelW, imgH)

	var r *renderer // 要等 OpenGL 上下文准备好才能创建，所以在第一帧里创建

	// ── 4. 每一帧 ──────────────────────────────────
	// JS：function frame() { ...; requestAnimationFrame(frame) }
	b.Run(func() {
		if r == nil {
			if err := gl.Init(); err != nil { // 上下文此时已经是当前的，可以加载 GL 函数
				log.Fatal(err)
			}
			fmt.Println("OpenGL 版本:", gl.GoStr(gl.GetString(gl.VERSION)))
			r = newRenderer(img)
		}

		// select + default：看一眼通道里有没有结果，没有就【不等】，继续画
		// 相当于 JS 里"每一帧检查 promise 是否已完成"
		select {
		case res := <-resultCh:
			if res.err != nil {
				fmt.Println("✗", res.err)
				s.status = "Depth map FAILED (flat lighting only). See the terminal for the error. If the server is running, restart it: python3 tools/gen_normal_dav2.py --serve --port 7860"
			} else {
				r.setDepth(res.depth)
				fmt.Printf("深度图已获取（%.2fs，%d×%d）\n", res.took.Seconds(), res.depth.width, res.depth.height)
				s.status = fmt.Sprintf("Depth map ready (%.1fs, %dx%d)", res.took.Seconds(), res.depth.width, res.depth.height)
			}
		default:
		}

		s.updateLight(float32(imgW), float32(imgH))

		// 视网膜屏：窗口坐标 → 像素，要乘缩放比例
		scale := imgui.CurrentIO().DisplayFramebufferScale().X
		r.draw(s, int32(float32(imgW)*scale), int32(float32(imgH)*scale))

		s.drawLightMarker(float32(imgW), float32(imgH)) // 光源位置的小圆点
		s.drawPanel(float32(imgW), float32(imgH))       // ImGui 控件在这之后由后端统一绘制
	})
}
