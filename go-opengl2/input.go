package main

import (
	"fmt"

	"github.com/go-gl/glfw/v3.3/glfw"
)

// state：所有会变的东西（光照参数 + 光源位置），对应 main.js 里的 params 和 light
type state struct {
	intensity, height, ambient, depthScale, specular float32
	shadowMode                                       int32
	showDepth                                        int32
	lightX, lightY                                   float32
	status                                           string // 显示在标题栏，比如"请求深度图中…"
}

func newState() *state {
	return &state{
		intensity: 1.5, height: 1.5, ambient: 0.2, depthScale: 0.08, specular: 0.4,
		shadowMode: 3,
		lightX:     0.5, lightY: 0.5,
	}
}

// 窗口标题：把参数显示出来，代替 JS 版侧栏滑块旁边的数字
func (s *state) title() string {
	return fmt.Sprintf("Relight [%s]  光强=%.1f 光高度=%.1f 环境光=%.2f 深度强=%.2f 高光=%.2f 阴影=%d",
		s.status, s.intensity, s.height, s.ambient, s.depthScale, s.specular, s.shadowMode)
}

// 一个按键对应"改哪个参数、每次加减多少、范围多大"
type adjust struct {
	value         *float32 // 指向 state 里某个字段的地址：改 *value 就是改那个字段
	delta, lo, hi float32
}

// bind 给窗口挂上鼠标和键盘回调。
// (s *state) 叫"方法接收者"：用指针，方法里改的才是原来那个 state（对应之前讲的传地址）
func (s *state) bind(window *glfw.Window) {
	// 表驱动：用一张表代替一大串 switch case
	keys := map[glfw.Key]adjust{
		glfw.KeyQ: {&s.intensity, +0.1, 0, 4},
		glfw.KeyA: {&s.intensity, -0.1, 0, 4},
		glfw.KeyW: {&s.height, +0.1, 0.2, 3},
		glfw.KeyS: {&s.height, -0.1, 0.2, 3},
		glfw.KeyE: {&s.ambient, +0.05, 0, 1},
		glfw.KeyD: {&s.ambient, -0.05, 0, 1},
		glfw.KeyR: {&s.depthScale, +0.01, 0.01, 0.5},
		glfw.KeyF: {&s.depthScale, -0.01, 0.01, 0.5},
		glfw.KeyT: {&s.specular, +0.05, 0, 2},
		glfw.KeyG: {&s.specular, -0.05, 0, 2},
	}

	// JS：canvas.addEventListener('mousemove', ...)
	window.SetCursorPosCallback(func(_ *glfw.Window, x, y float64) {
		w, h := window.GetSize() // 窗口坐标（不是像素），和鼠标坐标同一套单位
		s.lightX = float32(x / float64(w))
		s.lightY = 1 - float32(y/float64(h)) // UV 原点在左下，要翻转 y
	})

	// JS：滑块 + select 的 change 事件
	window.SetKeyCallback(func(_ *glfw.Window, key glfw.Key, _ int, action glfw.Action, _ glfw.ModifierKey) {
		if action == glfw.Release {
			return
		}

		// 逗号 ok 写法：从 map 里取值，同时告诉你"有没有这个键"（JS 里要自己判断 undefined）
		if a, ok := keys[key]; ok {
			*a.value = clamp(*a.value+a.delta, a.lo, a.hi)
		}

		switch key {
		case glfw.KeyEscape:
			window.SetShouldClose(true)
		case glfw.Key0, glfw.Key1, glfw.Key2, glfw.Key3, glfw.Key4:
			s.shadowMode = int32(key - glfw.Key0)
		case glfw.KeySpace:
			s.showDepth = 1 - s.showDepth // 切换：打光结果 / 深度图
		}
		window.SetTitle(s.title())
	})
}

func clamp(v, lo, hi float32) float32 { return min(max(v, lo), hi) }

func printHelp() {
	fmt.Println(`操作：
  鼠标移动  光源 XY
  Q / A     光强 +/-          W / S  光高度 +/-
  E / D     环境光 +/-        R / F  深度强度 +/-
  T / G     高光 +/-
  0-4       阴影：0 无 1 距离衰减 2 硬阴影 3 RayMarch软 4 RayMarch+抖动
  空格      切换：打光结果 / 深度图
  Esc       退出`)
}
