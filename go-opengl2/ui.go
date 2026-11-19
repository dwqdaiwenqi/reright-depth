package main

import (
	"fmt"

	"github.com/AllenDang/cimgui-go/imgui"
)

// 右侧控制面板的宽度（窗口坐标）
const panelW = 300

// state：所有会变的东西（光照参数 + 光源位置），对应 main.js 里的 params 和 light
type state struct {
	intensity, height, ambient, depthScale, specular float32
	shadowMode                                       int32
	showDepth                                        bool
	lightX, lightY                                   float32
	status                                           string // 显示在面板顶部
}

func newState() *state {
	return &state{
		intensity: 1.5, height: 1.5, ambient: 0.2, depthScale: 0.08, specular: 0.4,
		shadowMode: 3,
		lightX:     0.5, lightY: 0.5,
		status: "Requesting depth map...",
	}
}

// ImGui 自带字体不含中文，界面文字用英文
var shadowItems = []string{"None", "Distance falloff", "Hard shadow", "RayMarch soft", "RayMarch + jitter"}

// 鼠标在图像区域内移动 = 移动光源。
// imageW/imageH 是图像区域的窗口坐标尺寸（不含右侧面板）
// JS：canvas.addEventListener('mousemove', ...)
func (s *state) updateLight(imageW, imageH float32) {
	io := imgui.CurrentIO()
	if io.WantCaptureMouse() { // 鼠标正在面板上（比如拖滑块），不要同时移动光源
		return
	}
	vp := imgui.MainViewport()
	m := io.MousePos()
	x := (m.X - vp.Pos().X) / imageW
	y := (m.Y - vp.Pos().Y) / imageH
	if x < 0 || x > 1 || y < 0 || y > 1 {
		return
	}
	s.lightX = x
	s.lightY = 1 - y // UV 原点在左下，要翻转 y
}

// 画控制面板。ImGui 是"立即模式"：每一帧都重新描述一遍界面，没有控件对象要创建和销毁。
// 滑块直接接收字段的地址（&s.intensity），拖动时 ImGui 会直接改这个字段，不用写回调
// JS 里对应的是 lil-gui：gui.add(params, 'intensity', 0, 4)
//
// 面板是一个可以随意拖动的浮动窗口：
//   - 拖标题栏移动位置（面板挡住画面时，可以拖开）
//   - 点标题栏左边的小三角折叠 / 展开
//   - 拖右下角调整大小
//
// CondFirstUseEver = "只有第一次出现时才用这个位置和大小"，之后听用户的。
// 如果每帧都强制设置，窗口就会被"钉死"，拖不动（我们最开始的写法就是这样）
func (s *state) drawPanel(winW, winH float32) {
	vp := imgui.MainViewport()
	// 初始位置：窗口右侧。ImGui 开了多视口时坐标是屏幕坐标，所以要加上主窗口的位置 vp.Pos()
	imgui.SetNextWindowPosV(imgui.NewVec2(vp.Pos().X+winW, vp.Pos().Y), imgui.CondFirstUseEver, imgui.NewVec2(0, 0))
	imgui.SetNextWindowSizeV(imgui.NewVec2(panelW, winH), imgui.CondFirstUseEver)
	imgui.Begin("Controls")

	imgui.TextWrapped(s.status)
	imgui.Separator()

	// ── 光的位置：x、y 由鼠标控制，z（高度）由这个滑块控制 ──
	imgui.SeparatorText("Light position")
	imgui.SliderFloat("Z (height)", &s.height, 0.2, 3)
	imgui.TextUnformatted(fmt.Sprintf("x=%.2f  y=%.2f  z=%.2f", s.lightX, s.lightY, s.height)) // TextUnformatted 不解析 %，安全

	imgui.SeparatorText("Lighting")
	imgui.SliderFloat("Intensity", &s.intensity, 0, 4)
	imgui.SliderFloat("Ambient", &s.ambient, 0, 1)
	imgui.SliderFloat("Depth scale", &s.depthScale, 0.01, 0.5)
	imgui.SliderFloat("Specular", &s.specular, 0, 2)
	imgui.ComboStrarr("Shadow", &s.shadowMode, shadowItems, int32(len(shadowItems)))
	imgui.Checkbox("Show depth map", &s.showDepth)

	imgui.Separator()
	imgui.TextWrapped("Move the mouse over the image to move the light.")

	imgui.End()
}

// 在图像上画一个小圆点，标出光的 x、y 位置；旁边写上高度 z。
// 用 ImGui 的"背景绘制列表"：画在我们的 OpenGL 图像之上、控制面板之下，不会盖住面板
// 注意：多视口开启时 ImGui 用屏幕坐标，所以要加上主窗口位置 vp.Pos()
func (s *state) drawLightMarker(imageW, imageH float32) {
	vp := imgui.MainViewport()
	center := imgui.NewVec2(
		vp.Pos().X+s.lightX*imageW,
		vp.Pos().Y+(1-s.lightY)*imageH, // 光的 y 是"向上为正"，屏幕 y 是"向下为正"，要翻回来
	)

	// 光越高，圆点越大（给"高度"一个直观的感觉）
	radius := 4 + 3*s.height

	dl := imgui.BackgroundDrawList()
	black := imgui.ColorU32Vec4(imgui.NewVec4(0, 0, 0, 0.8))
	yellow := imgui.ColorU32Vec4(imgui.NewVec4(1, 0.9, 0.2, 1))
	white := imgui.ColorU32Vec4(imgui.NewVec4(1, 1, 1, 1))

	dl.AddCircleFilled(center, radius+2, black) // 黑色底，深浅背景上都看得清
	dl.AddCircleFilled(center, radius, yellow)  // 黄色圆点
	dl.AddCircle(center, radius+4, white)       // 外圈
	dl.AddTextVec2(imgui.NewVec2(center.X+radius+8, center.Y-8), white,
		fmt.Sprintf("z=%.2f", s.height))
}
