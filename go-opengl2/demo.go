//go:build ignore

// 传值 / 传地址 / 返回值 对比 demo
//
// 运行：go run demo.go
//
// 第一行 "//go:build ignore" 的作用：让 go run . 和 go build 忽略这个文件。
// 因为 main.go 里已经有 func main()，同一个目录下不能有第二个，
// 加了这行后，只有明确指定文件名（go run demo.go）时才会运行它。

package main

import "fmt"

// ───────────── 1. 传值：函数拿到的是"复印件" ─────────────
func addOneByValue(n int) {
	n = n + 1 // 只改了复印件
	fmt.Println("  函数内部 n =", n)
}

// ───────────── 2. 传地址：函数拿到的是"住址"，能改原件 ─────────────
func addOneByPointer(n *int) { // *int 读作"指向 int 的指针"
	*n = *n + 1 // *n 表示"去这个地址里取/改值"
	fmt.Println("  函数内部 *n =", *n)
}

// ───────────── 3. 返回值：算好直接还给你 ─────────────
func addOneReturn(n int) int {
	return n + 1
}

// ───────────── 模仿 OpenGL 的 Gen 函数 ─────────────

// C 风格（go-gl 现在的写法）：把结果写进你给的地址
func genIDPointer(out *uint32) {
	*out = 7 // 假装 GPU 分配了 ID 7
}

// Go 风格：直接返回
func genID() uint32 {
	return 7
}

// Go 风格：一次生成多个，返回切片（类似 JS 的数组）
func genIDs(count int) []uint32 {
	ids := make([]uint32, count)
	for i := range ids {
		ids[i] = uint32(100 + i)
	}
	return ids
}

func main() {
	fmt.Println("【传值】")
	a := 10
	addOneByValue(a)
	fmt.Println("  外面 a =", a, " ← 没变，因为函数改的是复印件")

	fmt.Println("\n【传地址】")
	b := 10
	addOneByPointer(&b) // &b：把 b 的住址给函数
	fmt.Println("  外面 b =", b, " ← 变了，因为函数直接去原件那里改")

	fmt.Println("\n【返回值】")
	c := 10
	c = addOneReturn(c)
	fmt.Println("  外面 c =", c, " ← 变了，靠赋值接住返回值")

	fmt.Println("\n【看一眼地址长什么样】")
	d := 10
	fmt.Println("  d 的值 =", d)
	fmt.Println("  d 的地址 =", &d, "← 内存里的位置，每次运行都不一样")

	fmt.Println("\n【模仿 OpenGL】")
	var vao1 uint32 // 先准备盒子，零值是 0
	fmt.Println("  调用前 vao1 =", vao1)
	genIDPointer(&vao1)
	fmt.Println("  C 风格   vao1 =", vao1, " ← 先 var，再传 &vao1")

	vao2 := genID()
	fmt.Println("  Go 风格  vao2 =", vao2, " ← 一行搞定")

	fmt.Println("  一次生成 3 个:", genIDs(3))
}
