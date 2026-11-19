package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"mime/multipart"
	"net/http"
	"path/filepath"
	"strings"
	"time"
)

// 深度图数据
type depthMap struct {
	width, height int
	data          []float32 // 行顺序已翻转：第 0 行在底部，可直接当纹理上传
}

// 请求深度图：POST {server}/depth，字段 image + size + format=float
// 响应是二进制：uint32 宽 + uint32 高（小端）+ 宽*高 个 float32（小端，第 0 行在顶部）
// 对应 main.js 的 fetchDepth
func fetchDepth(server, name string, imageBytes []byte, size string) (*depthMap, error) {
	body := &bytes.Buffer{}
	mw := multipart.NewWriter(body)
	fw, err := mw.CreateFormFile("image", filepath.Base(name)) // JS：form.append('image', blob, '30-3.png')
	if err != nil {
		return nil, err
	}
	fw.Write(imageBytes)
	mw.WriteField("size", size)
	mw.WriteField("format", "float")
	mw.Close()

	req, err := http.NewRequest(http.MethodPost, strings.TrimRight(server, "/")+"/depth", body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", mw.FormDataContentType())

	client := &http.Client{Timeout: 5 * time.Minute} // 首次加载模型可能较慢
	resp, err := client.Do(req)                      // JS：await fetch(...)，Go 里是普通的阻塞调用
	if err != nil {
		return nil, fmt.Errorf("连不上 %s，深度服务启动了吗？（python3 tools/gen_normal_dav2.py --serve）: %w", server, err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body) // JS：await res.arrayBuffer()
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
	}
	if strings.Contains(resp.Header.Get("Content-Type"), "json") {
		return nil, fmt.Errorf("服务端返回了 JSON，请重启 gen_normal_dav2.py（需要新版才支持 format=float）")
	}
	if len(data) < 8 {
		return nil, fmt.Errorf("响应太短: %d 字节", len(data))
	}

	// JS：view.getUint32(0, true) —— 小端读 4 字节
	w := int(binary.LittleEndian.Uint32(data[0:4]))
	h := int(binary.LittleEndian.Uint32(data[4:8]))
	if len(data) != 8+w*h*4 {
		return nil, fmt.Errorf("数据长度不对：宽%d 高%d 应有 %d 字节，实际 %d", w, h, 8+w*h*4, len(data))
	}

	// JS：new Float32Array(buf, 8, width*height)，同时把行顺序倒过来
	// 服务端第 0 行在顶部，纹理 v=0 在底部
	flipped := make([]float32, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			bits := binary.LittleEndian.Uint32(data[8+(y*w+x)*4:])
			flipped[(h-1-y)*w+x] = math.Float32frombits(bits)
		}
	}
	return &depthMap{width: w, height: h, data: flipped}, nil
}
