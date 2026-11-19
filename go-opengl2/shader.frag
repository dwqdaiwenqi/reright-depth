#version 410 core
in vec2 vUv;

uniform sampler2D uImage;      // 原图
uniform sampler2D uDepth;      // 深度图（白=近，黑=远）
uniform vec3      uLightPos;   // 光源位置（UV 空间 xy + 高度 z）
uniform float     uIntensity;  // 光强
uniform float     uSpecular;   // 高光强度
uniform float     uDepthScale; // 深度对法线的影响强度
uniform float     uAmbient;    // 环境光
uniform vec2      uResolution; // 图片像素尺寸（决定"一个像素"在 UV 里多大）
uniform int       uShadowMode; // 0=无 1=衰减 2=硬阴影 3=RayMarch软 4=RayMarch+抖动
uniform int       uShowDepth;  // 1 = 直接显示深度图（调试用）

out vec4 fragColor;

// 常量（取代代码里的魔法数字）
const float WRAP         = 0.3;    // Wrapped Lambert 的包裹量
const float SHININESS    = 32.0;   // 高光锐度
const float DEPTH_BIAS   = 0.02;   // 阴影判定的深度容差
const float STEP_COST    = 0.04;   // Ray March 每次命中遮挡减去的亮度
const float MIN_SHADOW   = 0.5;    // 软阴影最暗亮度
const float HARD_SHADOW  = 0.5;    // 硬阴影的暗度
const float HARD_OFFSET  = 4.0;    // 硬阴影采样偏移（像素）
const float MARCH_STEP   = 2.0;    // Ray March 步长（像素）
const int   MAX_STEPS    = 16;     // Ray March 最大步数
const float ATTEN_K      = 4.0;    // 距离衰减系数
const float ATTEN_MIN    = 0.3;    // 距离衰减最暗亮度

float getDepth(vec2 uv) {
    return texture(uDepth, uv).r;
}

void main() {
    if (uShowDepth == 1) {
        fragColor = vec4(vec3(getDepth(vUv)), 1.0);
        return;
    }

    // 一个像素在 UV 空间里的大小
    vec2 px = 1.0 / uResolution;

    // 1. 采样上下左右四个邻居的深度
    float dL = getDepth(vUv + vec2(-px.x, 0.0));
    float dR = getDepth(vUv + vec2(+px.x, 0.0));
    float dU = getDepth(vUv + vec2(0.0, +px.y));
    float dD = getDepth(vUv + vec2(0.0, -px.y));

    // 2. 中心差分求偏导（斜率）
    float dzdx = (dR - dL) / (2.0 * px.x);
    float dzdy = (dU - dD) / (2.0 * px.y);

    // 3. 曲面两条切线，叉积得到法线
    vec3 Tx = vec3(1.0, 0.0, dzdx * uDepthScale);
    vec3 Ty = vec3(0.0, 1.0, dzdy * uDepthScale);
    vec3 normal = normalize(cross(Tx, Ty));

    // 当前像素的 3D 位置、指向光源的方向
    float depth   = getDepth(vUv);
    vec3  pos     = vec3(vUv, depth);
    vec3  toLight = normalize(uLightPos - pos);

    // 4. Wrapped Lambert 漫反射
    float diffuse = clamp((dot(normal, toLight) + WRAP) / (1.0 + WRAP), 0.0, 1.0);

    // 5. Blinn-Phong 高光（正交相机，视线固定朝 +Z）
    vec3  viewDir = vec3(0.0, 0.0, 1.0);
    vec3  halfVec = normalize(toLight + viewDir);
    float spec    = pow(max(dot(normal, halfVec), 0.0), SHININESS) * uSpecular;

    // 6. 阴影
    vec2  dir2D  = uLightPos.xy - vUv;
    float dist   = length(dir2D);
    vec2  dir2Dn = dist > 1e-6 ? dir2D / dist : vec2(0.0);
    float shadow = 1.0;

    switch (uShadowMode) {
    case 0:
        // 无阴影
        shadow = 1.0;
        break;

    case 1: {
        // 距离平方衰减
        float attenuation = 1.0 / (1.0 + ATTEN_K * dist * dist);
        shadow = mix(ATTEN_MIN, 1.0, attenuation);
        break;
    }

    case 2: {
        // 硬阴影：只采样一个点
        float pDepth = getDepth(vUv + dir2Dn * px * HARD_OFFSET);
        shadow = (pDepth < depth - DEPTH_BIAS) ? HARD_SHADOW : 1.0;
        break;
    }

    case 3:
    default: {
        // Ray March 软阴影（mode 3 无抖动；mode 4 加抖动，起点在第一步的 0.5~1.0 之间）
        float jitter = (uShadowMode == 3)
            ? 1.0
            : 0.5 + 0.5 * fract(sin(dot(vUv, vec2(12.9898, 78.233))) * 43758.5453);
        vec2 stepV = dir2Dn * px * MARCH_STEP;
        vec2 cur   = vUv + stepV * jitter;
        for (int i = 0; i < MAX_STEPS; i++) {
            if (length(cur - vUv) >= dist) break;
            if (getDepth(cur) < depth - DEPTH_BIAS) shadow -= STEP_COST;
            cur += stepV;
        }
        shadow = clamp(shadow, MIN_SHADOW, 1.0);
        break;
    }
    }

    // 7. 合成。输出"预乘 alpha"（rgb 乘 a），配合 BlendFunc(ONE, ONE_MINUS_SRC_ALPHA)，透明处才会透出背景
    vec4  color = texture(uImage, vUv);
    float light = uAmbient + (diffuse + spec) * uIntensity * shadow;
    fragColor   = vec4(color.rgb * light * color.a, color.a);
}
