// Copyright 2021, Alexander Vestin (alex.vestin@gmail.com)

#pragma once

#include <inttypes.h>

#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

#include "kurbo_stroke.h"
#include "ColorMix.h"

enum class ClipMode : uint32_t {
    None = 0x00,
    Default = 0x01,
    Inverted = 0x02,
    Pop = 0x03,
};

enum class PaintStyle { kNone, kFill, kStroke, kFillAndStroke };

struct Col {
    float r, g, b, a;
};

enum class LineJoin { Miter, Bevel, Round };

enum class LineCap { Butt, Round, Square };

enum class FillRule { EvenOdd, NonZero };

enum class ResourceType  {
    None,
    Image,
    Video,
};

struct VColor {
    union {
        float rgba[4];
        Col color;
    };

    uint8_t toU8(float v) const { return std::min(255u, static_cast<uint32_t>(v * 255)); }

    const uint32_t GetU8ABGR() const {
        return (toU8(color.a) << 24) | (toU8(color.b) << 16) | (toU8(color.g) << 8) | toU8(color.r);
    }

    static constexpr VColor MakeABGR(uint32_t color) {
        uint8_t r = (color >> 0) & 0xff;
        uint8_t g = (color >> 8) & 0xff;
        uint8_t b = (color >> 16) & 0xff;
        uint8_t a = (color >> 24) & 0xff;
        return {r / 255.f, g / 255.f, b / 255.f, a / 255.f};
    }

    void Print() {
        std::cout << "{ r: " << color.r << ", g: " << color.g << ", b: " << color.b << ", a: " << color.a << " }"
                  << std::endl;
    }

    static constexpr VColor Make(float r, float g, float b, float a) { return {a, b, g, r}; }
};

class VPaint {
   public:
    void SetStyle(PaintStyle style) { this->style = style; }
    void SetFillColor(VColor color) { fillColor = color; }
    void SetFillColor(uint32_t color) { fillColor = VColor::MakeABGR(color); }

    void SetStrokeColor(VColor color) { strokeColor = color; }
    void SetStrokeColor(uint32_t color) { strokeColor = VColor::MakeABGR(color); }
    void SetStrokeOpacity(float opacity) { this->strokeOpacity = std::clamp(opacity, 0.0f, 1.0f); }
    void SetFillOpacity(float opacity) { this->fillOpacity = std::clamp(opacity, 0.0f, 1.0f); }
    void SetFillRule(FillRule fillRule) { this->fillRule = fillRule; }

    void SetResourceType(const ResourceType& type, bool isPremultiplied = false) {
        resourceType = type;
        this->isPremultiplied = isPremultiplied;
    }

    bool GetIsPremultiplied() const {
        return this->isPremultiplied;
    }
        
    uint32_t GetMotionBlurMode() const {
        return motionBlurMode;
    } 

    void SetMotionBlurMode(uint32_t mode) {
        motionBlurMode = mode;
    }

    void SetClipMode(ClipMode mode) { clipMode = mode; }

    ClipMode GetClipMode() const { return clipMode; }

    float GetStrokeOpacity() const { return strokeOpacity; }
    float GetFillOpacity() const { return fillOpacity; }

    void SetStrokeWidth(float w) { strokeWidth = w; }
    void SetLineJoin(LineJoin lj) { lineJoin = lj; }
    void SetLineCap(LineCap lc) { lineCap = lc; }

    kurbo::StrokeStyle GetKurboStrokeStyling() const {
        return {
            .tolerance = 0.3f,
            .strokeWidth = strokeWidth,
            .lineJoin = static_cast<uint32_t>(lineJoin),
            .lineCap = static_cast<uint32_t>(lineCap),
            .dashes = strokeDashArray.data(),
            .dashSize = strokeDashArray.size(),
        };
    }

    LineJoin GetLineJoin() const { return lineJoin; }
    LineCap GetLineCap() const { return lineCap; }

    PaintStyle GetPaintStyle() const { return style; }
    VColor GetFillColor() const {
        VColor copy{fillColor};
        copy.color.a *= fillOpacity;
        return copy;
    }

    const VColor& GetFillColorUnpremul() const {
        return fillColor;
    }

    bool HasTexture() const {
        return resourceType != ResourceType::None;
    }

    FillRule GetFillRule() const { return fillRule; }

    VColor GetStrokeColor() const {
        VColor copy{strokeColor};
        copy.color.a *= strokeOpacity;
        return copy;
    }

    const VColor& GetStrokeColorUnpremul() const {
        return strokeColor;
    }

    float GetStrokeWidth() const { return strokeWidth; }
    std::pair<float, float> GetStrokeRange() const { return {strokeStart, strokeEnd}; }
    void SetStrokeRange(float start, float end) {
        strokeStart = start;
        strokeEnd = end;
    }

    const lyra::ColorMix& GetColorMix() const {
        return colorMix;
    }

    void SetColorMix(uint32_t encodedColorMix)  {
        colorMix = lyra::ColorMix::FromEncoded(encodedColorMix);
    }

    void SetColorMix(const lyra::ColorMix& colorMix) {
        this->colorMix = colorMix;
    }

    void SetMotionBlurStep(uint32_t step) {
        motionBlurStep = step;
    }

    uint32_t GetMotionBlurStep() const {
        return motionBlurStep;
    }

    void SetStrokeDashArray(const std::vector<double>& dashes) { strokeDashArray = dashes; }

    void SetStrokeDashOffset(float offset) { strokeDashOffet = offset; }
    float GetStrokeDashOffset() const { return strokeDashOffet; }

    const std::vector<double>& GetStrokeDashArray() const { return strokeDashArray; }

   private:
    PaintStyle style = PaintStyle::kFill;
    VColor fillColor = {1.0, 0.0, 0.0, 1.0};
    float fillOpacity = 1.0;
    FillRule fillRule = FillRule::EvenOdd;
    ResourceType resourceType = ResourceType::None;
    bool isPremultiplied = false;
    ClipMode clipMode = ClipMode::None;
    lyra::ColorMix colorMix = { lyra::BlendMode::Normal, lyra::ComposeMode::SrcOver};

    // Stroke
    VColor strokeColor = {0.0, 1.0, 0.0, 1.0};
    float strokeWidth = 0.0f;
    float strokeStart = 0.f;
    float strokeEnd = 1.f;
    uint32_t motionBlurStep = 0u;
    LineJoin lineJoin = LineJoin::Miter;
    LineCap lineCap = LineCap::Butt;
    float strokeOpacity = 1.0;
    std::vector<double> strokeDashArray;
    uint32_t strokeDashCount = 0;
    float strokeDashOffet = 0.f;
    uint32_t motionBlurMode = 0u;
};
std::ostream& operator<<(std::ostream& o, const VColor& v);