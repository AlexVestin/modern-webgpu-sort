#pragma once

#include <inttypes.h>

namespace lyra {
enum class BlendMode {
    Normal = 0u,
    Multiply = 1u,
    Screen = 2u,
    Overlay = 3u,
    Darken = 4u,
    Lighten = 5u,
    ColorDodge = 6u,
    ColorBurn = 7u,
    HardLight = 8u,
    SoftLight = 9u,
    Difference = 10u,
    Exclusion = 11u,
    Hue = 12u,
    Saturation = 13u,
    Color = 14u,
    Luminosity = 15u,
};

enum class ComposeMode {
    Clear = 0u,
    Copy = 1u,
    Dest = 2u,
    SrcOver = 3u,
    DestOver = 4u,
    SrcIn = 5u,
    DestIn = 6u,
    SrcOut = 7u,
    DestOut = 8u,
    SrcAtop = 9u,
    DestAtop = 10u,
    Xor = 11u,
    Plus = 12u,
    PlusLighter = 13u
};

struct ColorMix {
    BlendMode mode = BlendMode::Normal;
    ComposeMode comp = ComposeMode::SrcOver;

    static const ColorMix From(const BlendMode&& blendMode) { return {.mode = blendMode}; }

    static const ColorMix From(const BlendMode&& blendMode, const ComposeMode&& composeMode) {
        return {.mode = blendMode, .comp = composeMode};
    }

    static const ColorMix From(const ComposeMode&& composeMode) { return {.comp = composeMode}; }

    uint32_t Encode() const { return (static_cast<uint32_t>(mode) << 8u) | static_cast<uint32_t>(comp); }

    static ColorMix FromEncoded(uint32_t v) {
        uint32_t m = (v >> 8u);
        uint32_t c = (v & 0xffu);

        return {.mode = static_cast<BlendMode>(m), .comp = static_cast<ComposeMode>(c)};
    }
};
}