#pragma once

#include <inttypes.h>

const uint32_t IMAGE_WIDTH = 1920;
const uint32_t IMAGE_HEIGHT = 1080;
const uint32_t DIRECTION_UP  = 1u;
const uint32_t DIRECTION_DOWN = 2u;
const uint32_t TILE_SIZE = 16u;

const float TILE_SIZE_DIV = 1.0f / static_cast<float>(TILE_SIZE);
const uint32_t linesPerSpan = 400u;

struct Span {
    uint32_t key; // y 16 bits, x 16 bits 
    uint32_t lineStartIndex; // 
    uint32_t lineEndIndex; // 
    int32_t spanMaxX; // in the range of [0, width] + left stuff
    uint32_t type;
};

struct DrawSpan {
    uint32_t position;
    uint32_t lineStartIndex;
    uint32_t lineEndIndex;
    uint32_t pathId;
    uint32_t atlasPosition;
};