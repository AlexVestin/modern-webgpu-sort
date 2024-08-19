#pragma once

#include <inttypes.h>

const uint32_t IMAGE_WIDTH = 1920;
const uint32_t IMAGE_HEIGHT = 1080;
const uint32_t DIRECTION_UP  = 1u;
const uint32_t DIRECTION_DOWN = 2u;
const uint32_t TILE_SIZE = 8u;

const float TILE_SIZE_DIV = 1.0f / static_cast<float>(TILE_SIZE);
const uint32_t linesPerQuad = 2000u;

struct Span {
    uint32_t key; // y 16 bits, x 16 bits 
    uint32_t lineStartIndex; //  24 bits
    int32_t spanMaxX; // in the range of [-1, width] -> 12 bits for 4k rendering

    inline void PackTypeLineEndIndex(uint32_t type, uint32_t lineEndIndex) {
        this->lineEndIndex = (type << 24u) | (lineEndIndex & 0xffffffu);
    }

    inline uint32_t GetLineEndIndex() const {
        return lineEndIndex & 0xffffffu;
    }

    inline uint32_t GetType() const {
        return lineEndIndex >> 24u;
    }

    inline void SetType(uint32_t type) {
        lineEndIndex = (type << 24u) | GetLineEndIndex();
    }

    inline uint32_t NumLines() const {
        return (GetLineEndIndex() - lineStartIndex) + 1u;
    }
private:
    uint32_t lineEndIndex; // 8 bits type, 24 bits lineIndex
};

struct DrawSpan {
    uint32_t position;
    uint32_t lineStartIndex;
    uint32_t lineEndIndex;
    uint32_t pathId;
    uint32_t atlasPosition;
    uint32_t padding;
};