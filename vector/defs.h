#pragma once

#include <inttypes.h>
#include <vector>

const uint32_t IMAGE_WIDTH = 1920;
const uint32_t IMAGE_HEIGHT = 1080;
const uint32_t DIRECTION_UP  = 1u;
const uint32_t DIRECTION_DOWN = 2u;
const uint32_t TILE_SIZE = 8u;

const float TILE_SIZE_DIV = 1.0f / static_cast<float>(TILE_SIZE);
const uint32_t linesPerQuad = 2000u;

struct BitArray {
    void reserve(uint32_t numBits) {
        uint32_t lastCapacicty = values.capacity();

        uint32_t valuesNeeded = (numBits + 31u) >> 5u;
        values.reserve(valuesNeeded + (valuesNeeded & 3u));
        // Zero initialize

        std::fill(values.begin() + lastCapacicty, values.begin() + values.capacity(), 0);
    }

    void clear() {
        uint32_t valuesNeeded = (maxIndexBitSet + 31u) >> 5u;
        std::fill(values.begin(), values.begin() + valuesNeeded, 0);
        maxIndexBitSet = 0u;
    }

    inline void SetBit(uint32_t index) {
        values[index >> 5u] |= (1u << (index & 31u));
        maxIndexBitSet = std::max(maxIndexBitSet, index);
    }

    inline bool IsBitSet(uint32_t index) const {
        return values[index >> 5u] & (1u << (index & 31u));
    }

    inline uint32_t capacity() const {
        return values.capacity() * 32u;
    }
 
 private:
    uint32_t maxIndexBitSet = 0u;
    std::vector<uint32_t> values;
};

struct Span {
    uint32_t key; // y 16 bits, x 16 bits 
    uint32_t lineStartIndex; //  24 bits
    int32_t spanMaxX; // in the range of [-1, width] -> 12 bits for 4k rendering


    Span() { }

    Span (uint32_t key, int32_t spanMaxX, uint32_t lineStartIndex, uint32_t packedTypeLineEndIndex): key{key}, spanMaxX{spanMaxX}, lineStartIndex{lineStartIndex}, packedTypeLineEndIndex{packedTypeLineEndIndex} {

    }

    inline void PackTypeLineEndIndex(uint32_t type, uint32_t lineEndIndex) {
        packedTypeLineEndIndex = (type << 24u) | (lineEndIndex & 0xffffffu);
    }

    inline uint32_t GetLineEndIndex() const {
        return packedTypeLineEndIndex & 0xffffffu;
    }

    inline uint32_t GetType() const {
        return packedTypeLineEndIndex >> 24u;
    }

    inline void SetType(uint32_t type) {
        packedTypeLineEndIndex = (type << 24u) | GetLineEndIndex();
    }

    inline uint32_t NumLines() const {
        return (GetLineEndIndex() - lineStartIndex) + 1u;
    }
private:
    uint32_t packedTypeLineEndIndex; // 8 bits type, 24 bits lineIndex
};

struct DrawSpan {
    uint32_t position;
    uint32_t lineStartIndex;
    uint32_t lineEndIndex;
    uint32_t pathId;
    // uint32_t atlasPosition;
    // uint32_t padding;
};