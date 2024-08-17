// Copyright 2021, Alexander Vestin (alex.vestin@gmail.com)

#pragma once

#include <cmath>
#include <iostream>

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

struct VPoint {
    float x;
    float y;

    static constexpr VPoint Make(float x, float y) { return {x, y}; }

    bool isZero() const { return (0 == x) & (0 == y); }
    bool isCorrect() const { return !std::isnan(x) && !std::isnan(y) && !std::isinf(x) && !std::isinf(y); }
    // VPoint transform(const float* t) const {
    //     return {
    //         x * t[0] + y * t[2] + t[4], 
    //         x * t[1] + y * t[3] + t[5]
    //     };
    // }

    VPoint translate(const VPoint& offset) const {
        return *this + offset;
    }

    VPoint scale(const VPoint& scale) const {
        return *this * scale;
    }

    VPoint transform(const float t[6]) const {
        return {
            x * t[0] + y * t[2] + t[4], 
            x * t[1] + y * t[3] + t[5]
        };
    }

    VPoint scale(const float t[6]) const {
        return {
            x * t[0] + y * t[2], 
            x * t[1] + y * t[3]
        };
    }

    VPoint yx() { return {y, x}; }

    // Creates copy
    VPoint inverseTransform(const float* t) const {
        // x * t0 + y * t2 = (x - t4)
        // x * t1 + y * t3 = (y - t5)
        // https://stackoverflow.com/questions/19619248/solve-system-of-two-equations-with-two-unknowns

        float a = t[0];
        float d = t[3];

        float b = t[2];
        float c = t[1];

        float e = x - t[4];
        float f = y - t[5];

        float determinant = a * d - b * c;
        if (determinant != 0) {
            float _x = (e * d - b * f) / determinant;
            float _y = (a * f - e * c) / determinant;
            return {_x, _y};
        }

        return {0.f, 0.f};
    }

    VPoint Lerp(const VPoint& other, float t) const { return (1.0f - t) * (*this) + t * other; }

    VPoint scale(float* transform) { return {x / transform[0], y / transform[3]}; }

    float Dot(const VPoint& other) const { return this->x * other.x + this->y * other.y; }

    float Cross(const VPoint& other) { return this->x * other.y - this->y * other.x; }

    float Length() const { return sqrt(this->Dot(*this)); }

    VPoint Sign() const { return VPoint::Make(sgn(this->x), sgn(this->y)); }

    float Angle() { return std::atan2(y, x); }

    float Distance(const VPoint& other) { return (other - *this).Length(); }

    float LengthSquared() { return this->Dot(*this); }

    friend VPoint operator*(const VPoint& a, float scalar) { return {a.x * scalar, a.y * scalar}; }

    friend VPoint operator*(float scalar, const VPoint& a) { return {a.x * scalar, a.y * scalar}; }

    VPoint Normalized() { return (*this) * (1.0f / this->Length()); }

    friend VPoint operator+(const VPoint& a, float scalar) { return {a.x + scalar, a.y + scalar}; }

    friend VPoint operator-(const VPoint& a, float scalar) { return {a.x + scalar, a.y + scalar}; }

    friend VPoint operator-(const VPoint& a) { return {-a.x, -a.y}; }

    friend VPoint operator*(const VPoint& a, const VPoint& b) { return {a.x * b.x, a.y * b.y}; }

    friend VPoint operator/(const VPoint& a, const VPoint& b) { return {a.x / b.x, a.y / b.y}; }

    friend VPoint operator/(const VPoint& a, float scalar) { return {a.x / scalar, a.y / scalar}; }

    friend VPoint operator+(const VPoint& a, const VPoint& b) { return {a.x + b.x, a.y + b.y}; }

    friend VPoint operator-(const VPoint& a, const VPoint& b) { return {a.x - b.x, a.y - b.y}; }

    friend bool operator==(const VPoint& a, const VPoint& b) { return a.x == b.x && a.y == b.y; }

    friend bool operator!=(const VPoint& a, const VPoint& b) { return !(a == b); }
};
std::ostream& operator<<(std::ostream& o, const VPoint& a);
