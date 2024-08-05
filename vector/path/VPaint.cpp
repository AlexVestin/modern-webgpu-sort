// Copyright 2021, Alexander Vestin (alex.vestin@gmail.com)
#include "VPaint.h"

std::ostream & operator<<(std::ostream& o, const VColor& v) { 
    o << "{ r: " << v.color.r << ", g: " << v.color.g << ", b:" << v.color.b << ", a:" << v.color.a << " }";
    return o;
}