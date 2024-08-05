

#include "VPoint.h"

std::ostream& operator << (std::ostream& o, const VPoint& a) {
    o << "{ x: " << a.x << ", y: " << a.y << " }";
    return o;
}
