use kurbo::{PathEl, Stroke, Cap, BezPath, Point, Join};


const K_MOVE: u8 = 0u8;
const K_LINE: u8 = 1u8; 
const K_QUAD: u8 = 2u8; 
const K_CLOSE: u8 = 3u8; 
const K_CUBIC: u8 = 4u8; 

struct PathC<'a> {
    points: &'a [f32],
    segment_types: &'a [u8],
    segment_count: usize,
    point_count: usize,
}

#[repr(C)]
pub struct Arrays {
    types: *mut u8,
    n_elements: usize,
    points: *mut f32,
    n_points: usize,
    error: u32,
}

#[repr(C)]
pub struct StrokeStyle {
    tolerance: f32,
    width: f32,
    line_join: u32,
    line_cap: u32,
    dashes: *mut f64,
    dash_size: usize,
}

impl Iterator for PathC<'_> {
    // We can refer to this type using Self::Item
    type Item = PathEl;

    fn next(&mut self) -> Option<Self::Item> {
        if self.segment_count >= self.segment_types.len() {
            return None; // Ensuring we don't go out of bounds
        }
    
        let ret = match self.segment_types[self.segment_count] {
            K_MOVE => {
                let point = Point::new(self.points[self.point_count] as f64, self.points[self.point_count + 1] as f64);
                self.point_count += 2;
                Some(PathEl::MoveTo(point))
            },
            K_LINE => { 
                let point = Point::new(self.points[self.point_count] as f64, self.points[self.point_count + 1] as f64);
                self.point_count += 2;
                Some(PathEl::LineTo(point))
            },
            K_QUAD => {
                let p1 = Point::new(self.points[self.point_count] as f64, self.points[self.point_count + 1] as f64);
                let p2 = Point::new(self.points[self.point_count + 2] as f64, self.points[self.point_count + 3] as f64);
                self.point_count += 4;
                Some(PathEl::QuadTo(p1, p2))
            },
            K_CUBIC => {
                let p1 = Point::new(self.points[self.point_count] as f64, self.points[self.point_count + 1] as f64);
                let p2 = Point::new(self.points[self.point_count + 2] as f64, self.points[self.point_count + 3] as f64);
                let p3 = Point::new(self.points[self.point_count + 4] as f64, self.points[self.point_count + 5] as f64);
                self.point_count += 6;
                Some(PathEl::CurveTo(p1, p2, p3))
            },
            K_CLOSE => Some(PathEl::ClosePath),
            _ => None,      
        };

        self.segment_count += 1;
        ret
    }    
}


pub fn write_path(path: &BezPath, points: &mut Vec<f32>, types: &mut Vec<u8>) {
    for el in path.elements() {
        match el {
            PathEl::MoveTo(p) => {
                types.push(K_MOVE);
                points.extend_from_slice(&[p.x as f32, p.y as f32]);
            },
            PathEl::LineTo(p) => {
                types.push(K_LINE);
                points.extend_from_slice(&[p.x as f32, p.y as f32]);
            },
            PathEl::QuadTo(p1, p2) => {
                types.push(K_QUAD);
                points.extend_from_slice(&[p1.x as f32, p1.y as f32, p2.x as f32, p2.y as f32]);
            },
            PathEl::CurveTo(p1, p2, p3) => {
                types.push(K_CUBIC);
                points.extend_from_slice(&[p1.x as f32, p1.y as f32, p2.x as f32, p2.y as f32, p3.x as f32, p3.y as f32]);
            },
            PathEl::ClosePath => {
                types.push(K_CLOSE);
            },
        }
    }
}

#[no_mangle]
pub extern "C" fn test_call() -> u32 {
    let mut v = Vec::new();
    v.push(42u32);
    v[0]
}

#[no_mangle]
pub extern "C" fn free_arrays(arrays: Arrays) {
    unsafe {
        Vec::from_raw_parts(arrays.types, arrays.n_elements, arrays.n_elements);
        Vec::from_raw_parts(arrays.points, arrays.n_points, arrays.n_points);
    }
}

#[no_mangle]
pub extern "C" fn kurbo_stroke(points_raw: *const f32, n_points: i32, types_raw: *const u8, n_elements: i32, c_stroke_style: StrokeStyle) -> Arrays {
    let result = std::panic::catch_unwind(|| {
        let points = unsafe { std::slice::from_raw_parts(points_raw, n_points as usize) };
        let types = unsafe { std::slice::from_raw_parts(types_raw, n_elements as usize )};
        let path_c = PathC {
            points: &points,
            segment_types: &types,
            segment_count: 0,
            point_count: 0
        };

        let cap = match c_stroke_style.line_cap {
            0u32 => Cap::Butt,
            1u32 => Cap::Round,
            2u32 => Cap::Square,
            _ => Cap::Butt,
        };

        let join = match c_stroke_style.line_join {
            0u32 => Join::Miter,
            1u32 => Join::Bevel,
            2u32 => Join::Round,
            _ => Join::Bevel,
        };

        let mut stroke_style: Stroke = Stroke::new(c_stroke_style.width as f64).with_start_cap(cap).with_end_cap(cap).with_join(join);
        if c_stroke_style.dash_size > 0usize {
            let dashes = unsafe { std::slice::from_raw_parts(c_stroke_style.dashes, c_stroke_style.dash_size as usize) };
            stroke_style = stroke_style.with_dashes(0.0, dashes);
        }

        let stroked_path: BezPath = kurbo::stroke(path_c.into_iter(), &stroke_style, &Default::default(), c_stroke_style.tolerance as f64);
        let mut out_points = Vec::new();
        let mut out_types = Vec::new();
        write_path(&stroked_path, &mut out_points, &mut out_types);
        let arrays = Arrays {
            error: 0u32,
            types: out_types.as_ptr() as *mut u8,
            n_elements: out_types.len(),
            points: out_points.as_ptr() as *mut f32,
            n_points: out_points.len(),
        };
        // Prevent Rust from cleaning up the vectors
        std::mem::forget(out_points);
        std::mem::forget(out_types);
        arrays
    });

    if result.is_err() {
        return Arrays {
            error: 1u32,
            types: std::ptr::null_mut(),
            n_elements: 0usize,
            points: std::ptr::null_mut(),
            n_points: 0usize
        };
    }

    result.unwrap()
}
