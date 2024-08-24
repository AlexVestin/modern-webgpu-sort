struct VSOutput {
    @builtin(position) position: vec4f,
};

struct Uniforms {
    view_size: vec2u,
    tile_size: u32,
    padding: u32
};

struct DrawSpan {
    position: u32,
    path_id: u32,
    line_start_index: u32,
    line_end_index: u32,
};

@group(0) @binding(0) var<storage, read> colors: array<u32>;
@group(0) @binding(1) var<storage, read> line_indices: array<u32>;
@group(0) @binding(2) var<storage, read> points: array<vec2f>;
@group(0) @binding(3) var<storage, read> draw_spans: array<DrawSpan>;
@group(0) @binding(4) var<storage, read> atlas_indices: array<u32>;
@group(0) @binding(5) var<uniform> uniforms: Uniforms;

@vertex
fn vert_main(@builtin(vertex_index) VertexIndex : u32) -> VSOutput {
    var out: VSOutput;
    let point = points[VertexIndex];

    let view_step = vec2f(1.0) / vec2f(uniforms.view_size);
    var p = point * view_step * 2.0 - 1.0;
    out.position = vec4<f32>(p.x, -p.y, 0.0, 1.0);
    return out;
}


@fragment
fn frag_main() -> @location(0) vec4<f32> {    
    return vec4f(0.0, 0.0, 1.0, 1.0);
}