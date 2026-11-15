// The canopy volume's index from scatter cell to tree (canopy.rs): each tree writes its index
// + 1 into the cell it stands in. scatter.wgsl puts at most one tree in a cell, so a cell's
// trees are found with one lookup.

struct CanopyStatic {
    kinds: array<vec4f, 12>,
    dims: vec4u,   // x cells per side, y tree count
    reach: vec4u,
    misc: vec4f,   // x cell size (m)
}

@group(0) @binding(0) var<uniform> CS: CanopyStatic;
@group(0) @binding(1) var<storage, read> trees: array<vec4u>;
@group(0) @binding(2) var<storage, read_write> cells: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let t = id.y * groups.x * 256u + id.x;
    if (t >= CS.dims.y) {
        return;
    }
    let base = bitcast<vec3f>(trees[t].xyz);
    let dim = i32(CS.dims.x);
    let c = vec2i(floor(base.xz / CS.misc.x));
    if (any(c < vec2i(0)) || any(c >= vec2i(dim))) {
        return;
    }
    cells[c.y * dim + c.x] = t + 1u;
}
