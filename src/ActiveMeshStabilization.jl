module ActiveMeshStabilization

using Optim
using ForwardDiff
using LinearAlgebra
using ThreadPools
using StaticArrays

import Optim: retract!, project_tangent!

export ActiveMeshStabilizer,
    edges_ideal_square,
    insufficient_edge_length_improvement,
    insufficient_triangle_compactness_improvement

_abs_square(x) = x ⋅ x
square_sides(x1, x2, x3) = (
    _abs_square(x1 .- x2),
    _abs_square(x2 .- x3),
    _abs_square(x3 .- x1)
)

_corner_vectors(triangle, vertices) = map(i -> vertices[i], triangle)
_vector(edge, vertices) = vertices[edge[2]] - vertices[edge[1]]
_area(x1, x2, x3) = 0.5norm((x2 .- x1) × (x3 .- x1))
compactness(area::Real, square_sides) = area / sum(square_sides)
edges_ideal_square(total_surface_area, nfaces) = 2total_surface_area / nfaces

norm_squared(x) = dot(x, x)

function scattering(edges, square_ideal_lengths, vertices)
    (norm_squared(vertices[i] - vertices[j]) / h2 for ((i, j), h2) in zip(edges, square_ideal_lengths)) |>
        x -> maximum(x) / minimum(x)
end

function compactness(triangle, vertices)
    x1, x2, x3 = _corner_vectors(triangle, vertices)
    _sqs = square_sides(x1, x2, x3)
    _a = _area(x1, x2, x3)
    return compactness(_a, _sqs)
end

potential_energy_edges(x, sqideal=1) =
    0.5( (x ⋅ x) / sqideal + sqideal / (x ⋅ x) )

function potential_energy(
    edges,
    triangles,
    vertices,
    square_ideal_edge_lengths,
    ideal_compactness = sqrt(3) / 12,
    p = 4,
    r = 8
)
    t1 = sum(zip(edges, square_ideal_edge_lengths)) do (edge, square_ideal)
        return potential_energy_edges(
            _vector(edge, vertices),
            square_ideal
        )^p
    end
    t2 = sum(triangles) do triangle
        return (ideal_compactness / compactness(triangle, vertices))^r
    end
    return t1 + t2
end

_area(vertices, triangle) = _area(_corner_vectors(triangle, vertices)...)

total_surface_area(vertices, faces) =
    sum(_area(vertices, face) for face in faces)

function normalized_constraint_normals(vertices, faces)
    points = vertices |> collect
    constraints = tmap(enumerate(points)) do (i, point)
        res = ForwardDiff.gradient(point) do xs
            _vertices =
                vcat((@view points[1:i - 1]), [xs], (@view points[i + 1:end]))
            return total_surface_area(_vertices, faces)
        end
        return normalize(res)
    end
    return constraints
end

struct NormalConstraint{
    V <: AbstractVector,
    W <: AbstractVector,
    S
} <: Optim.Manifold
    vertices::V
    protected_indices::W
    faces::S
end

function project_tangent!(m::NormalConstraint, gradf, vertices)
    vs = (SVector(vertices[i], vertices[i + 1], vertices[i + 2]) for i in 1:3:length(vertices))
    normals = normalized_constraint_normals(vs, m.faces) |>
        ns -> reduce(vcat, ns)
    for i in 1:3:length(gradf)
        n = view(normals, i:i + 2)
        gradf[i:i + 2] .-= (n ⋅ view(gradf, i:i + 2)) .* n
        # just subtract the normal component
    end
    for i in m.protected_indices
        gradf[3i - 2:3i] .= zero(eltype(gradf))
    end
    return gradf
end

function retract!(m::NormalConstraint, vertices)
    # NOTE this is not very rigorous, ideally we would solve for energy here
    vs = (SVector(vertices[i], vertices[i + 1], vertices[i + 2]) for i in 1:3:length(vertices))
    normals = normalized_constraint_normals(vs, m.faces) |>
        ns -> reduce(vcat, ns)
    for i in 1:3:length(normals)
        n = view(normals, i:i + 2)
        v = view(vertices, i:i + 2)
        v0 = view(m.vertices, i:i + 2)
        v .-= (n ⋅ (v .- v0)) .* n
        # restore normal components
    end
    for i in m.protected_indices
        vertices[3i - 2:3i] .= m.vertices[3i - 2:3i]
    end
    return vertices
end

struct ActiveMeshStabilizer
    square_ideal_edge_lengths
    edges
    faces
    constrained
    options
end

NormalConstraint(stabilizer::ActiveMeshStabilizer, velocities0) =
    NormalConstraint(
        velocities0,
        stabilizer.constrained,
        stabilizer.faces
    )

function objective(vertices::AbstractVector, stabilizer::ActiveMeshStabilizer)
    vs = map(1:3:length(vertices)) do i
        return SVector(
            vertices[i],
            vertices[i + 1],
            vertices[i + 2]
        )
    end
    return potential_energy(
        stabilizer.edges,
        stabilizer.faces,
        vs,
        stabilizer.square_ideal_edge_lengths
    )
end

function (stabilizer::ActiveMeshStabilizer)(vertices0)
    _constraint = NormalConstraint(stabilizer, copy(vertices0))
    obj0 = objective(vertices0, stabilizer)
    local vertices1
    try
        res = optimize(
            vertices -> objective(vertices, stabilizer),
            vertices0,
            BFGS(manifold = _constraint, alphaguess = 1e-6),
            #GradientDescent(manifold = _constraint, alphaguess = initial_scaling),
            stabilizer.options,
            autodiff = :forward
        )
        vertices1 = Optim.minimizer(res)
    catch exc
        println(exc)
        println("active stabilization: error during optimization")
        vertices1 = vertices0
    end
    obj1 = objective(vertices1, stabilizer)
    # dont make things worse
    @show obj1 < obj0
    return ifelse(obj1 < obj0, vertices1, vertices0)
end

insufficient_edge_length_improvement(
    candidate_vertices,
    status_quo_vertices,
    all_edges,
    all_squared_ideal_edge_lengths,
) = scattering(
            all_edges,
            all_squared_ideal_edge_lengths,
            candidate_vertices
        ) > 0.75scattering(
            all_edges,
            all_squared_ideal_edge_lengths,
            status_quo_vertices
        )

minimimal_compactness(vertices, faces) =
    minimum(compactness(triangle, vertices) for triangle in faces)

insufficient_triangle_compactness_improvement(
    candidate_vertices,
    status_quo_vertices,
    triangle_faces,
) = minimimal_compactness(candidate_vertices, triangle_faces) <
            1.15minimimal_compactness(status_quo_vertices, triangle_faces)

end # module ActiveMeshStabilization
