#pragma once

#include "lajolla.h"
#include "frame.h"
#include "ray.h"
#include "spectrum.h"
#include "vector.h"

#include <optional>

struct Scene;

/// We allow non-reciprocal BRDFs, so it's important
/// to distinguish which direction we are tracing the rays.
enum class TransportDirection
{
    TO_LIGHT,
    TO_VIEW
};

enum class VertexType { None, Camera, Light, Surface, Medium };

struct Bound3
{
    Vector3 minPos;
    Vector3 maxPos;

    Bound3()
    {
        minPos = Vector3(std::numeric_limits<Real>::infinity(), std::numeric_limits<Real>::infinity(), std::numeric_limits<Real>::infinity());
        maxPos = -minPos;
    }

    Bound3(const Vector3& minn, const Vector3& maxx)
        : minPos(minn), maxPos(maxx)
    {
      
    }

    Bound3 Union(const Bound3& box) const
    {
        Vector3 minn = minVec(minPos, box.minPos);
        Vector3 maxx = maxVec(maxPos, box.maxPos);
        return Bound3{ minn, maxx };
    }

    Bound3 Union(const Vector3& pos) const
    {
        Vector3 minn = minVec(minPos, pos);
        Vector3 maxx = maxVec(maxPos, pos);
        return Bound3{ minn, maxx };
    }

};


/// An "PathVertex" represents a vertex of a light path.
/// We store the information we need for computing any sort of path contribution & sampling density.
struct PathVertex {
    VertexType vertexType = VertexType::None;
    Vector3 position{};
    Vector3 geometric_normal{}; // always face at the same direction at shading_frame.n
    Frame shading_frame{};
    Vector2 st; // A 2D parametrization of the surface. Irrelavant to UV mapping.
                // for triangle this is the barycentric coordinates, which we use
                // for interpolating the uv map.
    Vector2 uv; // The actual UV we use for texture fetching.
    // For texture filtering, stores approximatedly min(abs(du/dx), abs(dv/dx), abs(du/dy), abs(dv/dy))
    Real uv_screen_size;
    Real mean_curvature; // For ray differential propagation.
    Real ray_radius; // For ray differential propagation.
    int shape_id = -1;
    int primitive_id = -1; // For triangle meshes. This indicates which triangle it hits.
    int material_id = -1;

    // If the path vertex is inside a medium, these two IDs
    // are the same.
    int interior_medium_id = -1;
    int exterior_medium_id = -1;
    int medium_id = -1;

    int light_id = -1;
    Spectrum beta;
    Spectrum throughput;
    Real pdfSampleNext = 1;
    Real pdfFwd = 0, pdfRev = 0;
    ComponentType componentType = ComponentType::None;
};

/// Intersect a ray with a scene. If the ray doesn't hit anything,
/// returns an invalid optional output. 
std::optional<PathVertex> intersect(const Scene &scene,
                                    const Ray &ray,
                                    const RayDifferential &ray_diff = RayDifferential{});

/// Test is a ray segment intersect with anything in a scene.
bool occluded(const Scene &scene, const Ray &ray);

/// Computes the emission at a path vertex v, with the viewing direction
/// pointing outwards of the intersection.
Spectrum emission(const PathVertex &v,
                  const Vector3 &view_dir,
                  const Scene &scene);

Spectrum vertex_f(const PathVertex& v,
    const Vector3& win,
    const Vector3& wout,
    const Scene& scene,
    TransportDirection dir);

Real vertex_pdf(const PathVertex& v, const PathVertex& prev, const PathVertex& next,
    const Scene& scene, TransportDirection dir);

inline Real convert_pdf(Real pdf, const PathVertex& from, const PathVertex& to)
{
    Vector3 w = from.position - to.position;
    if (length_squared(w) == 0) return 0;
    Real newPdf = pdf / length_squared(w);
    if (length(to.geometric_normal) != 0)
    {
        newPdf *= std::abs(dot(to.geometric_normal, normalize(w)));
    }
    return newPdf;
}


bool is_connectible(const PathVertex& v);

Real corrrect_shading_normal(const PathVertex& vertex, const Vector3& wo,
    const Vector3& wi, TransportDirection mode);