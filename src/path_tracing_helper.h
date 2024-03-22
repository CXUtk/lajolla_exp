#pragma once
#include "transform.h"
inline int updateMedium(Vector3 dir, const PathVertex& vertex, int medium_id)
{
    if (vertex.interior_medium_id != vertex.exterior_medium_id)
    {
        if (dot(dir, vertex.geometric_normal) > 0)
        {
            return vertex.exterior_medium_id;
        }
        else
        {
            return vertex.interior_medium_id;
        }
    }
    return medium_id;
}

inline TransportDirection reverse_dir(TransportDirection dir)
{
    if (dir == TransportDirection::TO_LIGHT) return TransportDirection::TO_VIEW;
    return TransportDirection::TO_LIGHT;
}

inline bool hasLight(const PathVertex& vertex, const Scene& scene)
{
    return (vertex.light_id != -1) || (vertex.vertexType != VertexType::Light && is_light(scene.shapes[vertex.shape_id]));
}

inline Ray spawn_ray(const Scene& scene, PathVertex vertex, const Vector3& dir)
{
    Ray ray;
    ray.tnear = 0;
    ray.tfar = infinity<Real>();
    if (vertex.vertexType == VertexType::Surface)
    {
        Vector3 N = dot(dir, vertex.geometric_normal) > 0 ? vertex.geometric_normal : -vertex.geometric_normal;
        Vector3 rayOrigin = vertex.position + N * get_intersection_epsilon(scene);
        ray.org = rayOrigin;
        ray.dir = dir;
    }
    else
    {
        ray.org = vertex.position;
        ray.dir = dir;
    }
    return ray;
}

inline Ray spawn_ray_to(const Scene& scene, const PathVertex& vertex, const PathVertex& target)
{
    Ray ray;
    ray.tnear = 0;
    ray.tfar = 1 - get_shadow_epsilon(scene);
    if (vertex.vertexType == VertexType::Surface)
    {
        Vector3 N = dot(target.position - vertex.position, vertex.geometric_normal) > 0 ? vertex.geometric_normal : -vertex.geometric_normal;
        Vector3 rayOrigin = vertex.position + N * get_intersection_epsilon(scene);
        ray.org = rayOrigin;
        ray.dir = target.position - rayOrigin;
    }
    else
    {
        ray.org = vertex.position;
        ray.dir = target.position - vertex.position;
    }
    return ray;
}

inline int get_medium(const PathVertex& vertex, const Scene& scene, const Vector3& dir)
{
    if (vertex.vertexType == VertexType::Camera || vertex.vertexType == VertexType::Medium)
    {
        return vertex.medium_id;
    }
    if (vertex.vertexType == VertexType::Light)
    {
        return get_medium_interface_light(scene.lights[vertex.light_id], PointAndNormal{ vertex.position, vertex.geometric_normal },
            dir, scene);
    }
    if (dot(dir, vertex.geometric_normal) >= 0)
    {
        return get_exterior_medium_id(scene.shapes[vertex.shape_id]);
    }
    return get_interior_medium_id(scene.shapes[vertex.shape_id]);
}

inline PathVertex CreateCamera(const Camera& camera, const Spectrum& beta)
{
    PathVertex v{};
    v.vertexType = VertexType::Camera;
    v.position = xform_point(camera.cam_to_world, Vector3{ 0, 0, 0 });
    v.beta = beta;
    v.throughput = make_const_spectrum(1.0);
    v.interior_medium_id = v.exterior_medium_id = v.medium_id = camera.medium_id;
    return v;
}

inline PathVertex CreateSurface(const PathVertex& vertex, const Spectrum& beta, Real pdf, const PathVertex& prev, int medium)
{
    PathVertex v = vertex;
    v.vertexType = VertexType::Surface;
    v.beta = beta;
    v.throughput = make_const_spectrum(1.0);
    v.medium_id = medium;

    v.pdfFwd = convert_pdf(pdf, prev, vertex);
    return v;
}

inline PathVertex CreateSurface(const PathVertex& vertex, const Spectrum& beta, const Spectrum& throughput, Real pdf, const PathVertex& prev, int medium, Real pdfSample = 1)
{
    PathVertex v = vertex;
    v.vertexType = VertexType::Surface;
    v.beta = beta;
    v.throughput = throughput;
    v.medium_id = medium;
    v.pdfSampleNext = pdfSample;

    v.pdfFwd = convert_pdf(pdf, prev, vertex);
    return v;
}

inline PathVertex CreateMedium(const Vector3& pos, int medium, const Spectrum& beta, Real pdf, const PathVertex& prev)
{
    PathVertex v{};
    v.vertexType = VertexType::Medium;
    v.position = pos;
    v.beta = beta;
    v.medium_id = medium;
    v.geometric_normal = Vector3{ 0,0,0 };
    v.shading_frame = Frame{};
    v.shading_frame.n = Vector3{ 0,0,0 };

    v.pdfFwd = convert_pdf(pdf, prev, v);
    return v;
}

inline PathVertex CreateLight(int lightIndex, const Vector3& position, const Spectrum& beta, Real pdf)
{
    PathVertex v{};
    v.vertexType = VertexType::Light;
    v.light_id = lightIndex;
    v.position = position;
    v.beta = beta;
    v.pdfFwd = pdf;
    return v;
}
