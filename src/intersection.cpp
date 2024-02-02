#include "intersection.h"
#include "material.h"
#include "ray.h"
#include "scene.h"
#include <embree3/rtcore.h>

std::optional<PathVertex> intersect(const Scene &scene,
                                    const Ray &ray,
                                    const RayDifferential &ray_diff) {
    RTCIntersectContext rtc_context;
    rtcInitIntersectContext(&rtc_context);
    RTCRayHit rtc_rayhit;
    RTCRay &rtc_ray = rtc_rayhit.ray;
    RTCHit &rtc_hit = rtc_rayhit.hit;
    rtc_ray = RTCRay{
        (float)ray.org.x, (float)ray.org.y, (float)ray.org.z,
        (float)ray.tnear,
        (float)ray.dir.x, (float)ray.dir.y, (float)ray.dir.z,
        0.f, // time
        (float)ray.tfar,
        (unsigned int)(-1), // mask
        0, // ray ID
        0  // ray flags
    };
    rtc_hit = RTCHit{
        0, 0, 0, // Ng_x, Ng_y, Ng_z
        0, 0, // u, v
        RTC_INVALID_GEOMETRY_ID, // primitive ID
        RTC_INVALID_GEOMETRY_ID, // geometry ID
        {RTC_INVALID_GEOMETRY_ID} // instance IDs
    };
    rtcIntersect1(scene.embree_scene, &rtc_context, &rtc_rayhit);
    if (rtc_hit.geomID == RTC_INVALID_GEOMETRY_ID) {
        return {};
    };
    assert(rtc_hit.geomID < scene.shapes.size());

    PathVertex vertex;
    vertex.position = Vector3{ray.org.x, ray.org.y, ray.org.z} +
        Vector3{ray.dir.x, ray.dir.y, ray.dir.z} * Real(rtc_ray.tfar);
    vertex.geometric_normal = normalize(Vector3{rtc_hit.Ng_x, rtc_hit.Ng_y, rtc_hit.Ng_z});
    vertex.shape_id = rtc_hit.geomID;
    vertex.primitive_id = rtc_hit.primID;
    const Shape &shape = scene.shapes[vertex.shape_id];
    vertex.material_id = get_material_id(shape);
    vertex.interior_medium_id = get_interior_medium_id(shape);
    vertex.exterior_medium_id = get_exterior_medium_id(shape);
    vertex.st = Vector2{rtc_hit.u, rtc_hit.v};

    ShadingInfo shading_info = compute_shading_info(scene.shapes[vertex.shape_id], vertex);
    vertex.shading_frame = shading_info.shading_frame;
    vertex.uv = shading_info.uv;
    vertex.mean_curvature = shading_info.mean_curvature;
    vertex.ray_radius = transfer(ray_diff, distance(ray.org, vertex.position));
    // vertex.ray_radius stores approximatedly dp/dx, 
    // we get uv_screen_size (du/dx) using (dp/dx)/(dp/du)
    vertex.uv_screen_size = vertex.ray_radius / shading_info.inv_uv_size;

    // Flip the geometry normal to the same direction as the shading normal
    if (dot(vertex.geometric_normal, vertex.shading_frame.n) < 0) {
        vertex.geometric_normal = -vertex.geometric_normal;
    }

    return vertex;
}

bool occluded(const Scene &scene, const Ray &ray) {
    RTCIntersectContext rtc_context;
    rtcInitIntersectContext(&rtc_context);
    RTCRay rtc_ray;
    rtc_ray.org_x = (float)ray.org[0];
    rtc_ray.org_y = (float)ray.org[1];
    rtc_ray.org_z = (float)ray.org[2];
    rtc_ray.dir_x = (float)ray.dir[0];
    rtc_ray.dir_y = (float)ray.dir[1];
    rtc_ray.dir_z = (float)ray.dir[2];
    rtc_ray.tnear = (float)ray.tnear;
    rtc_ray.tfar = (float)ray.tfar;
    rtc_ray.mask = (unsigned int)(-1);
    rtc_ray.time = 0.f;
    rtc_ray.flags = 0;
    // TODO: switch to rtcOccluded16
    rtcOccluded1(scene.embree_scene, &rtc_context, &rtc_ray);
    return rtc_ray.tfar < 0;
}

Spectrum emission(const PathVertex &v,
                  const Vector3 &view_dir,
                  const Scene &scene) {
    int light_id = (v.light_id == -1) ? get_area_light_id(scene.shapes[v.shape_id]) : v.light_id;
    assert(light_id >= 0);
    const Light &light = scene.lights[light_id];
    return emission(light,
                    view_dir,
                    v.uv_screen_size,
                    PointAndNormal{v.position, v.geometric_normal},
                    scene);
}

Spectrum vertex_f(const PathVertex& v, const Vector3& win,
    const Vector3& wout, const Scene& scene, TransportDirection dir)
{
    if (v.vertexType == VertexType::Medium)
    {
        const Medium& medium = scene.media[v.medium_id];
        PhaseFunction phaseFunction = get_phase_function(medium);
        return eval(phaseFunction, win, wout);
    }
    else if (v.vertexType == VertexType::Surface)
    {
        const Material& mat = scene.materials[v.material_id];
        return eval(mat, win, wout, v, scene.texture_pool, dir) * corrrect_shading_normal(v, win, wout, dir);
    }
    assert(false);
    return make_zero_spectrum();
}

Real vertex_pdf(const PathVertex& v, const PathVertex& prev, const PathVertex& next,
    const Scene& scene, TransportDirection dir)
{
    if (v.vertexType == VertexType::Light)
    {
        const Light& light = scene.lights[v.light_id];
        PointAndNormal point_on_light{ v.position, v.geometric_normal };
        Real pdf_pos, pdf_dir;
        pdf_point_direction_on_light(light, point_on_light, next.position, scene, &pdf_pos, &pdf_dir);
        return convert_pdf(pdf_dir, v, next);
    }
    Real pdf = 0;
    if (v.vertexType == VertexType::Medium)
    {
        const Medium& medium = scene.media[v.medium_id];
        PhaseFunction phaseFunction = get_phase_function(medium);
        pdf = pdf_sample_phase(phaseFunction, normalize(prev.position - v.position),
            normalize(next.position - v.position));
    }
    else if (v.vertexType == VertexType::Surface)
    {
        const Material& mat = scene.materials[v.material_id];
        pdf = pdf_sample_bsdf(mat, normalize(prev.position - v.position),
            normalize(next.position - v.position), v, scene.texture_pool, dir);
    }
    return convert_pdf(pdf, v, next);
}

bool is_connectible(const PathVertex& v)
{
    return true;
}

Real corrrect_shading_normal(const PathVertex& vertex, const Vector3& wo,
    const Vector3& wi, TransportDirection mode)
{
    if (mode == TransportDirection::TO_VIEW)
    {
        Real num = std::abs(dot(wo, vertex.shading_frame.n)) * std::abs(dot(wi, vertex.geometric_normal));
        Real denom = std::abs(dot(wo, vertex.geometric_normal)) * std::abs(dot(wi, vertex.shading_frame.n));
        // wi is occasionally perpendicular to isect.shading.n; this is
        // fine, but we don't want to return an infinite or NaN value in
        // that case.
        if (denom == 0) return 0;
        return num / denom;
    }
    else
        return 1;
}