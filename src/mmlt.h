#pragma once
#include <transform.h>
#include "sampler.h"
#include "path_tracing_helper.h"

static const int cameraStreamIndex = 0;
static const int lightStreamIndex = 1;
static const int connectionStreamIndex = 2;
static const int nSampleStreams = 3;


Spectrum transmission_homogenized_mmlt(int medium_id,
    const Scene& scene,
    MLTSampler& sampler,
    const Vector3& start,
    const Vector3& end,
    int bounces,
    Spectrum* p_trans_nee)
{
    Spectrum T_light = make_const_spectrum(1.0);
    int shadow_medium = medium_id;
    int shadow_bounces = 0;
    Vector3 P = start;
    Vector3 End = end;
    RayDifferential diff{ Real(0), Real(0) };

    *p_trans_nee = make_const_spectrum(1.0);
    while (true)
    {
        Ray shadow_ray{ P, normalize(End - P),
               get_shadow_epsilon(scene),
               (1 - get_shadow_epsilon(scene)) * distance(End, P) };
        std::optional<PathVertex> vertex_ = intersect(scene, shadow_ray, diff);

        // compute travel distance
        Real next_t = distance(End, P);
        if (vertex_)
        {
            next_t = distance(vertex_->position, P);
        }
        // compute transmission and pdf for travel this distance
        if (shadow_medium != -1)
        {
            const Medium& medium = scene.media[shadow_medium];
            Spectrum majorant = get_majorant(medium, shadow_ray);
            Real max_majorant = max(majorant);
            int channel = (int)(floor(sampler.next_double() * 3.0));
            int iteration = 0;
            Real accum_t = 0;

            while (true)
            {
                if (majorant[channel] <= 0)
                {
                    break;
                }
                if (iteration >= scene.options.max_null_collisions)
                {
                    break;
                }

                Real u = sampler.next_double();
                Real t = -std::log(1 - u) / majorant[channel];
                Real remaining_t = next_t - accum_t;
                accum_t = std::min(accum_t + t, next_t);
                Vector3 currentPos = shadow_ray.org + shadow_ray.dir * accum_t;

                if (t < remaining_t)
                {
                    // didn¡¯t hit the surface, so this is a null-scattering event
                    Spectrum sigma_t = get_sigma_a(medium, currentPos) + get_sigma_s(medium, currentPos);
                    Spectrum sigma_n = majorant - sigma_t;
                    Spectrum real_prob = sigma_t / majorant;
                    T_light *= exp(-majorant * t) * sigma_n / max_majorant;
                    *p_trans_nee *= exp(-majorant * t) * majorant / max_majorant;
                    if (luminance(T_light) <= 0)
                    {
                        break;
                    }
                }
                else
                {
                    T_light *= exp(-majorant * remaining_t) / max_majorant;
                    *p_trans_nee *= exp(-majorant * remaining_t) / max_majorant;
                    break;
                }
                iteration++;
            }
        }

        // If nothing hit, we are done
        if (!vertex_)
        {
            break;
        }
        PathVertex vertex = *vertex_;
        // If blocked by a real object, then return 0
        if (vertex.material_id != -1)
        {
            return make_zero_spectrum();
        }
        else
        {

            // If it is fake surface
            // If we reach the maximum depth, return 0
            if (scene.options.max_depth != -1 && shadow_bounces + bounces + 1 >= scene.options.max_depth)
            {
                return make_zero_spectrum();
            }
            shadow_bounces++;
            shadow_medium = updateMedium(shadow_ray.dir, vertex, shadow_medium);

            //Vector3 N = dot(shadow_ray.dir, vertex.geometric_normal) > 0 ? vertex.geometric_normal : -vertex.geometric_normal;
            //Vector3 rayOrigin = vertex.position + N * get_intersection_epsilon(scene);
            //P = P + next_t * shadow_ray.dir;
            P = vertex.position;
        }
    }
    return T_light / *p_trans_nee;
}

Spectrum sample_direct_lighting_mmlt(const Scene& scene, MLTSampler& sampler,
    const PathVertex& vertex,
    Real* pdf_pos, Real* pdf_dir, PointAndNormal* point_on_light, int* lightId)
{
    // First, we sample a point on the light source.
            // We do this by first picking a light source, then pick a point on it.
    Vector2 light_uv{ sampler.next_double(), sampler.next_double() };
    Real light_w = sampler.next_double();
    Real shape_w = sampler.next_double();
    int light_id = sample_light(scene, light_w);
    *lightId = light_id;
    const Light& light = scene.lights[light_id];
    *point_on_light =
        sample_point_on_light(light, vertex.position, light_uv, shape_w, scene);

    Vector3 dir_light = normalize(point_on_light->position - vertex.position);
    int medium = vertex.medium_id;
    // If the point on light is occluded, G is 0. So we need to test for occlusion.
    // To avoid self intersection, we need to set the tnear of the ray
    // to a small "epsilon". We set the epsilon to be a small constant times the
    // scale of the scene, which we can obtain through the get_shadow_epsilon() function.
    Vector3 N = dot(dir_light, vertex.geometric_normal) > 0 ? vertex.geometric_normal : -vertex.geometric_normal;
    Vector3 rayOrigin = vertex.position + N * get_intersection_epsilon(scene);
    //Ray shadow_ray{
    //    rayOrigin,
    //    point_on_light->position - rayOrigin,
    //    0,
    //    1 - get_shadow_epsilon(scene)
    //};
    *pdf_pos = Real(0);
    *pdf_dir = Real(0);

    //if (occluded(scene, shadow_ray))
    //{
    //    return make_zero_spectrum();
    //}

    // geometry term is cosine at v_{i+1} divided by distance squared
    // this can be derived by the infinitesimal area of a surface projected on
    // a unit sphere -- it's the Jacobian between the area measure and the solid angle
    // measure.
    Real G = max(dot(-dir_light, point_on_light->normal), Real(0)) /
        distance_squared(point_on_light->position, vertex.position);

    // Before we proceed, we first compute the probability density p1(v1)
    // The probability density for light sampling to sample our point is
    // just the probability of sampling a light times the probability of sampling a point

    Real pdf_pos_t;
    pdf_point_direction_on_light(light, *point_on_light, vertex.position, scene, &pdf_pos_t, pdf_dir);

    Spectrum E = emission(light, -dir_light, Real(0), *point_on_light, scene);

    *pdf_pos = pdf_pos_t * light_pmf(scene, light_id);

    if (luminance(E) == 0)
    {
        return make_zero_spectrum();
    }

    Spectrum pdf_t;
    Spectrum T = transmission_homogenized_mmlt(medium, scene, sampler, rayOrigin, point_on_light->position, 0, &pdf_t);
    return E * G * T;
}

Spectrum path_sample_phase_function_mmlt(int medium_id, const Ray& ray, const Scene& scene,
    MLTSampler& sampler, Vector3* wout, Real* pdf, Real* pdfRev)
{
    const Medium& medium = scene.media[medium_id];
    Spectrum sigma_s = get_sigma_s(medium, ray.org);
    PhaseFunction phaseFunction = get_phase_function(medium);

    Vector2 phase_rnd_param = Vector2(sampler.next_double(), sampler.next_double());

    Vector3 win = -ray.dir;
    std::optional<Vector3> next_dir = sample_phase_function(phaseFunction, win, phase_rnd_param);
    if (next_dir)
    {
        *wout = *next_dir;
        *pdf = pdf_sample_phase(phaseFunction, win, *next_dir);
        *pdfRev = pdf_sample_phase(phaseFunction, *next_dir, win);
        return eval(phaseFunction, win, *next_dir);
    }
    *pdf = 0;
    return make_zero_spectrum();
}

bool sample_vol_scattering_homogenized_mmlt(int medium_id,
    const Ray& ray,
    MLTSampler& sampler,
    Real tMax,
    Real* t_sample,
    Spectrum* transmittance,
    Spectrum* trans_dir_pdf,
    const Scene& scene)
{
    const Medium& medium = scene.media[medium_id];
    Spectrum majorant = get_majorant(medium, ray);
    Real max_majorant = max(majorant);

    // Sample a channel for sampling
    int channel = (int)(floor(sampler.next_double() * 3.0));
    *t_sample = 0;
    int iteration = 0;

    while (true)
    {
        if (majorant[channel] <= 0)
        {
            break;
        }
        if (iteration >= scene.options.max_null_collisions)
        {
            break;
        }

        Real u = sampler.next_double();
        Real t = -std::log(1 - u) / majorant[channel];
        Real remaining_t = tMax - *t_sample;
        *t_sample = std::min(*t_sample + t, tMax);
        Vector3 currentPos = ray.org + ray.dir * (*t_sample);

        // haven¡¯t reached the surface
        if (t < remaining_t)
        {
            Spectrum sigma_t = get_sigma_a(medium, currentPos) + get_sigma_s(medium, currentPos);
            Spectrum sigma_n = majorant - sigma_t;
            Spectrum real_prob = sigma_t / majorant;
            // hit a "real" particle
            if (sampler.next_double() < real_prob[channel])
            {
                Spectrum prev_trans = *transmittance;
                *transmittance *= exp(-majorant * t) / max_majorant;
                *trans_dir_pdf *= exp(-majorant * t) * majorant * real_prob / max_majorant;
                return true;
            }
            else
            {
                // hit a "fake" particle
                Spectrum prev_trans = *transmittance;
                *transmittance *= exp(-majorant * t) * sigma_n / max_majorant;
                *trans_dir_pdf *= exp(-majorant * t) * majorant * (make_const_spectrum(1.0) - real_prob) / max_majorant;
            }
        }
        else
        {
            // reach the surface
            *transmittance *= exp(-majorant * remaining_t) / max_majorant;
            *trans_dir_pdf *= exp(-majorant * remaining_t) / max_majorant;
            return false;
        }
        iteration++;
    }
    return false;
}

void random_walk_mmlt(std::vector<PathVertex>& path, const Scene& scene, const Ray& initRay, const RayDifferential& rayDiff,
    MLTSampler& sampler,
    Spectrum beta, Real pdf,
    int maxDepth, TransportDirection direction)
{
    if (maxDepth == 0) return;

    int bounces = 0;
    // Declare variables for forward and reverse probability densities
    Real pdfFwd = pdf, pdfRev = 0;

    Ray currentRay = initRay;
    int currentMedium = path[0].medium_id;
    while (true)
    {
        std::optional<PathVertex> vertex_ = intersect(scene, currentRay, rayDiff);
        if (luminance(beta) == 0) break;

        bool scatter = false;
        if (currentMedium != -1)
        {
            Real tMax = std::numeric_limits<Real>::infinity();
            if (vertex_)
            {
                tMax = length(vertex_->position - currentRay.org);
            }

            Real t;
            Spectrum transmittance = make_const_spectrum(1.0);
            Spectrum trans_dir_pdf = make_const_spectrum(1.0);

            scatter = sample_vol_scattering_homogenized_mmlt(currentMedium, currentRay, sampler, tMax,
                &t, &transmittance, &trans_dir_pdf, scene);

            beta *= transmittance / average(trans_dir_pdf);


            if (scatter)
            {
                currentRay.org = currentRay.org + t * currentRay.dir;
                Spectrum sigma_s = get_sigma_s(scene.media[currentMedium], currentRay.org);
                beta *= sigma_s;

                PathVertex mi = CreateMedium(currentRay.org, currentMedium, beta, pdfFwd, path.back());
                path.push_back(mi);

                if (++bounces >= maxDepth)
                {
                    break;
                }
                PathVertex& curVertex = path.back();
                PathVertex& prevVertex = path[path.size() - 2];
                Vector3 wout;
                Real pdf_phase;
                Real pdf_rev;

                Spectrum L = path_sample_phase_function_mmlt(currentMedium, currentRay, scene, sampler, &wout, &pdf_phase, &pdf_rev);
                if (luminance(L) != 0 && pdf_phase != 0)
                {
                    pdfFwd = pdf_phase;
                    beta *= L / pdfFwd;
                    pdfRev = pdf_rev;

                    currentRay.dir = wout;
                }
                prevVertex.pdfRev = convert_pdf(pdfRev, curVertex, prevVertex);
                continue;
            }
        }

        if (vertex_ && isfinite(vertex_->position) && !scatter)
        {
            PathVertex& vertex = *vertex_;

            if (vertex.material_id == -1)
            {
                Vector3 N = dot(currentRay.dir, vertex.geometric_normal) > 0 ? vertex.geometric_normal : -vertex.geometric_normal;
                Vector3 rayOrigin = vertex.position + N * get_intersection_epsilon(scene);
                currentRay.org = rayOrigin;

                // check which side of the surface we are hitting
                currentMedium = updateMedium(currentRay.dir, vertex, currentMedium);
                ++bounces;
                continue;
            }


            // Construct vertex
            vertex = CreateSurface(vertex, beta, pdfFwd, path.back(), currentMedium);
            path.push_back(vertex);

            if (++bounces >= maxDepth)
            {
                break;
            }

            PathVertex& curVertex = path.back();
            PathVertex& prevVertex = path[path.size() - 2];

            const Material& mat = scene.materials[curVertex.material_id];
            // Let's do the hemispherical sampling next.
            Vector3 win = -currentRay.dir;
            Vector2 bsdf_rnd_param_uv{ sampler.next_double(), sampler.next_double() };
            Real bsdf_rnd_param_w = sampler.next_double();
            std::optional<BSDFSampleRecord> bsdf_sample_ =
                sample_bsdf(mat,
                    win,
                    curVertex,
                    scene.texture_pool,
                    bsdf_rnd_param_uv,
                    bsdf_rnd_param_w, direction);
            if (!bsdf_sample_)
            {
                break;
            }
            Real pdf_bsdf = pdf_sample_bsdf(mat, win, bsdf_sample_->dir_out, curVertex, scene.texture_pool, direction);
            Spectrum f_bsdf = eval(mat, win, bsdf_sample_->dir_out, curVertex, scene.texture_pool, direction);


            pdfFwd = pdf_bsdf;
            beta *= f_bsdf / pdfFwd * corrrect_shading_normal(curVertex, win, bsdf_sample_->dir_out, direction);
            pdfRev = pdf_sample_bsdf(mat, bsdf_sample_->dir_out, win, curVertex, scene.texture_pool, reverse_dir(direction));

            currentRay = spawn_ray(scene, curVertex, bsdf_sample_->dir_out);
            prevVertex.pdfRev = convert_pdf(pdfRev, curVertex, prevVertex);
            currentMedium = updateMedium(currentRay.dir, curVertex, currentMedium);
        }
        else
        {
            // possible environment light vertex
            if (direction == TransportDirection::TO_LIGHT)
            {
                //PathVertex vertex;
                //vertex.vertexType = VertexType::Light;
                //vertex.beta = beta;
                //path.push_back(vertex);
            }
            break;
        }

    }
}

std::vector<PathVertex> gen_camera_subpath_mmlt(const Scene& scene,
    const Vector2& screenPos,
    MLTSampler& sampler,
    int maxDepth)
{
    std::vector<PathVertex> path;

    Ray ray = sample_primary(scene.camera, screenPos);
    RayDifferential ray_diff = RayDifferential{ Real(0), Real(0) };

    PathVertex cameraVertex = CreateCamera(scene.camera, make_const_spectrum(Real(1)));
    cameraVertex.geometric_normal = xform_vector(scene.camera.cam_to_world,
        Vector3{ 0, 0, 1 });
    cameraVertex.shading_frame = Frame(cameraVertex.geometric_normal);
    cameraVertex.medium_id = scene.camera.medium_id;
    path.push_back(cameraVertex);

    Real pdf_pos, pdf_dir;
    pdf_importance(scene.camera, ray.dir, &pdf_pos, &pdf_dir);
    
    random_walk_mmlt(path, scene, ray, ray_diff, sampler, cameraVertex.beta, pdf_dir, maxDepth - 1,
        TransportDirection::TO_LIGHT);
    return path;
}

std::vector<PathVertex> gen_light_subpath_mmlt(const Scene& scene,
    MLTSampler& sampler,
    int maxDepth)
{
    std::vector<PathVertex> path;
    if (maxDepth == 0)
    {
        return path;
    }

    RayDifferential ray_diff = RayDifferential{ Real(0), Real(0) };

    Vector2 light_uv{ sampler.next_double(), sampler.next_double() };
    Vector2 dir_uv{ sampler.next_double(), sampler.next_double() };
    Real light_w = sampler.next_double();
    Real shape_w = sampler.next_double();
    int light_id = sample_light(scene, light_w);
    const Light& light = scene.lights[light_id];
    Real pdf_light = light_pmf(scene, light_id);
    std::optional<LightLeSampleRecord> record = sample_point_direction_on_light(light,
        Vector3{ 0,0,0 }, light_uv, shape_w, dir_uv, scene);

    if (record)
    {
        Vector3 N = dot(record->dir_out, record->normal) > 0 ? record->normal : -record->normal;
        Vector3 rayOrigin = record->pos + N * get_intersection_epsilon(scene);
        Ray ray{ rayOrigin, record->dir_out, 0, infinity<Real>() };
        PathVertex v = CreateLight(light_id, record->pos, record->L, pdf_light * record->pdf_pos);
        v.geometric_normal = record->normal;
        v.shading_frame = Frame(v.geometric_normal);
        v.medium_id = get_medium_interface_light(light, PointAndNormal{ record->pos, record->normal }, record->dir_out, scene);
        path.push_back(v);

        Spectrum beta = (record->L * std::max(dot(record->normal, ray.dir), 0.0)) / (pdf_light * record->pdf_pos * record->pdf_dir);

        random_walk_mmlt(path, scene, ray, ray_diff, sampler, beta, record->pdf_dir, maxDepth - 1, TransportDirection::TO_VIEW);
    }
    return path;
}

Real MISWeight_mmlt(const Scene& scene, const std::vector<PathVertex>& cameraPath,
    const std::vector<PathVertex>& lightPath, int s, int t)
{
    if (s + t == 2) return 1;
    // return 1.0 / (s + t);
    auto remap0 = [](Real f) -> Real { return f != 0 ? f : 1; };
    Real sumRi = 0;
    // Consider hypothetical connection strategies along the camera subpath
    Real ri = 1;
    for (int i = t - 1; i > 0; --i)
    {
        ri *= remap0(cameraPath[i].pdfRev) / remap0(cameraPath[i].pdfFwd);
        assert(ri >= 0);
        sumRi += ri;
    }

    // Consider hypothetical connection strategies along the light subpath
    ri = 1;
    for (int i = s - 1; i >= 0; --i)
    {
        ri *= remap0(lightPath[i].pdfRev) / remap0(lightPath[i].pdfFwd);
        assert(ri >= 0);
        sumRi += ri;
    }
    assert(sumRi >= 0);
    return 1.0 / (1.0 + sumRi);
}

Spectrum connect_path_mmlt(const Scene& scene, MLTSampler& sampler, const std::vector<PathVertex>& cameraPath,
    const std::vector<PathVertex>& lightPath, int s, int t, Vector2* sampleCoord)
{
    Spectrum L = make_zero_spectrum();
    // Skip invalid connection: escaped ray
    if (t > 1 && s != 0 && cameraPath[t - 1].vertexType == VertexType::Light)
        return L;

    std::vector<PathVertex> cameraPath1;
    cameraPath1.assign(cameraPath.begin(), cameraPath.end());
    std::vector<PathVertex> lightPath1;
    lightPath1.assign(lightPath.begin(), lightPath.end());

    if (s == 0)
    {
        // Interpret the camera subpath as a complete path
        const PathVertex& pt = cameraPath1[t - 1];
        if (pt.shape_id >= 0 && is_light(scene.shapes[pt.shape_id]))
        {
            Vector3 dir = normalize(cameraPath1[t - 2].position - pt.position);
            L = pt.beta * emission(pt, dir, scene);
        }
    }
    else if (t == 1)
    {
        // Sample a point on the camera and connect it to the light subpath
        const PathVertex& qs = lightPath1[s - 1];
        /*Ray shadowRay{ cameraPath1[0].position, qs.position - cameraPath1[0].position, 0, 1 - get_shadow_epsilon(scene) };*/
        // Can shoot to the camera
        Vector3 dir = normalize(cameraPath1[0].position - qs.position);
        Vector2 pos;
        Real pdf_camera;
        Spectrum Wi = sample_importance(scene.camera, qs.position, qs.geometric_normal, &pdf_camera, &pos);

        if (pdf_camera != 0 && luminance(Wi) != 0)
        {
            cameraPath1[0].pdfRev = 0;
            cameraPath1[0].pdfFwd = 0;

            *sampleCoord = pos;
            Vector3 win = normalize(lightPath1[s - 2].position - qs.position);
            Spectrum f = vertex_f(qs, win, dir, scene, TransportDirection::TO_VIEW);
            Spectrum pdf_t;
            L = qs.beta * f * Wi / pdf_camera * transmission_homogenized_mmlt(cameraPath1[0].medium_id, scene, sampler,
                cameraPath1[0].position, qs.position, 0, &pdf_t);
        }

    }
    else if (s == 1)
    {
        // Sample a point on a light and connect it to the camera subpath
        PathVertex& pt = cameraPath1[t - 1];
        PathVertex& qs = lightPath1[s - 1];

        PointAndNormal point_on_light;
        Real pdf_pos, pdf_dir;
        int light_id;
        Spectrum Le = sample_direct_lighting_mmlt(scene, sampler, pt, &pdf_pos, &pdf_dir, &point_on_light, &light_id);

        Vector3 win = normalize(cameraPath1[t - 2].position - pt.position);

        if (luminance(Le) != 0)
        {
            qs.light_id = light_id;
            qs.position = point_on_light.position;
            qs.geometric_normal = point_on_light.normal;
            qs.shading_frame = Frame(point_on_light.normal);
            qs.pdfFwd = pdf_pos;
            qs.beta = Le / pdf_pos;
            L = pt.beta * vertex_f(pt, win, normalize(point_on_light.position - pt.position),
                scene, TransportDirection::TO_LIGHT) * qs.beta;
        }
    }
    else
    {
        // Handle all other bidirectional connection cases
        const PathVertex& pt = cameraPath1[t - 1];
        const PathVertex& qs = lightPath1[s - 1];
        if (is_connectible(pt) && is_connectible(qs))
        {
            Vector3 win_C = normalize(cameraPath1[t - 2].position - pt.position);
            Vector3 win_L = normalize(lightPath1[s - 2].position - qs.position);
            Vector3 wout_C = normalize(qs.position - pt.position);
            Vector3 wout_L = -wout_C;

            Real G = Real(1.0) / distance_squared(pt.position, qs.position);
            // Ray shadowRay = spawn_ray_to(scene, pt, qs);
            Spectrum pdf_t;
            L = qs.beta * vertex_f(qs, win_L, wout_L, scene, TransportDirection::TO_VIEW)
                * vertex_f(pt, win_C, wout_C, scene, TransportDirection::TO_LIGHT) * pt.beta;
            if (luminance(L) != 0)
            {
                L *= G * transmission_homogenized_mmlt(get_medium(pt, scene, wout_C), scene, sampler, pt.position, qs.position, 0, &pdf_t);
            }
        }
    }

    if (luminance(L) == 0)
    {
        return L;
    }

    //if (t > 1)
    //{
    //    printf("");
    //}

    if (t > 0)
    {
        PathVertex& cameraTail = cameraPath1[t - 1];
        // Has light vertices
        if (s > 0)
        {
            // If there are multiple light vertices, the pdf is by sampling bsdf
            Real pdf_dir = vertex_pdf(lightPath1[s - 1], s == 1 ? cameraTail : lightPath1[s - 2],
                cameraTail, scene, TransportDirection::TO_VIEW);
            cameraTail.pdfRev = pdf_dir;
        }
        else
        {
            // If there is no light vertices, sample the last camera vertex, use it as light source
            assert(is_light(scene.shapes[cameraTail.shape_id]));
            int light_id = get_area_light_id(scene.shapes[cameraTail.shape_id]);
            assert(light_id >= 0);
            const Light& light = scene.lights[light_id];

            //Real G = max(dot(normalize(cameraTail.position - cameraPath1[t - 2].position), cameraPath1[t - 2].geometric_normal), Real(0))
            //    / distance_squared(cameraPath1[t - 2].position, cameraTail.position);

            assert(t > 1);
            PointAndNormal point_on_light{ cameraTail.position, cameraTail.geometric_normal };
            cameraTail.pdfRev = pdf_point_on_light(light, point_on_light, cameraPath1[t - 2].position, scene)
                * light_pmf(scene, light_id);
        }
    }

    if (s > 0)
    {
        assert(t > 0);
        PathVertex& lightTail = lightPath1[s - 1];
        PathVertex& cameraTail = cameraPath1[t - 1];

        if (t == 1)
        {
            assert(cameraTail.vertexType == VertexType::Camera);
            // Only one camera vertex, then the pdfRev obtained from the camera sample
            Real pdf_pos, pdf_dir;
            pdf_importance(scene.camera, normalize(lightTail.position - cameraTail.position), &pdf_pos, &pdf_dir);
            lightTail.pdfRev = convert_pdf(pdf_dir, cameraTail, lightTail);
        }
        else
        {
            // Otherwise, we compute the pdf from bsdf
            Real pdf_dir = vertex_pdf(cameraTail, cameraPath1[t - 2],
                lightTail, scene, TransportDirection::TO_LIGHT);
            lightTail.pdfRev = pdf_dir;
        }
    }

    if (s > 1)
    {
        // Because the last vertex might change, we need to update the second tail's pdfRev
        assert(s > 1 && t > 0);
        PathVertex& lightTail = lightPath1[s - 1];
        PathVertex& lightTail2 = lightPath1[s - 2];
        Real pdf_dir = vertex_pdf(lightTail, cameraPath1[t - 1],
            lightTail2, scene, TransportDirection::TO_LIGHT);
        lightTail2.pdfRev = pdf_dir;
    }

    if (t > 1)
    {
        PathVertex& cameraTail = cameraPath1[t - 1];
        PathVertex& cameraTail2 = cameraPath1[t - 2];
        if (s > 0)
        {
            assert(t > 1 && s > 0);
            Real pdf_dir = vertex_pdf(cameraTail, lightPath1[s - 1],
                cameraTail2, scene, TransportDirection::TO_VIEW);
            cameraTail2.pdfRev = pdf_dir;
        }
        else
        {
            // If there is no light vertex
            int light_id = get_area_light_id(scene.shapes[cameraTail.shape_id]);
            assert(light_id != -1);
            const Light& light = scene.lights[light_id];
            PointAndNormal point_on_light{ cameraTail.position, cameraTail.geometric_normal };
            Real pdf_pos, pdf_dir;
            pdf_point_direction_on_light(light, point_on_light, cameraTail2.position, scene, &pdf_pos, &pdf_dir);
            cameraTail2.pdfRev = convert_pdf(pdf_dir, cameraTail, cameraTail2);
        }
    }

    Real weight = MISWeight_mmlt(scene, cameraPath1, lightPath1, s, t);
    //printf("Weight for s=%d, t=%d is %lf\n", s, t, weight);
    return L * weight;
}

Spectrum pssmlt_tracing(const Scene& scene,
    Vector2* pos_raster,/* pixel coordinates */
    MLTSampler& sampler,
    int depth
)
{
    // Start camera sample stream
    sampler.start_stream(cameraStreamIndex);

    // Determine strategy
    int s, t, nStrategies;
    if (depth == 0)
    {
        nStrategies = 1;
        s = 0;
        t = 2;
    }
    else
    {
        nStrategies = depth + 2;
        s = std::min((int)(sampler.next_double() * nStrategies), nStrategies - 1);
        t = nStrategies - s;
    }

    // Generate camera ray
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos(sampler.next_double(), sampler.next_double());

    *pos_raster = screen_pos;
   
    std::vector<PathVertex> cameraPath = gen_camera_subpath_mmlt(scene, screen_pos, sampler, t);
    if (cameraPath.size() != t)
    {
        //printf("Early stop: t, expected: %d, got: %d\n", t, cameraPath.size());
        return make_zero_spectrum();
    }

    sampler.start_stream(lightStreamIndex);
    std::vector<PathVertex> lightPath = gen_light_subpath_mmlt(scene, sampler, s);
    if (lightPath.size() != s)
    {
        // printf("Early stop: s, expected: %d, got: %d\n", s, lightPath.size());
        return make_zero_spectrum();
    }
    sampler.start_stream(connectionStreamIndex);
    return connect_path_mmlt(scene, sampler, cameraPath, lightPath, s, t, pos_raster) * (Real)nStrategies;

    //Spectrum L = make_zero_spectrum();

    ////printf("Begin Sample\n");
    //// length of a path is number of vertices
    //for (int t = 1; t <= cameraPath.size(); t++)
    //{
    //    for (int s = 0; s <= lightPath.size(); s++)
    //    {
    //        int depth = s + t - 2;

    //        // s = 1, t = 1 is skipped
    //        if ((s == 1 && t == 1) || depth < 0 || depth > scene.options.max_depth)
    //        {
    //            continue;
    //        }
    //        // if (s != 0 || t != 3) continue;
    //        // if (s != 0 || t != 3) continue;
    //        // if (s != 1) continue;
    //        Vector2 rasterCoord;
    //        Spectrum Lpath = connect_path_mmlt(scene, sampler, cameraPath, lightPath, s, t, &rasterCoord);

    //        if (t == 1)
    //        {
    //            std::lock_guard<std::mutex> lock(img_mutex);
    //            if (rasterCoord.x < 0 || rasterCoord.x > 1 || rasterCoord.y < 0 || rasterCoord.y > 1)
    //            {
    //                continue;
    //            }
    //            int x = rasterCoord.x * scene.camera.width;
    //            int y = rasterCoord.y * scene.camera.height;
    //            img_copy(x, y) += Lpath;
    //            img_copy_counter(x, y) = 1;
    //        }
    //        else
    //        {
    //            L += Lpath;
    //        }
    //    }
    //}
    ////printf("End Sample\n");
    //return L;
}