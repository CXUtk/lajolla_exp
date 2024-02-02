#pragma once
#include <transform.h>
#include "path_tracing_helper.h"

//Spectrum gdpt_trace(const Scene& scene,
//    int x, int y, /* pixel coordinates */
//    ReplayableSampler& sampler,
//    std::mutex& img_mutex,
//   
//)
//{
//    // Generate camera ray
//    int w = scene.camera.width, h = scene.camera.height;
//    Vector2 screen_pos((x + sampler.next_double()) / w,
//        (y + sampler.next_double()) / h);
//
//    PathTracingContext context{
//         scene,
//         rng,
//         sample_primary(scene.camera, screen_pos),
//         RayDifferential{ Real(0), Real(0) },
//         scene.camera.medium_id,
//         0
//    };
//
//    Spectrum radiance = make_zero_spectrum();
//    Spectrum current_path_throughput = make_const_spectrum(Real(1.0));
//    Real dir_last_pdf = 0;
//    Vector3 nee_p_cache = Vector3{ 0, 0, 0 };
//    Spectrum multi_trans_pdf = make_const_spectrum(Real(1.0));
//    Spectrum multi_nee_pdf = make_const_spectrum(Real(1.0));
//    bool specularPath = true;
//
//    while (true)
//    {
//        bool scatter = false;
//        std::optional<PathVertex> vertex_ = intersect(context.scene, context.ray, context.rayDiff);
//        Spectrum transmittance = make_const_spectrum(1.0);
//        Spectrum trans_dir_pdf = make_const_spectrum(1.0);
//        Spectrum trans_nee_pdf = make_const_spectrum(1.0);
//
//        // If the edge is inside a medium, we evaluate the scattering
//        Spectrum weight = make_const_spectrum(1.0);
//        if (context.medium_id != -1)
//        {
//            Real tMax = std::numeric_limits<Real>::infinity();
//            if (vertex_)
//            {
//                tMax = length(vertex_->position - context.ray.org);
//            }
//
//            Real t;
//            // scatter = sample_vol_scattering(context, tMax, &t, &transmittance, &trans_pdf);
//            scatter = sample_vol_scattering_homogenized(context, tMax, &t, &transmittance, &trans_dir_pdf, &trans_nee_pdf);
//            if (scatter)
//            {
//                specularPath = false;
//            }
//            context.ray.org = context.ray.org + t * context.ray.dir;
//            multi_trans_pdf *= trans_dir_pdf;
//            multi_nee_pdf *= trans_nee_pdf;
//        }
//
//        current_path_throughput *= transmittance / average(trans_dir_pdf);
//
//
//        // If we hit a surface, accout for emission
//        if (!scatter && vertex_)
//        {
//            PathVertex vertex = *vertex_;
//            // We hit a light immediately. 
//            // This path has only two vertices and has contribution
//            // C = W(v0, v1) * G(v0, v1) * L(v0, v1)
//            if (is_light(context.scene.shapes[vertex.shape_id]))
//            {
//                if (specularPath)
//                {
//                    radiance += current_path_throughput *
//                        emission(vertex, -context.ray.dir, context.scene);
//                }
//                else
//                {
//                    int light_id = get_area_light_id(context.scene.shapes[vertex.shape_id]);
//                    assert(light_id >= 0);
//                    const Light& light = context.scene.lights[light_id];
//                    Vector3 dir_light = normalize(vertex.position - nee_p_cache);
//                    Real G = max(-dot(dir_light, vertex.geometric_normal), Real(0)) /
//                        distance_squared(vertex.position, nee_p_cache);
//
//                    PointAndNormal point_on_light = { vertex.position, vertex.geometric_normal };
//                    Real pdf_nee = light_pmf(context.scene, light_id) * pdf_point_on_light(light, point_on_light, nee_p_cache, context.scene) * average(multi_nee_pdf);
//                    Real pdf_scatter = dir_last_pdf * average(multi_trans_pdf) * G;
//                    Real w = (pdf_scatter * pdf_scatter) / (pdf_nee * pdf_nee + pdf_scatter * pdf_scatter);
//                    radiance += current_path_throughput *
//                        emission(vertex, -context.ray.dir, context.scene) * w;
//                }
//            }
//        }
//
//        // If we cannot continue scattering
//        if (context.bounces == context.scene.options.max_depth - 1 && context.scene.options.max_depth != -1)
//        {
//            break;
//        }
//
//        // Account for surface scattering
//        if (!scatter && vertex_)
//        {
//            PathVertex vertex = *vertex_;
//            // index-matching interface, skip through it
//            if (vertex.material_id == -1)
//            {
//                // check which side of the surface we are hitting
//                context.medium_id = updateMedium(context.ray.dir, vertex, context.medium_id);
//
//                Vector3 N = dot(context.ray.dir, vertex.geometric_normal) > 0 ? vertex.geometric_normal : -vertex.geometric_normal;
//                Vector3 rayOrigin = vertex.position + N * get_intersection_epsilon(context.scene);
//                context.ray.org = rayOrigin;
//
//                context.bounces++;
//                continue;
//            }
//            else
//            {
//                // Handle normal surface scattering
//                // Sample direct lights
//                Spectrum pdf_trans_dir;
//                Spectrum pdf_trans_nee;
//                PointAndNormal point_on_light;
//                Spectrum L_nee = next_event_estimation_homogenized(context, vertex.position, vertex.geometric_normal,
//                    &pdf_trans_dir, &pdf_trans_nee, &point_on_light);
//
//                if (max(L_nee) > 0)
//                {
//                    const Material& mat = context.scene.materials[vertex.material_id];
//                    Vector3 win = -context.ray.dir;
//                    Vector3 wout = normalize(point_on_light.position - vertex.position);
//                    Spectrum f_bsdf = eval(mat, win, wout, vertex, context.scene.texture_pool);
//
//                    Real G = max(-dot(wout, point_on_light.normal), Real(0)) /
//                        distance_squared(point_on_light.position, vertex.position);
//
//                    Real p_bsdf = pdf_sample_bsdf(
//                        mat, win, wout, vertex, context.scene.texture_pool) * G * average(pdf_trans_dir);
//                    Real pdf_nee = average(pdf_trans_nee);
//                    Real w = (pdf_nee * pdf_nee) / (pdf_nee * pdf_nee + p_bsdf * p_bsdf);
//
//                    radiance += current_path_throughput * G * f_bsdf * L_nee / pdf_nee * w;
//                }
//
//                // scatter event
//                specularPath = false;
//
//                // Sample BSDF
//                BSDFSampleRecord record;
//                Real pdf_bsdf;
//                Spectrum f_bsdf = path_sample_bsdf(context, vertex, &record, &pdf_bsdf);
//                if (pdf_bsdf != 0)
//                {
//                    current_path_throughput *= f_bsdf / pdf_bsdf;
//                    dir_last_pdf = pdf_bsdf;
//                    nee_p_cache = vertex.position;
//                    multi_trans_pdf = make_const_spectrum(Real(1.0));
//                    multi_nee_pdf = make_const_spectrum(Real(1.0));
//
//                    Vector3 Wout = record.dir_out;
//                    Vector3 N = dot(Wout, vertex.geometric_normal) > 0 ? vertex.geometric_normal : -vertex.geometric_normal;
//                    Vector3 rayOrigin = vertex.position + N * get_intersection_epsilon(context.scene);
//                    context.medium_id = updateMedium(Wout, vertex, context.medium_id);
//                    context.ray.org = rayOrigin;
//                    context.ray.dir = Wout;
//                }
//            }
//        }
//
//        // Sample phase function
//        if (scatter)
//        {
//            Spectrum pdf_trans_dir;
//            Spectrum pdf_trans_nee;
//            PointAndNormal point_on_light;
//            Spectrum L_nee = next_event_estimation_homogenized(context, context.ray.org, Vector3{ 0, 0, 0 },
//                &pdf_trans_dir, &pdf_trans_nee, &point_on_light);
//            if (max(pdf_trans_nee) > 0)
//            {
//                Vector3 dir_light = normalize(point_on_light.position - context.ray.org);
//                Real G = max(-dot(dir_light, point_on_light.normal), Real(0)) /
//                    distance_squared(point_on_light.position, context.ray.org);
//                const Medium& medium = context.scene.media[context.medium_id];
//                PhaseFunction phaseFunction = get_phase_function(medium);
//                Spectrum f_nee = eval(phaseFunction, -context.ray.dir, dir_light);
//
//                // multiply by G because this time we put all pdf in area space
//                Real pdf_phase = pdf_sample_phase(phaseFunction, -context.ray.dir, dir_light) * G * average(pdf_trans_dir);
//                Real pdf_nee = average(pdf_trans_nee);
//                Real w = (pdf_nee * pdf_nee) / (pdf_nee * pdf_nee + pdf_phase * pdf_phase);
//                radiance += current_path_throughput * (f_nee * G * L_nee / pdf_nee * w) * get_sigma_s(medium, context.ray.org);
//                //printf("pdf_nee: %lf, pdf_phase: %lf\n", pdf_nee, pdf_phase);
//            }
//
//            // If we sampled a scattering event, trace according to phase function
//            Vector3 wout;
//            Real pdf_phase;
//            Spectrum L = path_sample_phase_function(context, &wout, &pdf_phase);
//            if (pdf_phase != 0)
//            {
//                current_path_throughput *= L / pdf_phase;
//                dir_last_pdf = pdf_phase;
//                nee_p_cache = context.ray.org;
//                multi_trans_pdf = make_const_spectrum(Real(1.0));
//                multi_nee_pdf = make_const_spectrum(Real(1.0));
//                context.ray.dir = wout;
//            }
//        }
//
//        Real rr_prob = 1.0;
//        if (context.bounces >= context.scene.options.rr_depth)
//        {
//            rr_prob = std::min(luminance(current_path_throughput), 0.95);
//            if (next_pcg32_real<Real>(context.rng) > rr_prob)
//            {
//                break;
//            }
//            else
//            {
//                current_path_throughput /= rr_prob;
//            }
//        }
//        context.bounces++;
//    }
//    return radiance;
//}