#include "render.h"
#include "intersection.h"
#include "material.h"
#include "parallel.h"
#include "path_tracing.h"
#include "vol_path_tracing.h"
#include "bidirectional_path_tracing.h"
#include "photon_mapping.h"
#include "pcg.h"
#include "progress_reporter.h"
#include "scene.h"
#include "mmlt.h"
#include "gdpt.h"
#include "restir_pt.h"
#include "distribution1D.h"
#include <3rdparty/cnpy.h>
#include <random>


/// Render auxiliary buffers e.g., depth.
Image3 aux_render(const Scene &scene) {
    int w = scene.camera.width, h = scene.camera.height;
    Image3 img(w, h);

    constexpr int tile_size = 16;
    int num_tiles_x = (w + tile_size - 1) / tile_size;
    int num_tiles_y = (h + tile_size - 1) / tile_size;

    parallel_for([&](const Vector2i &tile) {
        int x0 = tile[0] * tile_size;
        int x1 = min(x0 + tile_size, w);
        int y0 = tile[1] * tile_size;
        int y1 = min(y0 + tile_size, h);
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                Ray ray = sample_primary(scene.camera, Vector2((x + Real(0.5)) / w, (y + Real(0.5)) / h));
                RayDifferential ray_diff = init_ray_differential(w, h);
                if (std::optional<PathVertex> vertex = intersect(scene, ray, ray_diff)) {
                    Real dist = distance(vertex->position, ray.org);
                    Vector3 color{0, 0, 0};
                    if (scene.options.integrator == Integrator::Depth) {
                        color = Vector3{dist, dist, dist};
                    } else if (scene.options.integrator == Integrator::ShadingNormal) {
                        // color = (vertex->shading_frame.n + Vector3{1, 1, 1}) / Real(2);
                        color = vertex->shading_frame.n;
                    } else if (scene.options.integrator == Integrator::MeanCurvature) {
                        Real kappa = vertex->mean_curvature;
                        color = Vector3{kappa, kappa, kappa};
                    } else if (scene.options.integrator == Integrator::RayDifferential) {
                        color = Vector3{ray_diff.radius, ray_diff.spread, Real(0)};
                    } else if (scene.options.integrator == Integrator::MipmapLevel) {
                        const Material &mat = scene.materials[vertex->material_id];
                        const TextureSpectrum &texture = get_texture(mat);
                        auto *t = std::get_if<ImageTexture<Spectrum>>(&texture);
                        if (t != nullptr) {
                            const Mipmap3 &mipmap = get_img3(scene.texture_pool, t->texture_id);
                            Vector2 uv{modulo(vertex->uv[0] * t->uscale, Real(1)),
                                       modulo(vertex->uv[1] * t->vscale, Real(1))};
                            // ray_diff.radius stores approximatedly dpdx,
                            // but we want dudx -- we get it through
                            // dpdx / dpdu
                            Real footprint = vertex->uv_screen_size;
                            Real scaled_footprint = max(get_width(mipmap), get_height(mipmap)) *
                                                    max(t->uscale, t->vscale) * footprint;
                            Real level = log2(max(scaled_footprint, Real(1e-8f)));
                            color = Vector3{level, level, level};
                        }
                    }
                    img(x, y) = color;
                } else {
                    img(x, y) = Vector3{0, 0, 0};
                }
            }
        }
    }, Vector2i(num_tiles_x, num_tiles_y));

    return img;
}

Image3 path_render(const Scene &scene) {
    int w = scene.camera.width, h = scene.camera.height;
    Image3 img(w, h);

    constexpr int tile_size = 16;
    int num_tiles_x = (w + tile_size - 1) / tile_size;
    int num_tiles_y = (h + tile_size - 1) / tile_size;


    ProgressReporter reporter(num_tiles_x * num_tiles_y);
    parallel_for([&](const Vector2i &tile) {
        // Use a different rng stream for each thread.
        pcg32_state rng = init_pcg32(tile[1] * num_tiles_x + tile[0]);
        int x0 = tile[0] * tile_size;
        int x1 = min(x0 + tile_size, w);
        int y0 = tile[1] * tile_size;
        int y1 = min(y0 + tile_size, h);
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                Spectrum radiance = make_zero_spectrum();
                int spp = scene.options.samples_per_pixel;
                for (int s = 0; s < spp; s++) {
                    radiance += path_tracing(scene, x, y, rng);
                }

                img(x, y) = radiance / Real(spp);
            }
        }
        reporter.update(1);
    }, Vector2i(num_tiles_x, num_tiles_y));
    reporter.done();
    Real avgLuminance = 0;
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            avgLuminance += luminance(img(x, y));
        }
    }
    printf("Avg Luminance: %lf\n", avgLuminance / w / h);
    return img;
}

void atomicMax(std::atomic<Real>& atomicValue, Real value)
{
    Real oldValue = atomicValue.load();
    while (value > oldValue &&
        !atomicValue.compare_exchange_weak(oldValue, value));
}

Vector3 safeNormalize(const Vector3& v)
{
    if (length_squared(v) == 0)
    {
        return Vector3(0, 0, 0);
    }
    return normalize(v);
}

Image3 path_render_with_feature(const Scene& scene)
{
    int w = scene.camera.width, h = scene.camera.height;
    Image3 img(w, h);
    Image3 img_depth(w, h);
    Image3 img_normal(w, h);
    Image3 img_albedo_diffuse(w, h);
    Image3 img_color_diffuse(w, h);
    Image3 img_color_specular(w, h);
    Image3 img_color_variances(w, h);
    Image3 img_spp_diffuse(w, h);

    std::atomic<Real> max_dist = 0;

    constexpr int tile_size = 16;
    int num_tiles_x = (w + tile_size - 1) / tile_size;
    int num_tiles_y = (h + tile_size - 1) / tile_size;


    ProgressReporter reporter(num_tiles_x * num_tiles_y);
    parallel_for([&](const Vector2i& tile)
        {
            // Use a different rng stream for each thread.
            pcg32_state rng = init_pcg32(tile[1] * num_tiles_x + tile[0]);
            int x0 = tile[0] * tile_size;
            int x1 = min(x0 + tile_size, w);
            int y0 = tile[1] * tile_size;
            int y1 = min(y0 + tile_size, h);
            for (int y = y0; y < y1; y++)
            {
                for (int x = x0; x < x1; x++)
                {
                    int w = scene.camera.width, h = scene.camera.height;

                    Spectrum radiance_sp = make_zero_spectrum();
                    Spectrum radiance_diffuse = make_zero_spectrum();
                    Spectrum rad_diffuse2 = make_zero_spectrum();
                    int spp = scene.options.samples_per_pixel;

                    float depthSum = 0;
                    float depth2Sum = 0;
                    Vector3 normal = Vector3(0, 0, 0);
                    Vector3 normal2Sum = Vector3(0, 0, 0);
                    for (int s = 0; s < spp; s++)
                    {
                        TraceFeature traceFeature{};
                        Spectrum R = path_tracing_with_features(scene, x, y, rng, &traceFeature);

                        depthSum += traceFeature.depth;
                        depth2Sum += traceFeature.depth * traceFeature.depth;
                        normal += traceFeature.normal;
                        normal2Sum += traceFeature.normal * traceFeature.normal;

                        switch (traceFeature.componentType)
                        {
                        case ComponentType::Diffuse:
                            img_albedo_diffuse(x, y) = img_albedo_diffuse(x, y) + traceFeature.albedo;
                            img_spp_diffuse(x, y) = img_spp_diffuse(x, y) + Spectrum(1.0, 1.0, 1.0);
                            radiance_diffuse += R / (traceFeature.albedo + make_const_spectrum(0.00316));
                            rad_diffuse2 += R * R / (traceFeature.albedo + make_const_spectrum(0.00316)) / (traceFeature.albedo + make_const_spectrum(0.00316));
                            break;
                        case ComponentType::Specular:
                            radiance_sp += R;
                            break;
                        default:
                            break;
                        }
                    }

                    img_color_diffuse(x, y) = radiance_diffuse / Real(spp);
                    img_color_specular(x, y) = radiance_sp / Real(spp);
                    radiance_diffuse /= Real(spp);
                    rad_diffuse2 /= Real(spp);

                    img_depth(x, y) = make_const_spectrum(depthSum / Real(spp));
                    
                    img_normal(x, y) = safeNormalize(normal / Real(spp));

                    atomicMax(max_dist, depthSum / Real(spp));
                    img_color_variances(x, y).x = std::abs((depth2Sum / Real(spp)) - (depthSum / Real(spp)) * (depthSum / Real(spp)));
                    if (length(normal) != 0)
                    {
                        img_color_variances(x, y).y = length((normal2Sum / Real(spp) - (normal * normal) / Real(spp) / Real(spp)));
                    }
                    img_color_variances(x, y).z = luminance(rad_diffuse2 - radiance_diffuse * radiance_diffuse);
                }
            }
            reporter.update(1);
        }, Vector2i(num_tiles_x, num_tiles_y));

    parallel_for([&](const Vector2i& tile)
        {
            int x0 = tile[0] * tile_size;
            int x1 = min(x0 + tile_size, w);
            int y0 = tile[1] * tile_size;
            int y1 = min(y0 + tile_size, h);

            Real maxDepth = max_dist.load();
            for (int y = y0; y < y1; y++)
            {
                for (int x = x0; x < x1; x++)
                {
                    if (length(img_normal(x, y)) == 0)
                    {
                        img_depth(x, y) = make_const_spectrum(0);
                    }
                    else
                    {
                        img_depth(x, y) = (maxDepth == 0) ? Vector3(0, 0, 0) : (1.0 - (img_depth(x, y) / maxDepth));
                    }
                    img_color_variances(x, y).x = (maxDepth == 0) ? 0 : img_color_variances(x, y).x / maxDepth / maxDepth;
                    img_albedo_diffuse(x, y) = (img_spp_diffuse(x, y).x == 0) ?  Vector3(0, 0, 0) : img_albedo_diffuse(x, y) / img_spp_diffuse(x, y).x;
                }
            }
        }, Vector2i(num_tiles_x, num_tiles_y));

    reporter.done();
    
    imwrite("output_diffuse_color.exr", img_color_diffuse);
    imwrite("output_normal.exr", img_normal);
    imwrite("output_depth.exr", img_depth);
    imwrite("output_diffuse_albedo.exr", img_albedo_diffuse);
    imwrite("output_specular_color.exr", img_color_specular);
    imwrite("output_variance.exr", img_color_variances);
    return img;
}

Image3 vol_path_render(const Scene &scene) {
    int w = scene.camera.width, h = scene.camera.height;
    Image3 img(w, h);

    constexpr int tile_size = 16;
    int num_tiles_x = (w + tile_size - 1) / tile_size;
    int num_tiles_y = (h + tile_size - 1) / tile_size;

    //if (scene.options.vol_path_version == 1) {
    //    f = vol_path_tracing_1;
    //} else if (scene.options.vol_path_version == 2) {
    //    f = vol_path_tracing_2;
    //} else if (scene.options.vol_path_version == 3) {
    //    f = vol_path_tracing_3;
    //} else if (scene.options.vol_path_version == 4) {
    //    f = vol_path_tracing_4;
    //} else if (scene.options.vol_path_version == 5) {
    //    f = vol_path_tracing_5;
    //} else if (scene.options.vol_path_version == 6) {
    //    f = vol_path_tracing;
    //}

    ProgressReporter reporter(num_tiles_x * num_tiles_y);
    parallel_for([&](const Vector2i &tile) {
        // Use a different rng stream for each thread.
        pcg32_state rng = init_pcg32(tile[1] * num_tiles_x + tile[0]);
        ReplayableSampler sampler(rng);
        int x0 = tile[0] * tile_size;
        int x1 = min(x0 + tile_size, w);
        int y0 = tile[1] * tile_size;
        int y1 = min(y0 + tile_size, h);
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                Spectrum radiance = make_zero_spectrum();
                int spp = scene.options.samples_per_pixel;
                for (int s = 0; s < spp; s++) {
                    sampler.start_iteration();
                    Spectrum L = vol_path_tracing(scene, x, y, sampler);
                    if (isfinite(L)) {
                        // Hacky: exclude NaNs in the rendering.
                        radiance += L;
                    }
                }
                img(x, y) = radiance / Real(spp);
            }
        }
        reporter.update(1);
    }, Vector2i(num_tiles_x, num_tiles_y));
    reporter.done();
    return img;
}


Image3 bi_directional_render(const Scene& scene)
{
    int w = scene.camera.width, h = scene.camera.height;
    Image3 img(w, h);
    Image1 img_counter(w, h);
    Image3 img_copy(w, h);
    Image1 img_copy_counter(w, h);
    std::mutex img_mutex;

    constexpr int tile_size = 16;
    int num_tiles_x = (w + tile_size - 1) / tile_size;
    int num_tiles_y = (h + tile_size - 1) / tile_size;

    ProgressReporter reporter(num_tiles_x * num_tiles_y);
    parallel_for([&](const Vector2i& tile)
        {
            // Use a different rng stream for each thread.
            pcg32_state rng = init_pcg32(tile[1] * num_tiles_x + tile[0]);
            int x0 = tile[0] * tile_size;
            int x1 = min(x0 + tile_size, w);
            int y0 = tile[1] * tile_size;
            int y1 = min(y0 + tile_size, h);
            for (int y = y0; y < y1; y++)
            {
                for (int x = x0; x < x1; x++)
                {
                    Spectrum radiance = make_zero_spectrum();
                    int spp = scene.options.samples_per_pixel;
                    for (int s = 0; s < spp; s++)
                    {
                        Spectrum L = bidirectional_path_tracing(scene, x, y, rng, img_mutex, img_copy, img_copy_counter);
                        if (isfinite(L))
                        {
                            // Hacky: exclude NaNs in the rendering.
                            radiance += L;
                        }
                        else
                        {
                            printf("Error: L is not finite");
                        }
                    }
                    img(x, y) = radiance;
                    img_counter(x, y) += spp;
                }
            }
            reporter.update(1);
        }, Vector2i(num_tiles_x, num_tiles_y));
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            if (img_counter(x, y) > 0)
            {
                img(x, y) = (img(x, y) / img_counter(x, y)) + (img_copy(x, y)  * (1.0 / scene.options.samples_per_pixel));
            }
        }
    }
    Real avgLuminance = 0;
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            avgLuminance += luminance(img(x, y));
        }
    }
    printf("Avg Luminance: %lf\n", avgLuminance / w / h);
    reporter.done();
    return img;
}


Image3 render_pssmlt(const Scene& scene)
{
    int64_t w = scene.camera.width, h = scene.camera.height;
    const int max_depth = scene.options.max_depth;
    const int mutationsPerPixel = scene.options.mutations_per_pixel;
    const int64_t nBootstrap = scene.options.bootstrap_samples_per_pixel* w* h;
    const int64_t nBootstrapSamples = nBootstrap * (max_depth + 1);
    const Real sigma = 0.05;
    const Real largeStepProb = 0.33;
    const int streamCount = 3;
    const int nChains = 1000;

    Image3 img(w, h);
    Image1 img_counter(w, h);
    Image3 img_copy(w, h);
    std::mutex img_mutex;

    auto splatFilm = [&](Vector2 screenPos, Spectrum L, Real acceptRate)
    {
        std::lock_guard<std::mutex> lock(img_mutex);
        if (screenPos.x < 0 || screenPos.x > 1 || screenPos.y < 0 || screenPos.y > 1)
        {
            return;
        }
        int x = screenPos.x * w;
        int y = screenPos.y * h;
        Real l = luminance(L);
        if (std::abs(l) < 1e-6)
        {
            return;
        }
        img_copy(x, y) += L / l * acceptRate;
    };

    ProgressReporter progress_bootstrap(nBootstrap / 128);

    std::vector<Real> bootstrapWeights(nBootstrapSamples, 0);
    int64_t chunkSize = std::clamp(nBootstrap / 128, 1LL, 8192LL);
    parallel_for([&](int64_t index)
    {
        for (int depth = 0; depth <= max_depth; ++depth)
        {
            uint64_t rngIndex = index * (max_depth + 1) + depth;
            MLTSampler sampler(mutationsPerPixel, rngIndex, sigma,
                largeStepProb, streamCount);

            Vector2 screenPos;
            Real L = luminance(pssmlt_tracing(scene, &screenPos, sampler, depth));
            bootstrapWeights[rngIndex] = L;
        }
        if ((index + 1) % 128 == 0)
        {
            progress_bootstrap.update(1);
        }
    }, nBootstrap, chunkSize);
    progress_bootstrap.done();

    Distribution1D bootstrap(bootstrapWeights);
    Real b = bootstrap.funcInt * (max_depth + 1);

    printf("b = %lf, last = %lf\n", b, bootstrap.cdf.back());

    int64_t nTotalMutations = (int64_t)mutationsPerPixel * w * h;
    ProgressReporter progress_mutation(nTotalMutations);
    parallel_for(
    [&](int64_t index)
    {
        int64_t nChainMutations =
            std::min((index + 1) * nTotalMutations / nChains,
                nTotalMutations) - index * nTotalMutations / nChains;

        std::mt19937 mt(index);
        Real pdf;
        int bootstrapIndex = bootstrap.sample(&pdf, next_mt19937_real(mt));
        int depth = bootstrapIndex % (max_depth + 1);
        MLTSampler sampler(mutationsPerPixel, bootstrapIndex, sigma,
            largeStepProb, streamCount);
        Vector2 screenPosCurrent;
        Spectrum LCurrent =
            pssmlt_tracing(scene, &screenPosCurrent, sampler, depth);

        for (int64_t j = 0; j < nChainMutations; ++j)
        {
            sampler.start_iteration();

            Vector2 screenPosProposed;
            Spectrum LProposed =
                pssmlt_tracing(scene, &screenPosProposed, sampler, depth);
            Real accept = std::min((Real)1.0, luminance(LProposed) / luminance(LCurrent));
            if (accept > 0)
            {
                splatFilm(screenPosProposed, LProposed, accept);
            }
            splatFilm(screenPosCurrent, LCurrent, 1 - accept);

            if (next_mt19937_real(mt) < accept)
            {
                LCurrent = LProposed;
                screenPosCurrent = screenPosProposed;
                sampler.accept();
            }
            else
            {
                sampler.reject();
            }
        }
        progress_mutation.update(nChainMutations);
    }, nChains, 1);

   

    //constexpr int tile_size = 16;
    //int num_tiles_x = (w + tile_size - 1) / tile_size;
    //int num_tiles_y = (h + tile_size - 1) / tile_size;

    //ProgressReporter reporter(num_tiles_x * num_tiles_y);
    //parallel_for([&](const Vector2i& tile)
    //    {
    //        // Use a different rng stream for each thread.
    //        pcg32_state rng = init_pcg32(tile[1] * num_tiles_x + tile[0]);
    //        int x0 = tile[0] * tile_size;
    //        int x1 = min(x0 + tile_size, w);
    //        int y0 = tile[1] * tile_size;
    //        int y1 = min(y0 + tile_size, h);
    //        for (int y = y0; y < y1; y++)
    //        {
    //            for (int x = x0; x < x1; x++)
    //            {
    //                Spectrum radiance = make_zero_spectrum();
    //                int spp = scene.options.samples_per_pixel;
    //                for (int s = 0; s < spp; s++)
    //                {
    //                    Spectrum L = bidirectional_path_tracing(scene, x, y, rng, img_mutex, img_copy, img_copy_counter);
    //                    if (isfinite(L))
    //                    {
    //                        // Hacky: exclude NaNs in the rendering.
    //                        radiance += L;
    //                    }
    //                    else
    //                    {
    //                        printf("Error: L is not finite");
    //                    }
    //                }
    //                img(x, y) = radiance;
    //                img_counter(x, y) += spp;
    //            }
    //        }
    //        reporter.update(1);
    //    }, Vector2i(num_tiles_x, num_tiles_y));
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            img(x, y) = img_copy(x, y) * (b / mutationsPerPixel);
        }
    }
    progress_mutation.done();
    //reporter.done();
    return img;
}




Image3 vcm_render(const Scene& scene)
{
    int w = scene.camera.width, h = scene.camera.height;
    Image3 img(w, h);
    Image1 img_counter(w, h);
    Image3 img_copy(w, h);
    Image1 img_copy_counter(w, h);
    std::mutex img_mutex;

    constexpr int tile_size = 16;
    int num_tiles_x = (w + tile_size - 1) / tile_size;
    int num_tiles_y = (h + tile_size - 1) / tile_size;

    ProgressReporter reporter(num_tiles_x * num_tiles_y);
    parallel_for([&](const Vector2i& tile)
        {
            // Use a different rng stream for each thread.
            pcg32_state rng = init_pcg32(tile[1] * num_tiles_x + tile[0]);
    int x0 = tile[0] * tile_size;
    int x1 = min(x0 + tile_size, w);
    int y0 = tile[1] * tile_size;
    int y1 = min(y0 + tile_size, h);
    for (int y = y0; y < y1; y++)
    {
        for (int x = x0; x < x1; x++)
        {
            Spectrum radiance = make_zero_spectrum();
            int spp = scene.options.samples_per_pixel;
            for (int s = 0; s < spp; s++)
            {
                Spectrum L = bidirectional_path_tracing(scene, x, y, rng, img_mutex, img_copy, img_copy_counter);
                if (isfinite(L))
                {
                    // Hacky: exclude NaNs in the rendering.
                    radiance += L;
                }
                else
                {
                    printf("Error: L is not finite");
                }
            }
            img(x, y) = radiance;
            img_counter(x, y) += spp;
        }
    }
    reporter.update(1);
        }, Vector2i(num_tiles_x, num_tiles_y));
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            if (img_counter(x, y) > 0)
            {
                img(x, y) = (img(x, y) / img_counter(x, y)) + (img_copy(x, y) * (1.0 / scene.options.samples_per_pixel));
            }
        }
    }
    reporter.done();
    return img;
}



Image3 gdpt_render(const Scene& scene)
{
    return do_gdpt(scene);
}


Image3 restir_pt_render(const Scene& scene)
{
    return do_restir_pt(scene);
}

Image3 render(const Scene &scene) {
    if (scene.options.integrator == Integrator::Depth ||
            scene.options.integrator == Integrator::ShadingNormal ||
            scene.options.integrator == Integrator::MeanCurvature ||
            scene.options.integrator == Integrator::RayDifferential ||
            scene.options.integrator == Integrator::MipmapLevel) {
        return aux_render(scene);
    } 
    else if (scene.options.integrator == Integrator::Path) 
    {
        return path_render(scene);
    } 
    else if (scene.options.integrator == Integrator::PathWithFeatures)
    {
        path_render_with_feature(scene);
        return Image3(1, 1);
    }
    else if (scene.options.integrator == Integrator::VolPath) 
    {
        return vol_path_render(scene);
    }
    else if (scene.options.integrator == Integrator::BDPath)
    {
        return bi_directional_render(scene);
    }
    else if (scene.options.integrator == Integrator::SPPM)
    {
        return photon_mapping_render(scene);
    }
    else if (scene.options.integrator == Integrator::PSSMLT)
    {
        return render_pssmlt(scene);
    }
    else if (scene.options.integrator == Integrator::VCM)
    {
        return vcm_render(scene);
    }
    else if (scene.options.integrator == Integrator::GDPT)
    {
        return gdpt_render(scene);
    }
    else if (scene.options.integrator == Integrator::RESTIRPT)
    {
        return restir_pt_render(scene);
    }
    else
    {
        assert(false);
        return Image3();
    }
}
