#pragma once
#include "thread_safe.h"
#include "progress_reporter.h"
#include "parallel.h"

struct PMPixel
{
    Real radius = 0;
    Spectrum Ld{};

    struct VisiblePoint
    {
        VisiblePoint() {}
        VisiblePoint(const Vector3& p, const Vector3f& win, const PathVertex& vertex,
            const Spectrum& beta)
            : p(p), win(win), vertex(vertex), beta(beta)
        {
        }

        Vector3 p{};
        Vector3f win{};
        PathVertex vertex{};
        Spectrum beta{};
    } vp{};

    Spectrum phi{};
    int M = 0;
    std::mutex phi_lock{};

    Real N = 0;
    Spectrum tau{};
};

static bool ToGrid(const Vector3& p, const Bound3& bounds,
    const Vector3i& gridRes, Vector3i* pi)
{
    bool inBounds = true;
    Vector3f pg = (p - bounds.minPos) / (bounds.maxPos - bounds.minPos);
    for (int i = 0; i < 3; ++i)
    {
        (*pi)[i] = (int)(gridRes[i] * pg[i]);
        inBounds &= ((*pi)[i] >= 0 && (*pi)[i] < gridRes[i]);
        (*pi)[i] = std::clamp((*pi)[i], 0, gridRes[i] - 1);
    }
    return inBounds;
}

Spectrum compute_direct_lighting(const Scene& scene, pcg32_state& rng,
    const PathVertex& vertex,
    const Vector3& win)
{
    // First, we sample a point on the light source.
            // We do this by first picking a light source, then pick a point on it.
    Vector2 light_uv{ next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng) };
    Real light_w = next_pcg32_real<Real>(rng);
    Real shape_w = next_pcg32_real<Real>(rng);
    int light_id = sample_light(scene, light_w);
    const Light& light = scene.lights[light_id];
    PointAndNormal point_on_light =
        sample_point_on_light(light, vertex.position, light_uv, shape_w, scene);

    Vector3 dir_light = normalize(point_on_light.position - vertex.position);
    int medium = vertex.medium_id;
    // If the point on light is occluded, G is 0. So we need to test for occlusion.
    // To avoid self intersection, we need to set the tnear of the ray
    // to a small "epsilon". We set the epsilon to be a small constant times the
    // scale of the scene, which we can obtain through the get_shadow_epsilon() function.
    Vector3 N = dot(dir_light, vertex.geometric_normal) > 0 ? vertex.geometric_normal : -vertex.geometric_normal;
    Vector3 rayOrigin = vertex.position + N * get_intersection_epsilon(scene);
    Ray shadow_ray{
        rayOrigin,
        point_on_light.position - rayOrigin,
        0,
        1 - get_shadow_epsilon(scene)
    };

    const Material& mat = scene.materials[vertex.material_id];

    Spectrum L = make_zero_spectrum();
    if (!occluded(scene, shadow_ray))
    {
        // geometry term is cosine at v_{i+1} divided by distance squared
        // this can be derived by the infinitesimal area of a surface projected on
        // a unit sphere -- it's the Jacobian between the area measure and the solid angle
        // measure.
        Real G = max(dot(-dir_light, point_on_light.normal), Real(0)) /
            distance_squared(point_on_light.position, vertex.position);

        // Before we proceed, we first compute the probability density p1(v1)
        // The probability density for light sampling to sample our point is
        // just the probability of sampling a light times the probability of sampling a point

        Real pdf_pos_t = pdf_point_on_light(light, point_on_light, vertex.position, scene);

        Real pdf_nee = pdf_pos_t * light_pmf(scene, light_id);
        Real pdf_bsdf = pdf_sample_bsdf(mat, win,
            dir_light, vertex, scene.texture_pool, TransportDirection::TO_LIGHT) * G;
         
        Real weight = (pdf_nee * pdf_nee) / (pdf_nee * pdf_nee + pdf_bsdf * pdf_bsdf);
        L += eval(mat, win, dir_light, vertex, scene.texture_pool) 
            * emission(light, -dir_light, Real(0), point_on_light, scene) * G / pdf_nee * weight;
    }

    Vector2 bsdf_rnd_param_uv{ next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng) };
    Real bsdf_rnd_param_w = next_pcg32_real<Real>(rng);
    std::optional<BSDFSampleRecord> bsdf_sample_ =
        sample_bsdf(mat,
            win,
            vertex,
            scene.texture_pool,
            bsdf_rnd_param_uv,
            bsdf_rnd_param_w);
    if (bsdf_sample_)
    {
       Real pdf_bsdf = pdf_sample_bsdf(mat, win, bsdf_sample_->dir_out, vertex, scene.texture_pool, TransportDirection::TO_LIGHT);
       Ray tmp_ray = spawn_ray(scene, vertex, bsdf_sample_->dir_out);

       std::optional<PathVertex> hit = intersect(scene, tmp_ray);
       
       if (hit && is_light(scene.shapes[hit->shape_id]))
       {
           int light_id1 = get_area_light_id(scene.shapes[hit->shape_id]);
           if (light_id1 == light_id)
           {
               PointAndNormal hitLight = PointAndNormal{ hit->position, hit->geometric_normal };
               Real pdf_pos_t = pdf_point_on_light(light, hitLight, vertex.position, scene);
               Real pdf_nee = pdf_pos_t * light_pmf(scene, light_id);
               Real G = max(dot(-dir_light, point_on_light.normal), Real(0)) /
                   distance_squared(point_on_light.position, vertex.position);
               pdf_bsdf *= G;

               Real weight = (pdf_bsdf * pdf_bsdf) / (pdf_nee * pdf_nee + pdf_bsdf * pdf_bsdf);

               L += eval(mat, win, bsdf_sample_->dir_out, vertex, scene.texture_pool) *
                   emission(light, -bsdf_sample_->dir_out, Real(0), hitLight, scene) * G / pdf_bsdf * weight;
           }
       }
       
    }
    return L;
}

void gen_camera_subpath(
    PMPixel& pixel,
    const Scene& scene,
    const Vector2& screenPos,
    pcg32_state& rng,
    int maxDepth)
{
    Ray ray = sample_primary(scene.camera, screenPos);
    RayDifferential ray_diff = RayDifferential{ Real(0), Real(0) };

    // PathVertex cameraVertex = CreateCamera(scene.camera, make_const_spectrum(Real(1)));

    int medium_id = scene.camera.medium_id;
    Spectrum beta = make_const_spectrum(Real(1));
    bool specularBounce = true;
    for (int depth = 0; depth < scene.options.max_depth; depth++)
    {
        std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);

        if (!vertex_)
        {
            break;
        }

        PathVertex& vertex = *vertex_;
        if (vertex.material_id == -1)
        {
            Vector3 N = dot(ray.dir, vertex.geometric_normal) > 0 ? vertex.geometric_normal : -vertex.geometric_normal;
            Vector3 rayOrigin = vertex.position + N * get_intersection_epsilon(scene);
            ray.org = rayOrigin;

            // check which side of the surface we are hitting
            medium_id = updateMedium(ray.dir, vertex, medium_id);

            // Skip id surface because it is meaningless to stay here 
            depth--;
            continue;
        }

        if (specularBounce)
        {
            if (is_light(scene.shapes[vertex.shape_id]))
            {
                pixel.Ld += beta *
                    emission(vertex, -ray.dir, scene);
            }
        }

        specularBounce = false;
        pixel.Ld += compute_direct_lighting(scene, rng, vertex, -ray.dir);

        const Material& mat = scene.materials[vertex.material_id];
        bool diffuse = false;
        if (std::holds_alternative<Lambertian>(mat))
        {
            diffuse = true;
        }
        if (diffuse || depth == scene.options.max_depth - 1)
        {
            pixel.vp = { vertex.position, -ray.dir, vertex, beta };
            break;
        }


        Vector2 bsdf_rnd_param_uv{ next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng) };
        Real bsdf_rnd_param_w = next_pcg32_real<Real>(rng);
        std::optional<BSDFSampleRecord> bsdf_sample_ =
            sample_bsdf(mat,
                -ray.dir,
                vertex,
                scene.texture_pool,
                bsdf_rnd_param_uv,
                bsdf_rnd_param_w);
        if (bsdf_sample_)
        {
            Real pdf_bsdf = pdf_sample_bsdf(mat, -ray.dir, bsdf_sample_->dir_out, vertex, scene.texture_pool);
            beta *= eval(mat, -ray.dir, bsdf_sample_->dir_out, vertex, scene.texture_pool) / pdf_bsdf;

            ray.dir = bsdf_sample_->dir_out;
            Vector3 N = dot(ray.dir, vertex.geometric_normal) > 0 ? vertex.geometric_normal : -vertex.geometric_normal;
            Vector3 rayOrigin = vertex.position + N * get_intersection_epsilon(scene);
            ray.org = rayOrigin;
        }

        //Real rr_prob = 1;
        //if (luminance(beta) < 0.25)
        //{
        //    rr_prob = min(luminance(beta), Real(1.0));
        //    if (next_pcg32_real<Real>(rng) > rr_prob)
        //    {
        //        // Terminate the path
        //        break;
        //    }
        //}
        //beta /= rr_prob;
    }
}

void trace_photons(const Scene& scene, const Bound3& bound, 
    const Vector3i& gridRes, std::map<Vector3i, std::vector<PMPixel*>>& grid)
{
    parallel_for([scene, bound, gridRes, &grid](int index)
        {
            pcg32_state rng = init_pcg32(index + 
            (scene.options.photonsPerIteration * scene.options.iterations)
            + 3254564537ull);

    Vector2 light_uv{ next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng) };
    Vector2 dir_uv{ next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng) };
    Real light_w = next_pcg32_real<Real>(rng);
    Real shape_w = next_pcg32_real<Real>(rng);
    int light_id = sample_light(scene, light_w);
    const Light& light = scene.lights[light_id];
    Real pdf_light = light_pmf(scene, light_id);
    std::optional<LightLeSampleRecord> record = sample_point_direction_on_light(light,
        Vector3{ 0,0,0 }, light_uv, shape_w, dir_uv, scene);

    if (!record)
    {
        return;
    }
    Vector3 N = dot(record->dir_out, record->normal) > 0 ? record->normal : -record->normal;
    Vector3 rayOrigin = record->pos + N * get_intersection_epsilon(scene);
    Ray ray{ rayOrigin, record->dir_out, 0, infinity<Real>() };
    PathVertex v = CreateLight(light_id, record->pos, record->L, pdf_light * record->pdf_pos);
    v.geometric_normal = record->normal;
    v.shading_frame = Frame(v.geometric_normal);

    Spectrum beta = (record->L * std::max(dot(record->normal, ray.dir), 0.0)) / (pdf_light * record->pdf_pos * record->pdf_dir);

    if (luminance(beta) == 0)
    {
        return;
    }
    int medium_id = scene.camera.medium_id;
    for (int depth = 0; depth < scene.options.max_depth; depth++)
    {
        std::optional<PathVertex> vertex_ = intersect(scene, ray);

        if (!vertex_ || isnan(vertex_->geometric_normal))
        {
            break;
        }

        PathVertex& vertex = *vertex_;
        // We do not account for direct lighting in photon mapping stage
        if (depth > 0)
        {
            Vector3i photonGridIndex;
            if (ToGrid(vertex.position, bound, gridRes, &photonGridIndex))
            {
                if (grid.find(photonGridIndex) != grid.end())
                {
                    // Add photon contribution to visible points in grid[h]
                    auto& vec = grid[photonGridIndex];
                    for (auto pixel : vec)
                    {
                        Real radius = pixel->radius;
                        if (distance_squared(pixel->vp.p, vertex.position) > radius * radius)
                            continue;

                        Vector3 wout = -ray.dir;
                        const Material& material = scene.materials[pixel->vp.vertex.material_id];
                        Spectrum Phi = beta * eval(material, pixel->vp.win, wout, pixel->vp.vertex,
                            scene.texture_pool, TransportDirection::TO_LIGHT);
                        {
                            std::lock_guard<std::mutex> lck_guard(pixel->phi_lock);
                            pixel->M++;
                            pixel->phi += Phi;
                        }
                    }
                }
            }
        }


        if (vertex.material_id == -1)
        {
            Vector3 N = dot(ray.dir, vertex.geometric_normal) > 0 ? vertex.geometric_normal : -vertex.geometric_normal;
            Vector3 rayOrigin = vertex.position + N * get_intersection_epsilon(scene);
            ray.org = rayOrigin;

            // check which side of the surface we are hitting
            medium_id = updateMedium(ray.dir, vertex, medium_id);
            depth--;
            continue;
        }

        const Material& mat = scene.materials[vertex.material_id];

        Vector2 bsdf_rnd_param_uv{ next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng) };
        Real bsdf_rnd_param_w = next_pcg32_real<Real>(rng);
        std::optional<BSDFSampleRecord> bsdf_sample_ =
            sample_bsdf(mat,
                -ray.dir,
                vertex,
                scene.texture_pool,
                bsdf_rnd_param_uv,
                bsdf_rnd_param_w, TransportDirection::TO_VIEW);
        if (bsdf_sample_)
        {
            Real pdf_bsdf = pdf_sample_bsdf(mat, -ray.dir, bsdf_sample_->dir_out, vertex, scene.texture_pool, TransportDirection::TO_VIEW);
            beta *= eval(mat, -ray.dir, bsdf_sample_->dir_out, vertex, scene.texture_pool, TransportDirection::TO_VIEW) / pdf_bsdf;

            ray.dir = bsdf_sample_->dir_out;
            Vector3 N = dot(ray.dir, vertex.geometric_normal) > 0 ? vertex.geometric_normal : -vertex.geometric_normal;
            Vector3 rayOrigin = vertex.position + N * get_intersection_epsilon(scene);
            ray.org = rayOrigin;
        }
    }
        }, scene.options.photonsPerIteration, 1024);
}

Vector3i compute_grid_bounds(const PMPixel* pixels, int w, int h, Bound3* bound)
{
    Bound3 gridBounds{};
    Real maxRadius = 0.;
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            auto& pixel = pixels[i * w + j];
            if (luminance(pixel.vp.beta) == 0)
            {
                continue;
            }
            gridBounds = gridBounds.Union(Bound3{ pixel.vp.p - pixel.radius, pixel.vp.p + pixel.radius });
            maxRadius = std::max(pixel.radius, maxRadius);
        }
    }
    Vector3 diag = gridBounds.maxPos - gridBounds.minPos;
    Real maxDiag = max(diag);
    int baseGridRes = (int)(maxDiag / maxRadius);
    Vector3i gridRes{ max((int)(baseGridRes * diag.x / maxDiag), 1),
        max((int)(baseGridRes * diag.y / maxDiag), 1),
        max((int)(baseGridRes * diag.z / maxDiag), 1) };

    *bound = gridBounds;
    return gridRes;
}


void add_pixels_to_grid(PMPixel* pixels, int num_tiles_y, int num_tiles_x, int tile_size, int w, int h,
    const Bound3& bound, const Vector3i& gridRes, std::map<Vector3i, std::vector<PMPixel*>>& grid)
{
    static std::mutex lock_grid;
    {
        std::lock_guard<std::mutex> lck(lock_grid);
        grid.clear();
    }
    parallel_for([&](int index)
        {
            PMPixel& pixel = pixels[index];
    if (luminance(pixel.vp.beta) == 0)
    {
        return;
    }

    Real radius = pixel.radius;
    Vector3i pMin, pMax;
    ToGrid(pixel.vp.p - Vector3(radius, radius, radius),
        bound, gridRes, &pMin);
    ToGrid(pixel.vp.p + Vector3(radius, radius, radius),
        bound, gridRes, &pMax);

    {
        std::lock_guard<std::mutex> lck(lock_grid);
        for (int z = pMin.z; z <= pMax.z; ++z)
            for (int y = pMin.y; y <= pMax.y; ++y)
                for (int x = pMin.x; x <= pMax.x; ++x)
                {
                    if (grid.find(Vector3i{ x, y, z }) == grid.end())
                    {
                        grid[Vector3i{ x, y, z }] = std::vector<PMPixel*>{ &pixel };
                    }
                    else
                    {
                        grid[Vector3i{ x, y, z }].push_back(&pixel);
                    }
                }
    }
        }, (uint64_t)w * h, 1024);
}

void update_photons(const Scene& scene, PMPixel* pixels, int w, int h, int tile_size)
{
    parallel_for([&](int index)
        {
            auto& pixel = pixels[index];

    if (pixel.M > 0)
    {
        Real gamma = Real(2) / 3;
        Real Nnew = pixel.N + gamma * pixel.M;
        Real Rnew = pixel.radius * std::sqrt(Nnew / (pixel.N + pixel.M));
        Spectrum Phi = pixel.phi;

        pixel.tau = (pixel.tau + pixel.vp.beta * Phi) *
            (Rnew * Rnew) / (pixel.radius * pixel.radius);
        pixel.N = Nnew;
        pixel.radius = Rnew;
        pixel.M = 0;
        pixel.phi = make_zero_spectrum();
    }
    pixel.vp.beta = make_zero_spectrum();
    pixel.vp.vertex = PathVertex{};
        }, w * h, 1024);
}

Image3 photon_mapping_render(const Scene& scene)
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


    PMPixel* pixels = new PMPixel[w * h]{};
    // Initialize pixels
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            pixels[y * w + x].radius = 1;
        }
    }

    ProgressReporter reporter(5 * scene.options.iterations);
    static std::map<Vector3i, std::vector<PMPixel*>> grid;
    for (int i = 0; i < scene.options.iterations; i++)
    {
        // Generate SPPM visible points
        parallel_for([&](const Vector2i& tile)
            {
                int index = tile[1] * num_tiles_x + tile[0];
        // Use a different rng stream for each thread.
        pcg32_state rng = init_pcg32((uint64_t)tile[1] * num_tiles_x + tile[0] + (uint64_t)w * h * i);
        int x0 = tile[0] * tile_size;
        int x1 = min(x0 + tile_size, w);
        int y0 = tile[1] * tile_size;
        int y1 = min(y0 + tile_size, h);
        for (int y = y0; y < y1; y++)
        {
            for (int x = x0; x < x1; x++)
            {
                PMPixel& pixel = pixels[y * w + x];
 
                Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                    (y + next_pcg32_real<Real>(rng)) / h);
                gen_camera_subpath(pixel, scene, screen_pos, rng, scene.options.max_depth);
            }
        }
            }, Vector2i(num_tiles_x, num_tiles_y));
        reporter.update(1);
        
        // Compute grid bounds for SPPM visible points
        Bound3 gridBound;
        Vector3i gridRes = compute_grid_bounds(pixels, w, h, &gridBound);
        reporter.update(1);

        // Add visible points to SPPM grid
        add_pixels_to_grid(pixels, num_tiles_y, num_tiles_x, tile_size, w, h, gridBound, gridRes, grid);
        reporter.update(1);

        // Trace photons and accumulate contributions
        trace_photons(scene, gridBound, gridRes, grid);
        reporter.update(1);

        // Update pixel values from this pass¡¯s photons
        update_photons(scene, pixels, w, h, tile_size);
        reporter.update(1);
    }

    reporter.done();

    for (int i = 0; i < num_tiles_y; i++)
    {
        for (int j = 0; j < num_tiles_x; j++)
        {
            int x0 = j * tile_size;
            int x1 = min(x0 + tile_size, w);
            int y0 = i * tile_size;
            int y1 = min(y0 + tile_size, h);
            int index = i * num_tiles_x + j;

            uint64_t Np = (uint64_t)(scene.options.iterations + 1) * (uint64_t)scene.options.photonsPerIteration;

            for (int y = y0; y < y1; y++)
            {
                for (int x = x0; x < x1; x++)
                {
                    int innerIndex = (y - y0) * (x1 - x0) + (x - x0);
                    PMPixel& pixel = pixels[y * w + x];
                    Spectrum L = pixel.Ld / (Real)(scene.options.iterations + 1);
                    L += pixel.tau / (Np * c_PI * pixel.radius * pixel.radius);

                    img(x, y) = L;
                }
            }
        }
    }

    delete[] pixels;

    return img;
}
