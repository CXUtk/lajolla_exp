#pragma once

// The simplest volumetric renderer: 
// single absorption only homogeneous volume
// only handle directly visible light sources
Spectrum vol_path_tracing_1(const Scene& scene,
    int x, int y, /* pixel coordinates */
    pcg32_state& rng)
{
    // Generate camera ray
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
        (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    RayDifferential ray_diff = RayDifferential{ Real(0), Real(0) };

    std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);

    if (!vertex_)
    {
        // Hit background. Set it to zero
        return make_zero_spectrum();
    }

    PathVertex vertex = *vertex_;

    Spectrum radiance = make_zero_spectrum();

    // printf("%d %d\n", vertex.interior_medium_id, vertex.exterior_medium_id);

    Spectrum transmittance = make_const_spectrum(1.0);
    // If is inside a medium
    if (vertex.exterior_medium_id != -1)
    {
        const Medium& medium = scene.media[vertex.exterior_medium_id];
        Real distance = length(vertex.position - ray.org);
        transmittance = exp(-distance * get_sigma_a(medium, vertex.position));
    }

    // We hit a light immediately. 
    // This path has only two vertices and has contribution
    // C = W(v0, v1) * G(v0, v1) * L(v0, v1)
    if (is_light(scene.shapes[vertex.shape_id]))
    {
        radiance += transmittance *
            emission(vertex, -ray.dir, scene);
    }

    return radiance;
}


Spectrum sample_direct_lights(const Scene& scene, pcg32_state& rng,
    const Vector3& ref_point, const Vector3& ref_normal,
    Real* out_pdf, PointAndNormal* point_on_light)
{
    // First, we sample a point on the light source.
            // We do this by first picking a light source, then pick a point on it.
    Vector2 light_uv{ next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng) };
    Real light_w = next_pcg32_real<Real>(rng);
    Real shape_w = next_pcg32_real<Real>(rng);
    int light_id = sample_light(scene, light_w);
    const Light& light = scene.lights[light_id];
    *point_on_light =
        sample_point_on_light(light, ref_point, light_uv, shape_w, scene);

    Vector3 dir_light = normalize(point_on_light->position - ref_point);
    // If the point on light is occluded, G is 0. So we need to test for occlusion.
    // To avoid self intersection, we need to set the tnear of the ray
    // to a small "epsilon". We set the epsilon to be a small constant times the
    // scale of the scene, which we can obtain through the get_shadow_epsilon() function.
    Vector3 N = dot(dir_light, ref_normal) > 0 ? ref_normal : -ref_normal;
    Vector3 rayOrigin = ref_point + N * get_shadow_epsilon(scene);
    Ray shadow_ray{ rayOrigin, dir_light,
                   0,
                   (1 - get_shadow_epsilon(scene)) *
                       distance(point_on_light->position, rayOrigin) };
    *out_pdf = Real(0);

    if (!occluded(scene, shadow_ray))
    {
        // geometry term is cosine at v_{i+1} divided by distance squared
        // this can be derived by the infinitesimal area of a surface projected on
        // a unit sphere -- it's the Jacobian between the area measure and the solid angle
        // measure.
        Real G = max(-dot(dir_light, point_on_light->normal), Real(0)) /
            distance_squared(point_on_light->position, ref_point);

        // Before we proceed, we first compute the probability density p1(v1)
        // The probability density for light sampling to sample our point is
        // just the probability of sampling a light times the probability of sampling a point
        Real p1 = light_pmf(scene, light_id) *
            pdf_point_on_light(light, *point_on_light, ref_point, scene);


        Spectrum L = emission(light, -dir_light, Real(0), *point_on_light, scene);

        *out_pdf = p1;
        return L * G;
    }
    return make_zero_spectrum();
}

// The second simplest volumetric renderer: 
// single monochromatic homogeneous volume with single scattering,
// no need to handle surface lighting, only directly visible light source
Spectrum vol_path_tracing_2(const Scene& scene,
    int x, int y, /* pixel coordinates */
    pcg32_state& rng)
{
    // Generate camera ray
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
        (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    RayDifferential ray_diff = RayDifferential{ Real(0), Real(0) };


    std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
    Spectrum transmittance = make_const_spectrum(1.0);
    Spectrum radiance = make_zero_spectrum();


    const Medium& medium = scene.media[scene.camera.medium_id];

    Spectrum extinctFactor = get_sigma_a(medium, ray.org) + get_sigma_s(medium, ray.org);
    Real sigma_t = extinctFactor.x;
    Real sigma_s = get_sigma_s(medium, ray.org).x;

    Real u = next_pcg32_real<Real>(rng);
    Real t = -std::log(1 - u) / sigma_t;

    Real trans_pdf = std::exp(-sigma_t * t) * sigma_t;
    transmittance = exp(-extinctFactor * t);

    Vector3 position = ray.org + t * ray.dir;

    if (!vertex_ || t < length(vertex_->position - ray.org))
    {
        Real pdf;
        PointAndNormal light_pos;
        Spectrum L = sample_direct_lights(scene, rng, position, Vector3{ 0, 0, 0 }, &pdf, &light_pos);
        if (pdf == Real(0))
        {
            return make_zero_spectrum();
        }
        // transmission from light to scattering particle
        Spectrum Tr = exp(-extinctFactor * distance(light_pos.position, position));

        PhaseFunction phaseFunction = get_phase_function(medium);
        Spectrum rho = eval(phaseFunction, -ray.dir, normalize(light_pos.position - position));
        return transmittance / trans_pdf * sigma_s * rho * Tr * L / pdf;
    }
    else
    {
        Real hit_t = length(vertex_->position - ray.org);
        PathVertex vertex = *vertex_;
        Real trans_pdf = std::exp(-sigma_t * hit_t);
        transmittance = exp(-extinctFactor * hit_t);

        // We hit a light immediately. 
        // This path has only two vertices and has contribution
        // C = W(v0, v1) * G(v0, v1) * L(v0, v1)
        if (is_light(scene.shapes[vertex.shape_id]))
        {
            radiance += transmittance / trans_pdf *
                emission(vertex, -ray.dir, scene);
        }
        return radiance;
    }
}

// The third volumetric renderer (not so simple anymore): 
// multiple monochromatic homogeneous volumes with multiple scattering
// no need to handle surface lighting, only directly visible light source
Spectrum vol_path_tracing_3(const Scene& scene,
    int x, int y, /* pixel coordinates */
    pcg32_state& rng)
{

    // Generate camera ray
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
        (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    RayDifferential ray_diff = RayDifferential{ Real(0), Real(0) };
    int currentMedium_id = scene.camera.medium_id;

    Spectrum current_path_throughput = make_const_spectrum(Real(1.0));
    Spectrum radiance = make_zero_spectrum();
    int bounces = 0;
    while (true)
    {
        bool scatter = false;
        std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
        Spectrum transmittance = make_const_spectrum(1.0);
        Real trans_pdf = 1.0;

        if (currentMedium_id != -1)
        {
            const Medium& medium = scene.media[currentMedium_id];

            Spectrum extinctFactor = get_sigma_a(medium, ray.org) + get_sigma_s(medium, ray.org);
            Real sigma_t = extinctFactor.x;

            Real u = next_pcg32_real<Real>(rng);
            Real t = -std::log(1 - u) / sigma_t;


            if (!vertex_ || t < length(vertex_->position - ray.org))
            {
                trans_pdf = std::exp(-sigma_t * t) * sigma_t;
                transmittance = exp(-extinctFactor * t);
                ray.org = ray.org + t * ray.dir;
                scatter = true;
            }
        }
        current_path_throughput *= transmittance / trans_pdf;

        // If hit a surface
        if (!scatter)
        {
            PathVertex vertex = *vertex_;
            // We hit a light immediately. 
            // This path has only two vertices and has contribution
            // C = W(v0, v1) * G(v0, v1) * L(v0, v1)
            if (is_light(scene.shapes[vertex.shape_id]))
            {
                radiance += current_path_throughput *
                    emission(vertex, -ray.dir, scene);
            }

        }

        if (bounces == scene.options.max_depth - 1 && scene.options.max_depth != -1)
        {
            break;
        }

        if (!scatter && vertex_)
        {
            PathVertex vertex = *vertex_;
            // index-matching interface, skip through it
            if (vertex.material_id == -1)
            {
                // check which side of the surface we are hitting
                int nextMediumId = vertex.interior_medium_id;
                if (dot(ray.dir, vertex.geometric_normal) > 0)
                {
                    nextMediumId = vertex.exterior_medium_id;
                }
                currentMedium_id = nextMediumId;

                Vector3 N = dot(ray.dir, vertex.geometric_normal) > 0 ? vertex.geometric_normal : -vertex.geometric_normal;
                Vector3 rayOrigin = vertex.position + N * get_intersection_epsilon(scene);
                ray.org = rayOrigin;

                bounces++;
                continue;
            }
            else
            {
                // Deal with surface
            }
        }

        if (scatter)
        {
            const Medium& medium = scene.media[currentMedium_id];
            PhaseFunction phaseFunction = get_phase_function(medium);

            Vector2 phase_rnd_param = Vector2(next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng));
            std::optional<Vector3> next_dir = sample_phase_function(phaseFunction, -ray.dir, phase_rnd_param);

            if (next_dir)
            {
                Real sigma_s = get_sigma_s(medium, ray.org).x;
                Spectrum rho = eval(phaseFunction, -ray.dir, *next_dir);
                current_path_throughput *= rho / pdf_sample_phase(phaseFunction, -ray.dir, *next_dir) * sigma_s;
                ray.dir = *next_dir;
            }
        }
        else
        {
            // Empty hit
            break;
        }

        Real rr_prob = 1.0;
        if (bounces >= scene.options.rr_depth)
        {
            rr_prob = std::min(luminance(current_path_throughput), 0.95);
            if (next_pcg32_real<Real>(rng) > rr_prob)
            {
                break;
            }
            else
            {
                current_path_throughput /= rr_prob;
            }
        }
        bounces++;
    }
    return radiance;
}


struct PathTracingContext
{
    const Scene& scene;
    pcg32_state& rng;
    Ray ray;
    RayDifferential rayDiff;
    int medium_id;
    int bounces;
};


int updateMedium(const PathTracingContext& context, const PathVertex& vertex, int medium_id)
{

    if (vertex.interior_medium_id != vertex.exterior_medium_id)
    {
        if (dot(context.ray.dir, vertex.geometric_normal) > 0)
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

bool sample_vol_scattering(const PathTracingContext& context,
    Real tMax,
    Real* t_sample,
    Spectrum* transmittance,
    Real* trans_pdf)
{
    const Medium& medium = context.scene.media[context.medium_id];

    Spectrum extinctFactor = get_sigma_a(medium, context.ray.org) + get_sigma_s(medium, context.ray.org);
    Real sigma_t = extinctFactor.x;

    Real u = next_pcg32_real<Real>(context.rng);
    Real t = -std::log(1 - u) / sigma_t;

    if (t < tMax)
    {
        *t_sample = t;
        *trans_pdf = std::exp(-sigma_t * t) * sigma_t;
        *transmittance = exp(-extinctFactor * t);
        return true;
    }
    else
    {
        *t_sample = tMax;
        *trans_pdf = std::exp(-sigma_t * tMax);
        *transmittance = exp(-extinctFactor * tMax);
        return false;
    }
}


Spectrum path_sample_phase_function(const PathTracingContext& context, Vector3* wout, Real* pdf)
{
    Spectrum L = make_const_spectrum(1.0);
    const Medium& medium = context.scene.media[context.medium_id];
    Real sigma_s = get_sigma_s(medium, context.ray.org).x;
    PhaseFunction phaseFunction = get_phase_function(medium);

    Vector2 phase_rnd_param = Vector2(next_pcg32_real<Real>(context.rng), next_pcg32_real<Real>(context.rng));

    Vector3 win = -context.ray.dir;
    std::optional<Vector3> next_dir = sample_phase_function(phaseFunction, win, phase_rnd_param);
    if (next_dir)
    {
        *wout = *next_dir;
        *pdf = pdf_sample_phase(phaseFunction, win, *next_dir);

        Spectrum rho = eval(phaseFunction, win, *next_dir);
        return rho * sigma_s;
    }
    *pdf = 0;
    return L * sigma_s;
}

Spectrum next_event_estimation(const PathTracingContext& context, const Vector3& ref_point, const Vector3& ref_normal,
    Real* pdf,
    Real* p_trans,
    PointAndNormal* pointOnLight)
{
    // First, we sample a point on the light source.
        // We do this by first picking a light source, then pick a point on it.
    Vector2 light_uv{ next_pcg32_real<Real>(context.rng), next_pcg32_real<Real>(context.rng) };
    Real light_w = next_pcg32_real<Real>(context.rng);
    Real shape_w = next_pcg32_real<Real>(context.rng);
    int light_id = sample_light(context.scene, light_w);
    const Light& light = context.scene.lights[light_id];
    PointAndNormal point_on_light =
        sample_point_on_light(light, ref_point, light_uv, shape_w, context.scene);
    *pointOnLight = point_on_light;

    Vector3 dir_light = normalize(point_on_light.position - ref_point);
    // If the point on light is occluded, G is 0. So we need to test for occlusion.
    // To avoid self intersection, we need to set the tnear of the ray
    // to a small "epsilon". We set the epsilon to be a small constant times the
    // scale of the scene, which we can obtain through the get_shadow_epsilon() function.
    Vector3 N = dot(dir_light, ref_normal) > 0 ? ref_normal : -ref_normal;
    Vector3 rayOrigin = ref_point + N * get_shadow_epsilon(context.scene);

    Spectrum T_light = make_const_spectrum(1.0);
    int shadow_medium = context.medium_id;
    int shadow_bounces = 0;
    Real p_trans_dir = 1.0;
    Vector3 P = ref_point;
    *pdf = 0;
    while (true)
    {
        Ray shadow_ray{ P, normalize(point_on_light.position - P),
               get_shadow_epsilon(context.scene),
               (1 - get_shadow_epsilon(context.scene)) *
                   distance(point_on_light.position, P) };
        std::optional<PathVertex> vertex_ = intersect(context.scene, shadow_ray, context.rayDiff);

        // compute travel distance
        Real next_t = distance(point_on_light.position, P);
        if (vertex_)
        {
            next_t = length(vertex_->position - P);
        }

        // compute transmission and pdf for travel this distance
        if (shadow_medium != -1)
        {
            const Medium& medium = context.scene.media[shadow_medium];
            Spectrum extinctFactor = get_sigma_a(medium, P) + get_sigma_s(medium, P);
            Real sigma_t = extinctFactor.x;
            T_light *= exp(-extinctFactor * next_t);
            p_trans_dir *= exp(-sigma_t * next_t);
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
            if (context.scene.options.max_depth != -1 && shadow_bounces + context.bounces + 1 >= context.scene.options.max_depth)
            {
                return make_zero_spectrum();
            }
            shadow_bounces++;
            shadow_medium = updateMedium(context, vertex, shadow_medium);

            Vector3 N = dot(context.ray.dir, vertex.geometric_normal) > 0 ? vertex.geometric_normal : -vertex.geometric_normal;
            Vector3 rayOrigin = vertex.position + N * get_intersection_epsilon(context.scene);
            P = P + next_t * shadow_ray.dir;
            //shadow_ray.org = rayOrigin;
            //shadow_ray.tfar = (1 - get_shadow_epsilon(context.scene)) *
            //    distance(point_on_light.position, rayOrigin);
        }
    }
    *pdf = light_pmf(context.scene, light_id) *
        pdf_point_on_light(light, point_on_light, ref_point, context.scene);
    *p_trans = p_trans_dir;
    return T_light * emission(light, -dir_light, Real(0), point_on_light, context.scene);
}

// The fourth volumetric renderer: 
// multiple monochromatic homogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// still no surface lighting
Spectrum vol_path_tracing_4(const Scene& scene,
    int x, int y, /* pixel coordinates */
    pcg32_state& rng)
{
    // Generate camera ray
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
        (y + next_pcg32_real<Real>(rng)) / h);

    PathTracingContext context{
        scene,
        rng,
        sample_primary(scene.camera, screen_pos),
        RayDifferential{ Real(0), Real(0) },
        scene.camera.medium_id,
        0
    };

    Spectrum radiance = make_zero_spectrum();
    Spectrum current_path_throughput = make_const_spectrum(Real(1.0));
    Real dir_last_pdf = 0;
    Vector3 nee_p_cache = Vector3{ 0, 0, 0 };
    Real multi_trans_pdf = 1.0;
    bool specularPath = true;

    while (true)
    {
        bool scatter = false;
        std::optional<PathVertex> vertex_ = intersect(context.scene, context.ray, context.rayDiff);
        Spectrum transmittance = make_const_spectrum(1.0);
        Real trans_pdf = 1.0;

        // If the edge is inside a medium, we evaluate the scattering
        if (context.medium_id != -1)
        {
            Real tMax = std::numeric_limits<Real>::infinity();
            if (vertex_)
            {
                tMax = length(vertex_->position - context.ray.org);
            }
            Real t;
            scatter = sample_vol_scattering(context, tMax, &t, &transmittance, &trans_pdf);
            if (scatter)
            {
                specularPath = false;
                context.ray.org = context.ray.org + t * context.ray.dir;
                multi_trans_pdf *= trans_pdf;
            }

        }
        //else
        //{
        //    if (!vertex_)
        //    {
        //        // If no medium and doesn't hit anything, account for the environment map if needed.
        //        break;
        //    }
        //}

        current_path_throughput *= transmittance / trans_pdf;

        // If we hit a surface, accout for emission
        if (!scatter && vertex_)
        {
            PathVertex vertex = *vertex_;
            // We hit a light immediately. 
            // This path has only two vertices and has contribution
            // C = W(v0, v1) * G(v0, v1) * L(v0, v1)
            if (is_light(context.scene.shapes[vertex.shape_id]))
            {
                if (specularPath)
                {
                    radiance += current_path_throughput *
                        emission(vertex, -context.ray.dir, context.scene);
                }
                else
                {
                    int light_id = get_area_light_id(context.scene.shapes[vertex.shape_id]);
                    assert(light_id >= 0);
                    const Light& light = context.scene.lights[light_id];
                    Vector3 dir_light = normalize(vertex.position - nee_p_cache);
                    Real G = max(-dot(dir_light, vertex.geometric_normal), Real(0)) /
                        distance_squared(vertex.position, nee_p_cache);

                    PointAndNormal point_on_light = { vertex.position, vertex.geometric_normal };
                    Real pdf_nee = light_pmf(context.scene, light_id) * pdf_point_on_light(light, point_on_light, nee_p_cache, context.scene);
                    Real pdf_phase = dir_last_pdf * multi_trans_pdf * G;
                    Real w = (pdf_phase * pdf_phase) / (pdf_nee * pdf_nee + pdf_phase * pdf_phase);
                    radiance += current_path_throughput *
                        emission(vertex, -context.ray.dir, context.scene) * w;
                }
            }
        }

        // If we cannot continue scattering
        if (context.bounces == context.scene.options.max_depth - 1 && context.scene.options.max_depth != -1)
        {
            break;
        }

        // Account for surface scattering
        if (!scatter && vertex_)
        {
            PathVertex vertex = *vertex_;
            // index-matching interface, skip through it
            if (vertex.material_id == -1)
            {
                // check which side of the surface we are hitting
                context.medium_id = updateMedium(context, vertex, context.medium_id);

                Vector3 N = dot(context.ray.dir, vertex.geometric_normal) > 0 ? vertex.geometric_normal : -vertex.geometric_normal;
                Vector3 rayOrigin = vertex.position + context.ray.dir * get_intersection_epsilon(context.scene);
                context.ray.org = rayOrigin;

                context.bounces++;
                continue;
            }
            else
            {
                // Handle normal surface scattering
                break;
            }
        }

        // Sample phase function
        if (scatter)
        {
            Real pdf_nee;
            Real pdf_trans_dir;
            PointAndNormal point_on_light;
            Spectrum L_nee = next_event_estimation(context, context.ray.org, Vector3{ 0, 0, 0 },
                &pdf_nee, &pdf_trans_dir, &point_on_light);
            if (pdf_nee != 0)
            {
                Vector3 dir_light = normalize(point_on_light.position - context.ray.org);
                Real G = max(-dot(dir_light, point_on_light.normal), Real(0)) /
                    distance_squared(point_on_light.position, context.ray.org);
                const Medium& medium = context.scene.media[context.medium_id];
                PhaseFunction phaseFunction = get_phase_function(medium);
                Spectrum f_nee = eval(phaseFunction, -context.ray.dir, dir_light);

                // multiply by G because this time we put all pdf in area space
                Real pdf_phase = pdf_sample_phase(phaseFunction, -context.ray.dir, dir_light) * G * pdf_trans_dir;
                Real w = (pdf_nee * pdf_nee) / (pdf_nee * pdf_nee + pdf_phase * pdf_phase);
                radiance += current_path_throughput * (f_nee * G * L_nee / pdf_nee * w) * get_sigma_s(medium, context.ray.org);
                //printf("pdf_nee: %lf, pdf_phase: %lf\n", pdf_nee, pdf_phase);
            }

            // If we sampled a scattering event, trace according to phase function
            Vector3 wout;
            Real pdf_phase;
            Spectrum L = path_sample_phase_function(context, &wout, &pdf_phase);
            if (pdf_phase != 0)
            {
                current_path_throughput *= L / pdf_phase;
                dir_last_pdf = pdf_phase;
                nee_p_cache = context.ray.org;
                multi_trans_pdf = 1.0;
                context.ray.dir = wout;
            }
        }

        Real rr_prob = 1.0;
        if (context.bounces >= context.scene.options.rr_depth)
        {
            rr_prob = std::min(luminance(current_path_throughput), 0.95);
            if (next_pcg32_real<Real>(context.rng) > rr_prob)
            {
                break;
            }
            else
            {
                current_path_throughput /= rr_prob;
            }
        }
        context.bounces++;
    }
    return radiance;
}

// The fifth volumetric renderer: 
// multiple monochromatic homogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// with surface lighting
Spectrum vol_path_tracing_5(const Scene& scene,
    int x, int y, /* pixel coordinates */
    pcg32_state& rng)
{
    // Homework 2: implememt this!
    return make_zero_spectrum();
}

// The final volumetric renderer: 
// multiple chromatic heterogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// with surface lighting
Spectrum vol_path_tracing(const Scene& scene,
    int x, int y, /* pixel coordinates */
    pcg32_state& rng)
{
    // Homework 2: implememt this!
    return make_zero_spectrum();
}
