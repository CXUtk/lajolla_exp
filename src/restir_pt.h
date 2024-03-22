#pragma once
#include "path_tracing_helper.h"
#include "sampler.h"
#include "lightpaths.h"
#include <fstream>

#define RAW_PT_OUTPUT 0
#define DUMP_PT_PATHS 0



constexpr int RAND_SEQ_SIZE = 10;

void printfVec3(const Spectrum& v)
{
    printf("%lf %lf %lf", v.x, v.y, v.z);
}

LightPath restir_gen_camera_subpath(const Scene& scene,
    const Vector2& screenPos,
    ReplayableSampler& sampler,
    int maxDepth, const LightPathTree* halfVectorCopyTree = nullptr);


struct Reservoir
{
    LightPathTree pathTree;
    Real w_sum = 0;
    Real M = 0;
    Real W = 0;
    int x, y;

    void update(Real w, const LightPathTree& path, Real rnd_param)
    {
        if (isnan(w))
        {
            return;
        }
        w_sum += w;
        M += 1;

        if ( rnd_param * w_sum <= w)
        {
            this->pathTree = path;
        }
    }

    void combine(Real currentPdf, const Reservoir& R, Real rnd_param)
    {
        if (isnan(currentPdf))
        {
            return;
        }

        Real oldM = M;
        update(currentPdf * R.W * R.M, R.pathTree, rnd_param);
        //Real wi = (currentPdf / luminance(R.path.P_hat)) * R.w_sum;
        //w_sum += wi;
        M = oldM + R.M;

        //if (rnd_param * w_sum <= wi)
        //{
        //    path = R.path;
        //}
    }
};

bool restir_sample_vol_scattering_homogenized(int medium_id,
    const Ray& ray,
    ReplayableSampler& sampler,
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




void restir_random_walk(LightPath& path, const Scene& scene, const Ray& initRay, const RayDifferential& rayDiff,
    ReplayableSampler& sampler,
    Spectrum beta, Real pdf,
    int maxDepth, TransportDirection direction, const LightPathTree* halfVectorCopyTree)
{
    if (maxDepth == 0) return;

    int bounces = 0;
    // Declare variables for forward and reverse probability densities
    Real pdfFwd = pdf, pdfRev = 0;

    Ray currentRay = initRay;
    Spectrum throughput = make_const_spectrum(1.0);
    int currentMedium = path.Vertices[0].medium_id;
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

            scatter = restir_sample_vol_scattering_homogenized(currentMedium, currentRay, sampler, tMax,
                &t, &transmittance, &trans_dir_pdf, scene);

            beta *= transmittance / average(trans_dir_pdf);
            throughput *= transmittance;

            if (scatter)
            {
                currentRay.org = currentRay.org + t * currentRay.dir;
                Spectrum sigma_s = get_sigma_s(scene.media[currentMedium], currentRay.org);
                beta *= sigma_s;
                throughput *= sigma_s;

                PathVertex mi = CreateMedium(currentRay.org, currentMedium, beta, pdfFwd, path.Vertices.back());
                path.Vertices.push_back(mi);

                if (++bounces >= maxDepth)
                {
                    break;
                }
                PathVertex& curVertex = path.Vertices.back();
                PathVertex& prevVertex = path.Vertices[path.Vertices.size() - 2];
                Vector3 wout;
                Real pdf_phase;
                Real pdf_rev;

                Spectrum f = path_sample_phase_function(currentMedium, currentRay, scene, sampler, &wout, &pdf_phase, &pdf_rev);
                if (luminance(f) != 0 && pdf_phase != 0)
                {
                    pdfFwd = pdf_phase;
                    beta *= f / pdfFwd;
                    throughput *= f;
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
            vertex = CreateSurface(vertex, beta, throughput, pdfFwd, path.Vertices.back(), currentMedium);
            path.Vertices.push_back(vertex);

            if (++bounces >= maxDepth)
            {
                break;
            }

            PathVertex& curVertex = path.Vertices.back();
            PathVertex& prevVertex = path.Vertices[path.Vertices.size() - 2];

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

            curVertex.componentType = bsdf_sample_->sampleType;

            // Need copy half vector
            if (halfVectorCopyTree != nullptr)
            {
                int index = path.Vertices.size() - 1;

                if (index < halfVectorCopyTree->Vertices.size())
                {
                    const PathVertex& prev = halfVectorCopyTree->Vertices[index - 1].Vertex;
                    const PathVertex& cur = halfVectorCopyTree->Vertices[index].Vertex;
                    if (index + 1 < halfVectorCopyTree->Vertices.size())
                    {
                        const PathVertex& nxt = halfVectorCopyTree->Vertices[index + 1].Vertex;

                        Vector3 H = normalize(prev.position - cur.position) + normalize(nxt.position - cur.position);
                        H = normalize(H);

                        Vector3 wout = reflect(-win, H);

                        // TODO: Not implemented for transmission materials
                        if (dot(wout, vertex.shading_frame.n) > 0)
                        {
                            bsdf_sample_->dir_out = wout;
                        }
                    }
                }
            }
            Real pdf_bsdf = pdf_sample_bsdf(mat, win, bsdf_sample_->dir_out, curVertex, scene.texture_pool, direction);
            Spectrum f_bsdf = eval(mat, win, bsdf_sample_->dir_out, curVertex, scene.texture_pool, direction);


            pdfFwd = pdf_bsdf;

            if (pdfFwd == 0)
            {
                beta = make_zero_spectrum();
            }
            else
            {
                beta *= f_bsdf / pdfFwd * corrrect_shading_normal(curVertex, win, bsdf_sample_->dir_out, direction);
            }

            throughput *= f_bsdf;
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
                PathVertex vertex{};
                vertex.vertexType = VertexType::Light;
                vertex.geometric_normal = Vector3(0,0,0);
                vertex.shading_frame.n = Vector3(0, 0, 0);
                vertex.position = currentRay.org + currentRay.dir * 99999.0;
                vertex.light_id = -1;
                vertex.beta = beta;
                vertex.throughput = beta;
                vertex.componentType = ComponentType::None;
                path.Vertices.push_back(vertex);
            }
            break;
        }

    }
}

LightPathTree restir_gen_camera_nee_pathtree(const Scene& scene,
    int x, int y, int h, int w,
    ReplayableSampler& sampler,
    int maxDepth, bool replay=false, const LightPathTree* halfVectorCopyTree = nullptr)
{
    if (!replay)
    {
        sampler.start_iteration();
    }
    sampler.switch_stream(0);
    sampler.replay();

    //Vector2 screen_pos((x + 0.5) / w,
    //    (y + 0.5) / h);
    Vector2 screen_pos((x + sampler.next_double()) / w,
        (y + sampler.next_double()) / h);
    PathTracingContext context{
        scene,
        sampler,
        sample_primary(scene.camera, screen_pos),
        RayDifferential{ Real(0), Real(0) },
        scene.camera.medium_id,
        0 };
    LightPath path = restir_gen_camera_subpath(scene, screen_pos, sampler, maxDepth, halfVectorCopyTree);

    sampler.switch_stream(1);
    sampler.replay();

    LightPathTree pathtree{};

    PathVertex empty{};
    // Camera vertex does not have NEE branch
    pathtree.Vertices.push_back(LightPathTreeNode{ path.Vertices[0], empty });

    // Generate NNE branch for each vertex node
    for (int i = 1; i < path.Vertices.size(); i++)
    {
        const PathVertex& vertex = path.Vertices[i];
        const PathVertex& prevVertex = path.Vertices[i - 1];

        LightPathTreeNode curNode{};
        curNode.Vertex = vertex;

        if (vertex.vertexType == VertexType::Light)
        {
            curNode.NeeVertex = empty;
        }
        else
        {
            int self_light_id = -1;
            if (vertex.shape_id != -1)
            {
                self_light_id = get_area_light_id(scene.shapes[vertex.shape_id]);
            }

            Spectrum pdf_trans_dir;
            Spectrum pdf_trans_nee;
            PointAndNormal point_on_light;
            int lightId;
            Spectrum L_nee = next_event_estimation_homogenized(context, vertex.position, vertex.geometric_normal,
                &pdf_trans_dir, &pdf_trans_nee, &point_on_light, &lightId, self_light_id);

            const Material& mat = context.scene.materials[vertex.material_id];
            Vector3 win = normalize(prevVertex.position - vertex.position);
            Vector3 wout = normalize(point_on_light.position - vertex.position);
            Spectrum f_bsdf = eval(mat, win, wout, vertex, context.scene.texture_pool);

            Real G = max(dot(-wout, point_on_light.normal), Real(0)) /
                distance_squared(point_on_light.position, vertex.position);

            Real p_bsdf = pdf_sample_bsdf(
                mat, win, wout, vertex, context.scene.texture_pool) * G * average(pdf_trans_dir);
            Real pdf_nee = average(pdf_trans_nee);
            Real w = (pdf_nee * pdf_nee) / (pdf_nee * pdf_nee + p_bsdf * p_bsdf);

            PathVertex light_vertex{};
            light_vertex.vertexType = VertexType::Light;
            light_vertex.position = point_on_light.position;
            light_vertex.geometric_normal = point_on_light.normal;
            light_vertex.light_id = lightId;
            light_vertex.pdfFwd = pdf_nee;
            light_vertex.beta = vertex.beta * f_bsdf / pdf_nee * G;
            light_vertex.throughput = vertex.throughput * f_bsdf;

            curNode.L_Nee = L_nee;
            curNode.NeeVertex = light_vertex;
        }

        pathtree.Vertices.push_back(curNode);
    }
    pathtree.Sampler = sampler;
    return pathtree;
}


void restir_eval_nee_pathtree(LightPathTree& tree, const Scene& scene, bool selectLength, int selectedLength = -1)
{
    Spectrum L = make_zero_spectrum();
    Spectrum P_hat = make_zero_spectrum();

    for (int i = 1; i < tree.Vertices.size(); i++)
    {
        const LightPathTreeNode& node = tree.Vertices[i];
        const PathVertex& vertex = node.Vertex;
        const PathVertex& nee_vertex = node.NeeVertex;

        const PathVertex& prev_vertex = tree.Vertices[i - 1].Vertex;

        int selflightId = -1;
        if (hasLight(vertex, scene))
        {
            Vector3 win = prev_vertex.position - vertex.position;
            if (prev_vertex.vertexType == VertexType::Camera)
            {
                if (!selectLength || i == selectedLength - 1)
                {
                    L += emission(vertex, win, scene);
                    P_hat += emission(vertex, win, scene);
                }
            }
            else
            {
                int light_id = (vertex.light_id == -1) ? get_area_light_id(scene.shapes[vertex.shape_id]) : vertex.light_id;
                selflightId = light_id;
                assert(light_id >= 0);
                const Light& light = scene.lights[light_id];
                Vector3 dir_light = normalize(win);
                Real G = max(dot(dir_light, vertex.geometric_normal), Real(0)) /
                    distance_squared(vertex.position, prev_vertex.position);

                PointAndNormal point_on_light = { vertex.position, vertex.geometric_normal };
                Real pdf_nee = light_pmf(scene, light_id) * pdf_point_on_light(light, point_on_light, prev_vertex.position, scene);
                Real pdf_scatter = vertex.pdfFwd;


                Real w = (pdf_scatter * pdf_scatter) / (pdf_nee * pdf_nee + pdf_scatter * pdf_scatter);

                if (!selectLength || i == selectedLength - 1)
                {
                    L += vertex.beta * emission(vertex, dir_light, scene) * w;
                    P_hat += vertex.throughput * emission(vertex, dir_light, scene) * w;
                }
            }
        }

        if (vertex.vertexType == VertexType::Light || i == scene.options.max_depth - 1)
        {
            continue;
        }
        if (luminance(node.L_Nee) > 0)
        {
            const Material& mat = scene.materials[vertex.material_id];
            Vector3 win = normalize(prev_vertex.position - vertex.position);
            Vector3 wout = normalize(nee_vertex.position - vertex.position);
            Spectrum f_bsdf = eval(mat, win, wout, vertex, scene.texture_pool);

            Real G = max(dot(-wout, nee_vertex.geometric_normal), Real(0)) /
                distance_squared(nee_vertex.position, vertex.position);

            Real p_bsdf = pdf_sample_bsdf(
                mat, win, wout, vertex, scene.texture_pool) * G ;
            Real pdf_nee = nee_vertex.pdfFwd;
            Real w = (pdf_nee * pdf_nee) / (pdf_nee * pdf_nee + p_bsdf * p_bsdf);

            if (!selectLength || i == selectedLength - 2)
            {
                L += nee_vertex.beta * node.L_Nee * w;
                P_hat += nee_vertex.throughput * node.L_Nee * w;
            }
        }
    }

    tree.L = L;
    tree.P_hat = P_hat;
}

Spectrum restir_select_pathtree(const LightPathTree& tree, Spectrum* throughput, const Scene& scene, int selectedLength = 3, bool hasNEE = false)
{
    Spectrum L = make_zero_spectrum();
    *throughput = make_zero_spectrum();

    for (int i = 1; i < tree.Vertices.size(); i++)
    {
        const LightPathTreeNode& node = tree.Vertices[i];
        const PathVertex& vertex = node.Vertex;
        const PathVertex& nee_vertex = node.NeeVertex;

        const PathVertex& prev_vertex = tree.Vertices[i - 1].Vertex;

        if (hasNEE && vertex.vertexType != VertexType::Light && i != scene.options.max_depth - 1)
        {
            if (luminance(node.L_Nee) > 0)
            {
                const Material& mat = scene.materials[vertex.material_id];
                Vector3 win = normalize(prev_vertex.position - vertex.position);
                Vector3 wout = normalize(nee_vertex.position - vertex.position);
                Spectrum f_bsdf = eval(mat, win, wout, vertex, scene.texture_pool);

                Real G = max(dot(-wout, nee_vertex.geometric_normal), Real(0)) /
                    distance_squared(nee_vertex.position, vertex.position);

                Real p_bsdf = pdf_sample_bsdf(
                    mat, win, wout, vertex, scene.texture_pool) * G;
                Real pdf_nee = nee_vertex.pdfFwd;
                Real w = (pdf_nee * pdf_nee) / (pdf_nee * pdf_nee + p_bsdf * p_bsdf);

                if (i == selectedLength - 2)
                {
                    L += nee_vertex.beta * node.L_Nee * w;
                    *throughput += nee_vertex.throughput * node.L_Nee;
                }
            }
        }
        
        if (hasLight(vertex, scene))
        {
            Vector3 win = prev_vertex.position - vertex.position;
            if (prev_vertex.vertexType == VertexType::Camera)
            {
                if (i == selectedLength - 1)
                {
                    L += emission(vertex, win, scene);
                    *throughput += emission(vertex, win, scene);
                }
            }
            else
            {
                int light_id = (vertex.light_id == -1) ? get_area_light_id(scene.shapes[vertex.shape_id]) : vertex.light_id;
                assert(light_id >= 0);
                const Light& light = scene.lights[light_id];
                Vector3 dir_light = normalize(win);
                Real G = max(dot(dir_light, vertex.geometric_normal), Real(0)) /
                    distance_squared(vertex.position, prev_vertex.position);

                PointAndNormal point_on_light = { vertex.position, vertex.geometric_normal };
                Real pdf_nee = light_pmf(scene, light_id) * pdf_point_on_light(light, point_on_light, prev_vertex.position, scene);
                Real pdf_scatter = vertex.pdfFwd;


                Real w = (pdf_scatter * pdf_scatter) / (pdf_nee * pdf_nee + pdf_scatter * pdf_scatter);

                if (i == selectedLength - 1)
                {
                    
                    if (!hasNEE)
                    {
                        w = 1;
                    }
                    L += vertex.beta * emission(vertex, dir_light, scene) * w;
                    *throughput += vertex.throughput * emission(vertex, dir_light, scene);
                }
            }
        }
    }
    return L;
}


LightPath restir_gen_camera_subpath(const Scene& scene,
    const Vector2& screenPos,
    ReplayableSampler& sampler,
    int maxDepth, const LightPathTree* halfVectorCopyTree)
{
    LightPath path{};
    path.ScreenPos = screenPos;
    if (maxDepth == 0) return path;

    Ray ray = sample_primary(scene.camera, screenPos);
    RayDifferential ray_diff = RayDifferential{ Real(0), Real(0) };

    PathVertex cameraVertex = CreateCamera(scene.camera, make_const_spectrum(Real(1)));
    cameraVertex.geometric_normal = xform_vector(scene.camera.cam_to_world,
        Vector3{ 0, 0, 1 });
    cameraVertex.shading_frame = Frame(cameraVertex.geometric_normal);
    cameraVertex.medium_id = scene.camera.medium_id;
    path.Vertices.push_back(cameraVertex);

    Real pdf_pos, pdf_dir;
    pdf_importance(scene.camera, ray.dir, &pdf_pos, &pdf_dir);

    cameraVertex.pdfFwd = pdf_dir;
    restir_random_walk(path, scene, ray, ray_diff, sampler, cameraVertex.beta, pdf_dir, maxDepth - 1,
        TransportDirection::TO_LIGHT, halfVectorCopyTree);
    return path;
}

std::vector<LightPath> restir_select_subpath_nee(const PathTracingContext& context,
    const LightPath& path,
    int length)
{
    std::vector<LightPath> paths;
    LightPath output{};
    output.ScreenPos = path.ScreenPos;
    if (path.Vertices.size() < length - 1)
    {
        output.Vertices = path.Vertices;
        output.L = make_const_spectrum(0);
        output.P_hat = make_const_spectrum(0);
        output.Sampler = context.sampler;
        paths.push_back(output);
        output.SampleMethod = SampleMethod::NEE;
        paths.push_back(output);
        return paths;
    }

    int light_id = -1;
    // Direct Case
    if (path.Vertices.size() >= length && path.Vertices[length - 1].shape_id != -1 && is_light(context.scene.shapes[path.Vertices[length - 1].shape_id]))
    {
        const PathVertex& vertex = path.Vertices[length - 1];
        const PathVertex& prev_vertex = path.Vertices[length - 2];
        output.Vertices = path.Vertices;

        light_id = (vertex.light_id == -1) ? get_area_light_id(context.scene.shapes[vertex.shape_id]) : vertex.light_id;
        assert(light_id >= 0);
        const Light& light = context.scene.lights[light_id];
        Vector3 win = prev_vertex.position - vertex.position;
        Vector3 dir_light = normalize(win);
        Real G = max(dot(dir_light, vertex.geometric_normal), Real(0)) / dot(win, win);

        PointAndNormal point_on_light = { vertex.position, vertex.geometric_normal };
        Real pdf_nee = light_pmf(context.scene, light_id) * pdf_point_on_light(light, point_on_light, prev_vertex.position, context.scene);
        Real pdf_scatter = vertex.pdfFwd;
        Real w = (pdf_scatter * pdf_scatter) / (pdf_nee * pdf_nee + pdf_scatter * pdf_scatter);

        Spectrum E = emission(vertex, dir_light, context.scene);
        output.L = vertex.beta * E;
        output.P_hat = vertex.throughput * E;
        output.Sampler = context.sampler;
        output.Pdf1 = pdf_scatter;
        output.Pdf2 = pdf_nee;
        paths.push_back(output);
    }
    else
    {
        output.Vertices = path.Vertices;
        output.L = make_const_spectrum(0);
        output.P_hat = make_const_spectrum(0);
        output.Sampler = context.sampler;
        paths.push_back(output);
    }

    // NEE Case
    assert(length >= 3);
    const PathVertex& vertex = path.Vertices[length - 2];
    const PathVertex& prev_vertex = path.Vertices[length - 3];

    if (vertex.vertexType == VertexType::Light)
    {
        output.Vertices = path.Vertices;
        output.L = make_const_spectrum(0);
        output.P_hat = make_const_spectrum(0);
        output.Sampler = context.sampler;
        output.SampleMethod = SampleMethod::NEE;
        paths.push_back(output);
        return paths;
    }

    output.Vertices.clear();
    for (int i = 0; i < length - 1; i++)
    {
        output.Vertices.push_back(path.Vertices[i]);
    }

    int self_light_id = -1;
    if (vertex.shape_id != -1)
    {
        self_light_id = get_area_light_id(context.scene.shapes[vertex.shape_id]);
    }

    Spectrum pdf_trans_dir;
    Spectrum pdf_trans_nee;
    PointAndNormal point_on_light;
    int lightId;
    Spectrum L_nee = next_event_estimation_homogenized(context, vertex.position, vertex.geometric_normal,
        &pdf_trans_dir, &pdf_trans_nee, &point_on_light, &lightId, self_light_id);

    if (max(L_nee) > 0)
    {
        const Material& mat = context.scene.materials[vertex.material_id];
        Vector3 win = normalize(prev_vertex.position - vertex.position);
        Vector3 wout = normalize(point_on_light.position - vertex.position);
        Spectrum f_bsdf = eval(mat, win, wout, vertex, context.scene.texture_pool);

        Real G = max(dot(-wout, point_on_light.normal), Real(0)) /
            distance_squared(point_on_light.position, vertex.position);

        Real p_bsdf = pdf_sample_bsdf(
            mat, win, wout, vertex, context.scene.texture_pool) * G * average(pdf_trans_dir);
        Real pdf_nee = average(pdf_trans_nee);
        Real w = (pdf_nee * pdf_nee) / (pdf_nee * pdf_nee + p_bsdf * p_bsdf);

        PathVertex lvertex{};
        lvertex.vertexType = VertexType::Light;
        lvertex.position = point_on_light.position;
        lvertex.geometric_normal = point_on_light.normal;
        lvertex.light_id = lightId;
        lvertex.beta = vertex.beta * f_bsdf * G / pdf_nee;
        lvertex.throughput = vertex.throughput * f_bsdf * G;
        lvertex.pdfFwd = pdf_nee / G;

        output.Vertices.push_back(lvertex);
        output.L = lvertex.beta * L_nee;
        output.P_hat = lvertex.throughput * L_nee;

        output.Sampler = context.sampler;
        output.SampleMethod = SampleMethod::NEE;
        paths.push_back(output);

    }
    else
    {
        output.Vertices = path.Vertices;
        output.L = make_const_spectrum(0);
        output.P_hat = make_const_spectrum(0);
        output.Sampler = context.sampler;
        output.SampleMethod = SampleMethod::NEE;
        paths.push_back(output);
    }
    return paths;
}

std::vector<LightPath> restir_select_subpath(const PathTracingContext& context,
    const LightPath& path,
    int length)
{
    std::vector<LightPath> paths;
    LightPath output;
    output.ScreenPos = path.ScreenPos;
    if (path.Vertices.size() < length)
    {
        output.Vertices = path.Vertices;
        output.L = make_const_spectrum(0);
        output.P_hat = make_const_spectrum(0);
        output.Sampler = context.sampler;
        paths.push_back(output);
        return paths;
    }

    int light_id = -1;
    // Direct Case
    if (path.Vertices.size() == length && path.Vertices[length - 1].shape_id != -1 && is_light(context.scene.shapes[path.Vertices[length - 1].shape_id]))
    {
        const PathVertex& vertex = path.Vertices[length - 1];
        const PathVertex& prev_vertex = path.Vertices[length - 2];
        output.Vertices = path.Vertices;

        light_id = (vertex.light_id == -1) ? get_area_light_id(context.scene.shapes[vertex.shape_id]) : vertex.light_id;
        assert(light_id >= 0);
        const Light& light = context.scene.lights[light_id];
        Vector3 win = prev_vertex.position - vertex.position;
        Vector3 dir_light = normalize(win);
        Real G = max(dot(dir_light, vertex.geometric_normal), Real(0)) / dot(win, win);

        PointAndNormal point_on_light = { vertex.position, vertex.geometric_normal };
        Real pdf_nee = light_pmf(context.scene, light_id) * pdf_point_on_light(light, point_on_light, prev_vertex.position, context.scene);
        Real pdf_scatter = vertex.pdfFwd;
        Real w = (pdf_scatter * pdf_scatter) / (pdf_nee * pdf_nee + pdf_scatter * pdf_scatter);

        Spectrum E = emission(vertex, dir_light, context.scene);
        output.L = vertex.beta * E;
        output.P_hat = vertex.throughput * E;
        output.Sampler = context.sampler;
        paths.push_back(output);
    }
    else
    {
        output.Vertices = path.Vertices;
        output.L = make_const_spectrum(0);
        output.P_hat = make_const_spectrum(0);
        output.Sampler = context.sampler;
        paths.push_back(output);
    }


    // NEE Case
    //if(length >= 3)
    //{
    //    const PathVertex& vertex = path.Vertices[length - 2];
    //    const PathVertex& prev_vertex = path.Vertices[length - 3];

    //    if (vertex.vertexType == VertexType::Light)
    //    {
    //        output.Vertices = path.Vertices;
    //        output.L = make_const_spectrum(0);
    //        output.P_hat = make_const_spectrum(0);
    //        output.Sampler = context.sampler;
    //        paths.push_back(output);
    //        return paths;
    //    }

    //    output.Vertices.clear();
    //    for (int i = 0; i < length - 1; i++)
    //    {
    //        output.Vertices.push_back(path.Vertices[i]);
    //    }

    //    int self_light_id = -1;
    //    if (vertex.shape_id != -1)
    //    {
    //        self_light_id = get_area_light_id(context.scene.shapes[vertex.shape_id]);
    //    }

    //    Spectrum pdf_trans_dir;
    //    Spectrum pdf_trans_nee;
    //    PointAndNormal point_on_light;
    //    int lightId;
    //    Spectrum L_nee = next_event_estimation_homogenized(context, vertex.position, vertex.geometric_normal,
    //        &pdf_trans_dir, &pdf_trans_nee, &point_on_light, &lightId, self_light_id);

    //    if (max(L_nee) > 0)
    //    {
    //        const Material& mat = context.scene.materials[vertex.material_id];
    //        Vector3 win = normalize(prev_vertex.position - vertex.position);
    //        Vector3 wout = normalize(point_on_light.position - vertex.position);
    //        Spectrum f_bsdf = eval(mat, win, wout, vertex, context.scene.texture_pool);

    //        Real G = max(dot(-wout, point_on_light.normal), Real(0)) /
    //            distance_squared(point_on_light.position, vertex.position);

    //        Real p_bsdf = pdf_sample_bsdf(
    //            mat, win, wout, vertex, context.scene.texture_pool) * G * average(pdf_trans_dir);
    //        Real pdf_nee = average(pdf_trans_nee);
    //        Real w = (pdf_nee * pdf_nee) / (pdf_nee * pdf_nee + p_bsdf * p_bsdf);

    //        PathVertex lvertex{};
    //        lvertex.vertexType = VertexType::Light;
    //        lvertex.position = point_on_light.position;
    //        lvertex.geometric_normal = point_on_light.normal;
    //        lvertex.light_id = lightId;
    //        lvertex.beta = vertex.beta * f_bsdf / pdf_nee * G;
    //        lvertex.throughput = vertex.throughput * f_bsdf;
    //        lvertex.pdfFwd = pdf_nee / G;

    //        output.Vertices.push_back(lvertex);
    //        output.L = lvertex.beta * L_nee;
    //        output.P_hat = lvertex.throughput * L_nee;

    //        output.Sampler = context.sampler;
    //        paths.push_back(output);

    //    }
    //    else
    //    {
    //        output.Vertices = path.Vertices;
    //        output.L = make_const_spectrum(0);
    //        output.P_hat = make_const_spectrum(0);
    //        output.Sampler = context.sampler;
    //        paths.push_back(output);
    //    }

    //}
    //else
    //{
    //    output.Vertices = path.Vertices;
    //    output.L = make_const_spectrum(0);
    //    output.P_hat = make_const_spectrum(0);
    //    output.Sampler = context.sampler;
    //    paths.push_back(output);
    //    return paths;
    //}

    return paths;
}

bool restir_reeval_lightPath(const PathTracingContext& context, LightPath& path)
{
    TransportDirection direction = TransportDirection::TO_LIGHT;
    Spectrum beta = make_const_spectrum(1) ;
    Spectrum throughput = make_const_spectrum(1);
    for (int i = 1; i < path.Vertices.size(); i++)
    {
        PathVertex& start = path.Vertices[i - 1];
        PathVertex& end = path.Vertices[i];
        Vector3 orig = start.position;
        Ray shadow_ray{ orig, end.position - orig,
            get_shadow_epsilon(context.scene),
            (1 - get_shadow_epsilon(context.scene)) };
        std::optional<PathVertex> vertex_ = intersect(context.scene, shadow_ray, context.rayDiff);

        if (vertex_)
        {
            return false;
        }

        if (i < path.Vertices.size() - 1)
        {
            PathVertex& next = path.Vertices[i + 1];
            const Material& mat = context.scene.materials[end.material_id];
            // Let's do the hemispherical sampling next.
            Vector3 win = normalize(start.position - end.position);
            Vector3 wout = normalize(next.position - end.position);
            Real pdf_bsdf = pdf_sample_bsdf(mat, win, wout, end, context.scene.texture_pool, direction);
            Spectrum f_bsdf = eval(mat, win, wout, end, context.scene.texture_pool, direction);

            if (pdf_bsdf == 0)
            {
                return false;
            }

            beta *= f_bsdf / pdf_bsdf;
            throughput *= f_bsdf;

            next.beta = beta;
            next.throughput = throughput;
        }
    }
    return true;
    
}


bool restir_reeval_lightPathTree(const Scene& scene, LightPathTree& pathTree)
{
    TransportDirection direction = TransportDirection::TO_LIGHT;
    Spectrum beta = make_const_spectrum(1);
    Spectrum throughput = make_const_spectrum(1);
    for (int i = 1; i < pathTree.Vertices.size(); i++)
    {
        PathVertex& start = pathTree.Vertices[i - 1].Vertex;
        PathVertex& end = pathTree.Vertices[i].Vertex;
        Vector3 orig = start.position;
        Ray shadow_ray{ orig, end.position - orig,
            get_shadow_epsilon(scene),
            (1 - get_shadow_epsilon(scene)) };
        std::optional<PathVertex> vertex_ = intersect(scene, shadow_ray, RayDifferential{ Real(0), Real(0) });

        if (vertex_)
        {
            return false;
        }

        if (i < pathTree.Vertices.size() - 1)
        {
            PathVertex& next = pathTree.Vertices[i + 1].Vertex;
            const Material& mat = scene.materials[end.material_id];
            // Let's do the hemispherical sampling next.
            Vector3 win = normalize(start.position - end.position);
            Vector3 wout = normalize(next.position - end.position);
            Real pdf_bsdf = pdf_sample_bsdf(mat, win, wout, end, scene.texture_pool, direction);
            Spectrum f_bsdf = eval(mat, win, wout, end, scene.texture_pool, direction);

            if (pdf_bsdf == 0)
            {
                return false;
            }

            beta *= f_bsdf / pdf_bsdf;
            throughput *= f_bsdf;

            next.beta = beta;
            next.throughput = throughput;
        }
    }
    return true;

}

Spectrum restir_eval_lightPath_no_nee(const PathTracingContext& context, const LightPath& path, Spectrum* throughput, int selectLength = -1)
{
    int length = path.Vertices.size();
    Spectrum L = make_zero_spectrum();
    *throughput = make_zero_spectrum();
    if (selectLength != -1 && length < selectLength)
    {
        return L;
    }

    PathTracingContext ctx(context);

    // Next event estimation
    for (int i = 1; i < length; i++)
    {
        ctx.bounces = i;

        const PathVertex& vertex = path.Vertices[i];
        const PathVertex& prev_vertex = path.Vertices[i - 1];

        //if (i == selectLength && !hasLight(vertex, context.scene))
        //{
        //    break;
        //}

        int selflightId = -1;
        if (hasLight(vertex, context.scene))
        {
            Vector3 win = prev_vertex.position - vertex.position;
            if (vertex.vertexType == VertexType::Light)
            {
                if (selectLength == -1 || i == selectLength - 1)
                {
                    int light_id = (vertex.light_id == -1) ? get_area_light_id(context.scene.shapes[vertex.shape_id]) : vertex.light_id;
                    assert(light_id >= 0);
                    const Light& light = context.scene.lights[light_id];
                    Vector3 dir_light = normalize(win);
                    Real G = max(dot(dir_light, vertex.geometric_normal), Real(0)) /
                        distance_squared(vertex.position, prev_vertex.position);

                    // Ignore
                    L += vertex.beta * G * emission(vertex, win, context.scene);
                    *throughput += vertex.throughput * emission(vertex, win, context.scene);


                }
            }
            else if (prev_vertex.vertexType == VertexType::Camera)
            {
                if (selectLength == -1 || i == selectLength - 1)
                {
                    L += emission(vertex, win, context.scene);
                    *throughput += emission(vertex, win, context.scene);
                }
            }
            else
            {

                //PointAndNormal point_on_light = { vertex.position, vertex.geometric_normal };
                //Real pdf_nee = light_pmf(context.scene, light_id) * pdf_point_on_light(light, point_on_light, prev_vertex.position, context.scene);
                //Real pdf_scatter = vertex.pdfFwd;
                //Real w = (pdf_scatter * pdf_scatter) / (pdf_nee * pdf_nee + pdf_scatter * pdf_scatter);

                if (selectLength == -1 || i == selectLength - 1)
                {
                    L += vertex.beta * emission(vertex, win, context.scene);
                    *throughput += vertex.throughput * emission(vertex, win, context.scene);
                }
            }
        }

        if (vertex.vertexType != VertexType::Light)
        {
            //Spectrum pdf_trans_dir;
            //Spectrum pdf_trans_nee;
            //PointAndNormal point_on_light;
            //int lightId;
            //Spectrum L_nee = next_event_estimation_homogenized(context, vertex.position, vertex.geometric_normal,
            //    &pdf_trans_dir, &pdf_trans_nee, &point_on_light, &lightId, selflightId);

            //if (max(L_nee) > 0)
            //{
            //    const Material& mat = context.scene.materials[vertex.material_id];
            //    Vector3 win = normalize(prev_vertex.position - vertex.position);
            //    Vector3 wout = normalize(point_on_light.position - vertex.position);
            //    Spectrum f_bsdf = eval(mat, win, wout, vertex, context.scene.texture_pool);

            //    Real G = max(dot(-wout, point_on_light.normal), Real(0)) /
            //        distance_squared(point_on_light.position, vertex.position);

            //    Real p_bsdf = pdf_sample_bsdf(
            //        mat, win, wout, vertex, context.scene.texture_pool) * G * average(pdf_trans_dir);
            //    Real pdf_nee = average(pdf_trans_nee);
            //    Real w = (pdf_nee * pdf_nee) / (pdf_nee * pdf_nee + p_bsdf * p_bsdf);

            //    if (selectLength == -1 || i == selectLength - 2)
            //    {
            //        //L += vertex.beta * f_bsdf * L_nee / pdf_nee * w * G;
            //    }
            //}
        }
    }

    return L;
}

Spectrum restir_eval_lightPath_full(const PathTracingContext& context, const LightPath& path, Spectrum* throughput, bool selectLength, int selectedLength = -1)
{
    int length = path.Vertices.size();
    Spectrum L = make_zero_spectrum();
    *throughput = make_zero_spectrum();
    if (selectLength && length < selectedLength - 1)
    {
        return L;
    }

    PathTracingContext ctx(context);

    // Next event estimation
    for (int i = 1; i < length; i++)
    {
        ctx.bounces = i;

        const PathVertex& vertex = path.Vertices[i];
        const PathVertex& prev_vertex = path.Vertices[i - 1];

        int selflightId = -1;
        if (hasLight(vertex, context.scene))
        {
            Vector3 win = prev_vertex.position - vertex.position;
            if (prev_vertex.vertexType == VertexType::Camera)
            {
                if (!selectLength || i == selectedLength - 1)
                {
                    L += emission(vertex, win, context.scene);
                    *throughput += emission(vertex, win, context.scene);
                }
            }
            else
            {
                int light_id = (vertex.light_id == -1) ? get_area_light_id(context.scene.shapes[vertex.shape_id]) : vertex.light_id;
                selflightId = light_id;
                assert(light_id >= 0);
                const Light& light = context.scene.lights[light_id];
                Vector3 dir_light = normalize(win);
                Real G = max(dot(dir_light, vertex.geometric_normal), Real(0)) /
                    distance_squared(vertex.position, prev_vertex.position);

                PointAndNormal point_on_light = { vertex.position, vertex.geometric_normal };
                Real pdf_nee = light_pmf(context.scene, light_id) * pdf_point_on_light(light, point_on_light, prev_vertex.position, context.scene);
                Real pdf_scatter = vertex.pdfFwd;


                Real w = (pdf_scatter * pdf_scatter) / (pdf_nee * pdf_nee + pdf_scatter * pdf_scatter);

                if (!selectLength || i == selectedLength - 1)
                {
                    L += vertex.beta * emission(vertex, dir_light, context.scene) * w;
                    *throughput += vertex.throughput * emission(vertex, dir_light, context.scene);
                }
            }
        }
        if (vertex.vertexType == VertexType::Light)
        {
            continue;
        }
        if (i >= length - 1)
        {
            continue;
        }
        Spectrum pdf_trans_dir;
        Spectrum pdf_trans_nee;
        PointAndNormal point_on_light;
        int lightId;
        Spectrum L_nee = next_event_estimation_homogenized(context, vertex.position, vertex.geometric_normal,
            &pdf_trans_dir, &pdf_trans_nee, &point_on_light, &lightId, selflightId);

        if (max(L_nee) > 0)
        {
            const Material& mat = context.scene.materials[vertex.material_id];
            Vector3 win = normalize(prev_vertex.position - vertex.position);
            Vector3 wout = normalize(point_on_light.position - vertex.position);
            Spectrum f_bsdf = eval(mat, win, wout, vertex, context.scene.texture_pool);

            Real G = max(dot(-wout, point_on_light.normal), Real(0)) /
                distance_squared(point_on_light.position, vertex.position);

            Real p_bsdf = pdf_sample_bsdf(
                mat, win, wout, vertex, context.scene.texture_pool) * G * average(pdf_trans_dir);
            Real pdf_nee = average(pdf_trans_nee);
            Real w = (pdf_nee * pdf_nee) / (pdf_nee * pdf_nee + p_bsdf * p_bsdf);

            if (!selectLength || i == selectedLength - 2)
            {
                L += vertex.beta * f_bsdf / pdf_nee * L_nee * G * w;
                *throughput += vertex.throughput * f_bsdf * G * L_nee;
            }
        }
    }

    return L;
}

void exportTrainingData(const std::unique_ptr<LightPathSerializationInfo[]>& train, const std::unique_ptr<LightPathSerializationInfo[]>& val, int h, int w, int spp)
{
    std::ofstream myfile("train.dat", std::ios::binary | std::ios::out);
    std::ofstream valFile("val.dat", std::ios::binary | std::ios::out);

    if (myfile.is_open())
    {
        int x = RAND_SEQ_SIZE;
        int y = 3;
        uint64_t z = (uint64_t)h * w * spp / 2;
        myfile.write(reinterpret_cast<const char*>(&x), sizeof(int));
        myfile.write(reinterpret_cast<const char*>(&y), sizeof(int));
        myfile.write(reinterpret_cast<const char*>(&z), sizeof(uint64_t));
        // Write the array of doubles to the file
        for (uint64_t i = 0; i < z; i++)
        {
            myfile.write(reinterpret_cast<const char*>(train[i].Sequence.data()), sizeof(Real) * RAND_SEQ_SIZE);
            myfile.write(reinterpret_cast<const char*>(&train[i].Throughput), sizeof(Real) * 3);
        }
        myfile.close(); // Close the file
        std::cout << "File written successfully." << std::endl;
    }
    else
    {
        std::cout << "Unable to open file for writing." << std::endl;
    }

    if (valFile.is_open())
    {
        int x = RAND_SEQ_SIZE;
        int y = 3;
        uint64_t z = (uint64_t)h * w * spp / 2;
        valFile.write(reinterpret_cast<const char*>(&x), sizeof(int));
        valFile.write(reinterpret_cast<const char*>(&y), sizeof(int));
        valFile.write(reinterpret_cast<const char*>(&z), sizeof(uint64_t));
        // Write the array of doubles to the file
        for (uint64_t i = 0; i < z; i++)
        {
            valFile.write(reinterpret_cast<const char*>(val[i].Sequence.data()), sizeof(Real) * RAND_SEQ_SIZE);
            valFile.write(reinterpret_cast<const char*>(&val[i].Throughput), sizeof(Real) * 3);
        }
        valFile.close(); // Close the file
        std::cout << "File written successfully." << std::endl;
    }
    else
    {
        std::cout << "Unable to open file for writing." << std::endl;
    }
}


int dx[4] = { -1, 1, 0, 0 };
int dy[4] = { 0, 0, -1, 1 };

Image3 do_restir_pt(const Scene& scene)
{
    int w = scene.camera.width, h = scene.camera.height;
    Image3 img(w, h);
    Image1 img_cnt(w, h);
    std::mutex img_mutex;

    constexpr int tile_size = 16;
    int num_tiles_x = (w + tile_size - 1) / tile_size;
    int num_tiles_y = (h + tile_size - 1) / tile_size;

    int spp = scene.options.samples_per_pixel;

    Real MSE = 0;
    uint64_t countShift = 0;

    ProgressReporter reporter((uint64_t)num_tiles_x * num_tiles_y * (2 + scene.options.iterations));
    auto pathReservoirs = std::make_unique<Reservoir[]>(w * h);
    auto pathReservoirsSwap = std::make_unique<Reservoir[]>(w * h);

    auto path_train = std::make_unique<LightPathSerializationInfo[]>((uint64_t)w * h * spp / 2);
    auto path_val = std::make_unique<LightPathSerializationInfo[]>((uint64_t)w * h * spp / 2);

    parallel_for([&](const Vector2i& tile)
        {
            // Use a different rng stream for each thread.
            pcg32_state rng = init_pcg32(tile[1] * num_tiles_x + tile[0]);
            pcg32_state rng2 = init_pcg32(2435768 + tile[1] * num_tiles_x + tile[0]);
            int x0 = tile[0] * tile_size;
            int x1 = min(x0 + tile_size, w);
            int y0 = tile[1] * tile_size;
            int y1 = min(y0 + tile_size, h);
            ReplayableSampler sampler(rng);

            for (int y = y0; y < y1; y++)
            {
                for (int x = x0; x < x1; x++)
                {
                    Reservoir R_BSDF{};
                    Reservoir R_NEE{};
                    R_NEE.x = x;
                    R_NEE.y = y;

                    for (int i = 0; i < spp; i++)
                    {
                        Spectrum radiance = make_zero_spectrum();


                        //LightPath cameraPath = restir_gen_camera_subpath(scene, screen_pos, sampler, scene.options.max_depth);
                        LightPathTree cameraPathTree = restir_gen_camera_nee_pathtree(scene, x, y, w, h, sampler, scene.options.max_depth);

#if RAW_PT_OUTPUT
                        //restir_eval_nee_pathtree(cameraPathTree, scene, false, scene.options.max_depth);
                        Spectrum throughput;
                        Spectrum L = restir_select_pathtree(cameraPathTree, &throughput, scene, scene.options.max_depth, true);
#else
                        //restir_eval_nee_pathtree(cameraPathTree, scene, false, scene.options.max_depth);

                        Spectrum throughput;
                        

                        Spectrum L = restir_select_pathtree(cameraPathTree, &throughput, scene, scene.options.max_depth);
                        if (isfinite(L) && !isnan(L))
                        {
                            cameraPathTree.L = L;
                            cameraPathTree.P_hat = throughput;
                            R_NEE.update(luminance(L), cameraPathTree, next_pcg32_real<Real>(rng2));

#if DUMP_PT_PATHS
                            while (sampler.X.size() < 10)
                            {
                                sampler.next_double();
                            }
                            std::vector<Real> rand_seq;
                            rand_seq.push_back(x / (Real)w);
                            rand_seq.push_back(y / (Real)h);

                            rand_seq.insert(rand_seq.end(), sampler.X.begin(), sampler.X.end());
                            if (i < spp / 2)
                            {
                                path_train[(y * w + x) * spp / 2 + i] = LightPathSerializationInfo{ rand_seq, throughput };
                            }
                            else
                            {
                                path_val[(y * w + x) * spp / 2 + (i - spp / 2)] = LightPathSerializationInfo{ rand_seq, throughput };
                            }
#endif
                        }
                    //sampler.switch_stream(1);
                        //sampler.start_iteration();
                        //std::vector<LightPath> paths = restir_select_subpath_nee(context, cameraPath, scene.options.max_depth);
                        //for (const auto& path : paths)
                        //{
                        //    Spectrum L = path.L;
                        //    if (isfinite(L) && !isnan(L))
                        //    {
                        //        //Reservoir smallRes{};
                        //        //smallRes.update(luminance(L), path, sampler.next_double());
                        //        //pathReservoir.combine(luminance(path.P_hat), smallRes, sampler.next_double());
                        //        if (path.SampleMethod == SampleMethod::BSDF)
                        //        {
                        //            R_BSDF.update(luminance(L), path, sampler.next_double());
                        //        }
                        //        else if (path.SampleMethod == SampleMethod::NEE)
                        //        {
                        //            R_NEE.update(luminance(L), path, sampler.next_double());
                        //        }
                        //        //if (luminance(L) > 0)
                        //        //{
                        //        //    printf("");
                        //        //}

                        //    }
                        //}
#endif

                        //cameraPath.Sampler = sampler;
                        //cameraPath.L = L;
                        //Real pdf = 1;
                        //for (int k = 1; k < cameraPath.Vertices.size(); k++)
                        //{
                        //    pdf *= cameraPath.Vertices[k].pdfFwd;
                        //}
                        //if (pdf > 0)
                        //{
                        //    printf("");
                        //}

#if RAW_PT_OUTPUT
                        if (isfinite(L) && !isnan(L))
                        {
                            // Hacky: exclude NaNs in the rendering.
                            img(x, y) += L;
                            img_cnt(x, y) += 1;
                        }
                        else
                        {
                            printf("Error: L is not finite.\n");
                        }
#endif
                    }

                    //Real pdf_bsdf = R_BRDF.path.Vertices[scene.options.max_depth - 1].pdfFwd;
                    //Reservoir R{};
                    //R.combine(R_BRDF.M * )
                    //R_NEE.combine(luminance(R_BSDF.path.P_hat), R_BSDF, next_pcg32_real<Real>(rng));
                    //R_BSDF
                    if (R_NEE.w_sum > 0)
                    {
                        R_NEE.W = R_NEE.w_sum / (R_NEE.M * luminance(R_NEE.pathTree.P_hat));
                    }
                    pathReservoirs[y * w + x] = R_NEE;
                }
            }
            reporter.update(1);
        }, Vector2i(num_tiles_x, num_tiles_y));


#if !RAW_PT_OUTPUT
    // Spatial Reuse
    for (int iter = 0; iter < scene.options.iterations; iter++)
    {
        parallel_for([&, iter](const Vector2i& tile)
            {
                // Use a different rng stream for each thread.
                pcg32_state rng = init_pcg32(1999999 + iter + tile[1] * num_tiles_x + tile[0]);
                int x0 = tile[0] * tile_size;
                int x1 = min(x0 + tile_size, w);
                int y0 = tile[1] * tile_size;
                int y1 = min(y0 + tile_size, h);

                for (int y = y0; y < y1; y++)
                {
                    for (int x = x0; x < x1; x++)
                    {
                        Reservoir R{};

                        std::vector<Reservoir> neighbors;
                        const Reservoir& Rq = pathReservoirs[y * w + x];

                        R.combine(luminance(Rq.pathTree.P_hat), Rq, next_pcg32_real<Real>(rng));
                        neighbors.push_back(Rq);
                        R.x = x;
                        R.y = y;

                        Real Z = 0;
                        if (Rq.pathTree.Vertices.size() > 2)
                        {
                            //
                            //while(neighbors.size() < 15)
                            for (int k = 0; k < 15; k++)
                            {
                                int xx = x + next_pcg32(rng) % 50 - 25;
                                int yy = y + next_pcg32(rng) % 50 - 25;
                                if (xx < 0 || xx >= w || yy < 0 || yy >= h)
                                {
                                    continue;
                                }
                                Reservoir Rn = pathReservoirs[yy * w + xx];
                                if (scene.options.shiftMapping == ShiftMappingType::RandomReplay)
                                {
                                    // ----------------------------Random replay-------------------------------
                                    ReplayableSampler sampler = Rn.pathTree.Sampler;
                                    //sampler.switch_stream(0);
                                    //sampler.replay();

                                    //Spectrum radiance = make_zero_spectrum();
                                    //Vector2 screen_pos((x + sampler.next_double()) / w,
                                    //    (y + sampler.next_double()) / h);

                                    if (Rn.pathTree.Vertices[1].Vertex.primitive_id != R.pathTree.Vertices[1].Vertex.primitive_id)
                                    {
                                        continue;
                                    }
                                    if (dot(Rn.pathTree.Vertices[1].Vertex.shading_frame.n, R.pathTree.Vertices[1].Vertex.shading_frame.n) < 0.98)
                                    {
                                        continue;
                                    }

                                   LightPathTree tree = restir_gen_camera_nee_pathtree(scene, x, y, w, h, sampler, scene.options.max_depth, true);
                                   /* PathTracingContext context{
                                        scene,
                                        sampler,
                                        sample_primary(scene.camera, screen_pos),
                                        RayDifferential{ Real(0), Real(0) },
                                        scene.camera.medium_id,
                                        0 };
                                    LightPath cameraPath = restir_gen_camera_subpath(scene, screen_pos, sampler, scene.options.max_depth);*/
                                    //if (tree.Vertices.size() < scene.options.max_depth)
                                    //{
                                    //    continue;
                                    //}
                                    Real J = 1;

                                    //for (int j = 1; j < std::min(R.pathTree.Vertices.size(), tree.Vertices.size()) - 1; j++)
                                    //{
                                    //    Vector3 win = normalize(R.pathTree.Vertices[j - 1].Vertex.position - R.pathTree.Vertices[j].Vertex.position);
                                    //    Vector3 wout = normalize(R.pathTree.Vertices[j + 1].Vertex.position - R.pathTree.Vertices[j].Vertex.position);
                                    //    const Material& mat = scene.materials[R.pathTree.Vertices[j].Vertex.material_id];
                                    //    Real pdf_bsdf1 = pdf_sample_bsdf(mat, win, wout, R.pathTree.Vertices[j].Vertex, scene.texture_pool);

                                    //    win = normalize(tree.Vertices[j - 1].Vertex.position - tree.Vertices[j].Vertex.position);
                                    //    wout = normalize(tree.Vertices[j + 1].Vertex.position - tree.Vertices[j].Vertex.position);
                                    //    const Material& mat2 = scene.materials[tree.Vertices[j].Vertex.material_id];
                                    //    Real pdf_bsdf2 = pdf_sample_bsdf(mat2, win, wout, tree.Vertices[j].Vertex, scene.texture_pool);
                                    //    J *= pdf_bsdf1 / pdf_bsdf2;
                                    //}
                                   

                                    Spectrum oldL = Rn.pathTree.P_hat;
                                    // restir_eval_nee_pathtree(tree, scene, false, scene.options.max_depth);
                                    Spectrum throughput;
                                    Spectrum L = restir_select_pathtree(tree, &throughput, scene, scene.options.max_depth);
                                    tree.L = L;
                                    tree.P_hat = throughput;
                                    Rn.pathTree = tree;
                                    R.combine(luminance(throughput) / J, Rn, next_pcg32_real<Real>(rng));
                                    neighbors.push_back(pathReservoirs[yy * w + xx]);
                                    {
                                        std::lock_guard<std::mutex> lock(img_mutex);
                                        Real d = (luminance(throughput - oldL));
                                        MSE += d * d;
                                        countShift++;
                                    }

                                    //sampler.switch_stream(1);
                                    //sampler.replay();
                                    //for (const auto& path : restir_select_subpath_nee(context, cameraPath, scene.options.max_depth))
                                    //{
                                    //    


                                    //    if (path.SampleMethod == SampleMethod::NEE)
                                    //    {
                                    //       
                                    //    }
                                    //}
                                }
                                else if (scene.options.shiftMapping == ShiftMappingType::VertexReconnect)
                                {

                                    // ------------------------Vertex Reconnect-----------------------------------
                                    ReplayableSampler sampler = Rn.pathTree.Sampler;
                                    sampler.replay();
                                    Vector2 screen_pos((x + sampler.next_double()) / w,
                                        (y + sampler.next_double()) / h);
                                    PathTracingContext context{
                                        scene,
                                        sampler,
                                        sample_primary(scene.camera, screen_pos),
                                        RayDifferential{ Real(0), Real(0) },
                                        scene.camera.medium_id,
                                        0 };
                                    LightPathTree cameraPath = R.pathTree;

                                    if (Rn.pathTree.Vertices[1].Vertex.shape_id != R.pathTree.Vertices[1].Vertex.shape_id)
                                    {
                                        continue;
                                    }
                                    if (dot(Rn.pathTree.Vertices[1].Vertex.shading_frame.n, R.pathTree.Vertices[1].Vertex.shading_frame.n) < 0.98)
                                    {
                                        continue;
                                    }

                                    if (Rn.pathTree.Vertices.size() < scene.options.max_depth)
                                    {
                                        continue;
                                    }


                                    Real J = 1;

                                    PathVertex xi = Rn.pathTree.Vertices[1].Vertex;
                                    PathVertex xi1 = Rn.pathTree.Vertices[2].Vertex;
                                    PathVertex yi = R.pathTree.Vertices[1].Vertex;

                                    Real A = std::abs(dot(xi1.geometric_normal, normalize(xi1.position - yi.position)) / dot(xi1.geometric_normal, normalize(xi1.position - xi.position)));
                                    Real B = length_squared(xi1.position - xi.position) / length_squared(xi1.position - yi.position);

                                    if (length(xi1.position - xi.position) < 10 || length(xi1.position - yi.position) < 10)
                                    {
                                        continue;
                                    }

                                    if (length(xi1.geometric_normal) > 0)
                                    {
                                        J /= A * B;
                                    }
                                    //cameraPath.Vertices[1].pdfFwd = 1;

                                    for (int s = 2; s < Rn.pathTree.Vertices.size(); s++)
                                    {
                                        if (s >= cameraPath.Vertices.size())
                                        {
                                            cameraPath.Vertices.push_back(Rn.pathTree.Vertices[s]);
                                        }
                                        else
                                        {
                                            cameraPath.Vertices[s] = Rn.pathTree.Vertices[s];
                                        }
                                    }

                                    while (cameraPath.Vertices.size() > Rn.pathTree.Vertices.size())
                                    {
                                        cameraPath.Vertices.pop_back();
                                    }


                                    Spectrum throughput = make_zero_spectrum();
                                    Spectrum L = make_zero_spectrum();

                                    if (restir_reeval_lightPathTree(scene, cameraPath))
                                    {
                                        L = restir_select_pathtree(cameraPath, &throughput, scene, scene.options.max_depth, false);
                                        // L = restir_eval_lightPath_no_nee(context, cameraPath, &throughput, scene.options.max_depth);
                                    }
                                    Spectrum oldL = Rn.pathTree.P_hat;
                                    if (isfinite(L) && !isnan(L))
                                    {
                                        Rn.pathTree = cameraPath;
                                        Rn.pathTree.L = L;
                                        Rn.pathTree.P_hat = throughput;

                                        // R.update(luminance(throughput) / J * Rn.W * Rn.M, Rn.path, next_pcg32_real<Real>(rng));
                                        R.combine(luminance(throughput) / J, Rn, next_pcg32_real<Real>(rng));
                                        neighbors.push_back(pathReservoirs[yy * w + xx]);
                                        {
                                            std::lock_guard<std::mutex> lock(img_mutex);
                                            Real d = (luminance(throughput - oldL));
                                            MSE += d * d;
                                            countShift++;
                                        }

                                    }
                                }
                                else if (scene.options.shiftMapping == ShiftMappingType::HalfVector)
                                {
                                    ReplayableSampler sampler = Rn.pathTree.Sampler;
                                    if (dot(Rn.pathTree.Vertices[1].Vertex.shading_frame.n, R.pathTree.Vertices[1].Vertex.shading_frame.n) < 0.98)
                                    {
                                        continue;
                                    }

                                    LightPathTree tree = restir_gen_camera_nee_pathtree(scene, x, y, w, h, sampler, scene.options.max_depth, true, &Rn.pathTree);
                                    if (tree.Vertices.size() < scene.options.max_depth)
                                    {
                                        continue;
                                    }
                                    Real J = 1;
                                    for (int i = 1; i < std::min(tree.Vertices.size(), Rn.pathTree.Vertices.size()) - 1; i++)
                                    {
                                        const PathVertex& prev = tree.Vertices[i - 1].Vertex;
                                        const PathVertex& cur = tree.Vertices[i].Vertex;
                                        const PathVertex& nxt = tree.Vertices[i + 1].Vertex;

                                        const PathVertex& prev2 = Rn.pathTree.Vertices[i - 1].Vertex;
                                        const PathVertex& cur2 = Rn.pathTree.Vertices[i].Vertex;
                                        const PathVertex& nxt2 = Rn.pathTree.Vertices[i + 1].Vertex;
                                        
                                        Vector3 wout = normalize(nxt.position - cur.position);
                                        Vector3 Hy = normalize(prev.position - cur.position) + wout;
                                        Hy = normalize(Hy);

                                        Vector3 wout2 = normalize(nxt2.position - cur2.position);
                                        Vector3 Hx = normalize(prev2.position - cur2.position) + wout2;
                                        Hx = normalize(Hx);

                                        J *= dot(Hy, wout) / dot(Hx, wout2);
                                    }
                                    Spectrum oldL = Rn.pathTree.P_hat;
                                    // restir_eval_nee_pathtree(tree, scene, false, scene.options.max_depth);
                                    Spectrum throughput;
                                    Spectrum L = restir_select_pathtree(tree, &throughput, scene, scene.options.max_depth);
                                    tree.L = L;
                                    tree.P_hat = throughput;
                                    Rn.pathTree = tree;
                                    R.combine(luminance(throughput) / J, Rn, next_pcg32_real<Real>(rng));

                                    neighbors.push_back(pathReservoirs[yy * w + xx]);
                                    {
                                        std::lock_guard<std::mutex> lock(img_mutex);
                                        Real d = (luminance(throughput - oldL));
                                        MSE += d * d;
                                        countShift++;
                                    }
                                }
                                else if (scene.options.shiftMapping == ShiftMappingType::Hybrid)
                                {
                                    // First, run half vector copy pass
                                    ReplayableSampler sampler = Rn.pathTree.Sampler;
                                    // Ignore points that are not in the same plane
                                    if (dot(Rn.pathTree.Vertices[1].Vertex.shading_frame.n, R.pathTree.Vertices[1].Vertex.shading_frame.n) < 0.98)
                                    {
                                        continue;
                                    }

                                    LightPathTree offsetPathTree = restir_gen_camera_nee_pathtree(scene, x, y, w, h, sampler, scene.options.max_depth, true, &Rn.pathTree);
                                    if (offsetPathTree.Vertices.size() < scene.options.max_depth)
                                    {
                                        continue;
                                    }


                                    Real J = 1;
                                    // Second, select the first pair of vertices that are both diffuse
                                    for (int i = 1; i < std::min(offsetPathTree.Vertices.size(), Rn.pathTree.Vertices.size()) - 1; i++)
                                    {
                                        const PathVertex& prev = offsetPathTree.Vertices[i - 1].Vertex;
                                        const PathVertex& cur = offsetPathTree.Vertices[i].Vertex;
                                        const PathVertex& nxt = offsetPathTree.Vertices[i + 1].Vertex;

                                        if (cur.componentType == ComponentType::Diffuse 
                                            && Rn.pathTree.Vertices[i].Vertex.componentType != ComponentType::Specular
                                            && Rn.pathTree.Vertices[i+1].Vertex.componentType != ComponentType::Specular)
                                        {
                                            const PathVertex& xi = Rn.pathTree.Vertices[i].Vertex;
                                            const PathVertex& xi1 = Rn.pathTree.Vertices[i + 1].Vertex;
                                            const PathVertex& yi = offsetPathTree.Vertices[i].Vertex;

                                            if (length(xi1.position - xi.position) < 10 || length(xi1.position - yi.position) < 10)
                                            {
                                                
                                            }
                                            else
                                            {
                                                Real A = std::abs(dot(xi1.geometric_normal, normalize(xi1.position - yi.position)) / dot(xi1.geometric_normal, normalize(xi1.position - xi.position)));
                                                Real B = length_squared(xi1.position - xi.position) / length_squared(xi1.position - yi.position);

                                                if (length(xi1.geometric_normal) > 0)
                                                {
                                                    J /= A * B;
                                                }


                                                for (int s = i + 1; s < Rn.pathTree.Vertices.size(); s++)
                                                {
                                                    if (s >= offsetPathTree.Vertices.size())
                                                    {
                                                        offsetPathTree.Vertices.push_back(Rn.pathTree.Vertices[s]);
                                                    }
                                                    else
                                                    {
                                                        offsetPathTree.Vertices[s] = Rn.pathTree.Vertices[s];
                                                    }
                                                }

                                                while (offsetPathTree.Vertices.size() > Rn.pathTree.Vertices.size())
                                                {
                                                    offsetPathTree.Vertices.pop_back();
                                                }
                                                break;
                                            }
                                        }

                                        const PathVertex& prev2 = Rn.pathTree.Vertices[i - 1].Vertex;
                                        const PathVertex& cur2 = Rn.pathTree.Vertices[i].Vertex;
                                        const PathVertex& nxt2 = Rn.pathTree.Vertices[i + 1].Vertex;

                                        Vector3 wout = normalize(nxt.position - cur.position);
                                        Vector3 Hy = normalize(prev.position - cur.position) + wout;
                                        Hy = normalize(Hy);

                                        Vector3 wout2 = normalize(nxt2.position - cur2.position);
                                        Vector3 Hx = normalize(prev2.position - cur2.position) + wout2;
                                        Hx = normalize(Hx);

                                        J *= dot(Hy, wout) / dot(Hx, wout2);
                                    }


                                    Spectrum throughput = make_zero_spectrum();
                                    Spectrum L = make_zero_spectrum();

                                    if (restir_reeval_lightPathTree(scene, offsetPathTree))
                                    {
                                        L = restir_select_pathtree(offsetPathTree, &throughput, scene, scene.options.max_depth, false);
                                        // L = restir_eval_lightPath_no_nee(context, cameraPath, &throughput, scene.options.max_depth);
                                    }
                                    Spectrum oldL = Rn.pathTree.P_hat;
                                    if (isfinite(L) && !isnan(L))
                                    {
                                        Rn.pathTree = offsetPathTree;
                                        Rn.pathTree.L = L;
                                        Rn.pathTree.P_hat = throughput;

                                        // R.update(luminance(throughput) / J * Rn.W * Rn.M, Rn.path, next_pcg32_real<Real>(rng));
                                        R.combine(luminance(throughput) / J, Rn, next_pcg32_real<Real>(rng));
                                        neighbors.push_back(pathReservoirs[yy * w + xx]);
                                        {
                                            std::lock_guard<std::mutex> lock(img_mutex);
                                            Real d = (luminance(throughput - oldL));
                                            MSE += d * d;
                                            countShift++;
                                        }

                                    }
                                }
                            }
                            
                        }



                        if (scene.options.shiftMapping == ShiftMappingType::RandomReplay)
                        {
                            for (const auto& Rn : neighbors)
                            {
                                ReplayableSampler sampler = R.pathTree.Sampler;
                                LightPathTree tree = restir_gen_camera_nee_pathtree(scene, Rn.x, Rn.y, w, h, sampler, scene.options.max_depth, true);
                                /* PathTracingContext context{
                                     scene,
                                     sampler,
                                     sample_primary(scene.camera, screen_pos),
                                     RayDifferential{ Real(0), Real(0) },
                                     scene.camera.medium_id,
                                     0 };
                                 LightPath cameraPath = restir_gen_camera_subpath(scene, screen_pos, sampler, scene.options.max_depth);*/
                                //if (tree.Vertices.size() < scene.options.max_depth)
                                //{
                                //    continue;
                                //}
                                //
                                // restir_eval_nee_pathtree(tree, scene, false, scene.options.max_depth);
                                Spectrum throughput;
                                Spectrum L = restir_select_pathtree(tree, &throughput, scene, scene.options.max_depth);
                                if (luminance(throughput) > 0)
                                {
                                    Z += Rn.M;
                                }
                                else
                                {
                                    if (luminance(Rn.pathTree.L) > 0)
                                    {
                                        printf("");
                                    }
                                }

                                //sampler.switch_stream(0);
                                //sampler.replay();

                                //Spectrum radiance = make_zero_spectrum();
                                //Vector2 screen_pos(( + sampler.next_double()) / w,
                                //    ( + sampler.next_double()) / h);


                                //PathTracingContext context{
                                //    scene,
                                //    sampler,
                                //    sample_primary(scene.camera, screen_pos),
                                //    RayDifferential{ Real(0), Real(0) },
                                //    scene.camera.medium_id,
                                //    0 };

                                //LightPath cameraPath = restir_gen_camera_subpath(scene, screen_pos, sampler, scene.options.max_depth);

                                //sampler.switch_stream(1);
                                //sampler.replay();
                                //for (const auto& path : restir_select_subpath_nee(context, cameraPath, scene.options.max_depth))
                                //{
                                //    if (path.SampleMethod == SampleMethod::NEE)
                                //    {
                                //        if (luminance(path.P_hat) > 0)
                                //        {
                                //            Z += Rn.M;
                                //        }
                                //    }
                                //}
                            }
                        }
                        else if (scene.options.shiftMapping == ShiftMappingType::VertexReconnect)
                        {
                            for (auto Rn : neighbors)
                            {
                                ReplayableSampler sampler = R.pathTree.Sampler;
                                sampler.replay();
                                Vector2 screen_pos((x + sampler.next_double()) / w,
                                    (y + sampler.next_double()) / h);
                                PathTracingContext context{
                                    scene,
                                    sampler,
                                    sample_primary(scene.camera, screen_pos),
                                    RayDifferential{ Real(0), Real(0) },
                                    scene.camera.medium_id,
                                    0 };
                                LightPathTree mainPath = Rn.pathTree;
                                for (int s = 2; s < R.pathTree.Vertices.size(); s++)
                                {
                                    if (s >= mainPath.Vertices.size())
                                    {
                                        mainPath.Vertices.push_back(R.pathTree.Vertices[s]);
                                    }
                                    else
                                    {
                                        mainPath.Vertices[s] = R.pathTree.Vertices[s];
                                    }
                                }

                                while (mainPath.Vertices.size() > R.pathTree.Vertices.size())
                                {
                                    mainPath.Vertices.pop_back();
                                }
                                if (restir_reeval_lightPathTree(scene, mainPath))
                                {
                                    //R.M += Rn.M;
                                    Spectrum throughput = make_zero_spectrum();
  
                                    Spectrum L = restir_select_pathtree(mainPath, &throughput, scene, scene.options.max_depth);
                                    if (luminance(throughput) > 0)
                                    {
                                        Z += Rn.M;
                                    }
                                    //else{
                                    //    if (Rn.path.Vertices.size() == scene.options.max_depth)
                                    //    {
                                    //        printf("");
                                    //    }
                                    //}
                                }
                            }
                        }
                        else if (scene.options.shiftMapping == ShiftMappingType::HalfVector)
                        {
                            for (const auto& Rn : neighbors)
                            {
                                ReplayableSampler sampler = R.pathTree.Sampler;
                                LightPathTree tree = restir_gen_camera_nee_pathtree(scene, Rn.x, Rn.y, w, h, sampler, scene.options.max_depth, true, &R.pathTree);
                                if (tree.Vertices.size() < scene.options.max_depth)
                                {
                                    continue;
                                }

                                // restir_eval_nee_pathtree(tree, scene, false, scene.options.max_depth);
                                Spectrum throughput;
                                Spectrum L = restir_select_pathtree(tree, &throughput, scene, scene.options.max_depth);
                                if (luminance(throughput) > 0)
                                {
                                    Z += Rn.M;
                                }
                                else
                                {
                                    if (luminance(Rn.pathTree.L) > 0)
                                    {
                                        printf("");
                                    }
                                }
                            }
                        }
                        else if (scene.options.shiftMapping == ShiftMappingType::Hybrid)
                        {
                            for (const auto& Rn : neighbors)
                            {
                                ReplayableSampler sampler = R.pathTree.Sampler;
                                LightPathTree offsetPathTree = restir_gen_camera_nee_pathtree(scene, Rn.x, Rn.y, w, h, sampler, scene.options.max_depth, true, &R.pathTree);
                                if (offsetPathTree.Vertices.size() < scene.options.max_depth)
                                {
                                    continue;
                                }

                                Real J = 1;
                                // Second, select the first pair of vertices that are both diffuse
                                for (int i = 1; i < std::min(offsetPathTree.Vertices.size(), R.pathTree.Vertices.size()) - 1; i++)
                                {
                                    const PathVertex& prev = offsetPathTree.Vertices[i - 1].Vertex;
                                    const PathVertex& cur = offsetPathTree.Vertices[i].Vertex;
                                    const PathVertex& nxt = offsetPathTree.Vertices[i + 1].Vertex;

                                    if (cur.componentType == ComponentType::Diffuse
                                        && R.pathTree.Vertices[i].Vertex.componentType != ComponentType::Specular
                                        && R.pathTree.Vertices[i + 1].Vertex.componentType != ComponentType::Specular)
                                    {
                                        const PathVertex& xi = R.pathTree.Vertices[i].Vertex;
                                        const PathVertex& xi1 = R.pathTree.Vertices[i + 1].Vertex;
                                        const PathVertex& yi = Rn.pathTree.Vertices[i].Vertex;
                                        if (length(xi1.position - xi.position) < 10 || length(xi1.position - yi.position) < 10)
                                        {

                                        }
                                        else
                                        {
                                            for (int s = i + 1; s < R.pathTree.Vertices.size(); s++)
                                            {
                                                if (s >= offsetPathTree.Vertices.size())
                                                {
                                                    offsetPathTree.Vertices.push_back(R.pathTree.Vertices[s]);
                                                }
                                                else
                                                {
                                                    offsetPathTree.Vertices[s] = R.pathTree.Vertices[s];
                                                }
                                            }

                                            while (offsetPathTree.Vertices.size() > R.pathTree.Vertices.size())
                                            {
                                                offsetPathTree.Vertices.pop_back();
                                            }
                                            break;
                                        }
                                    }
                                }


                                // restir_eval_nee_pathtree(tree, scene, false, scene.options.max_depth);

                                if (restir_reeval_lightPathTree(scene, offsetPathTree))
                                {
                                    Spectrum throughput;
                                    Spectrum L = restir_select_pathtree(offsetPathTree, &throughput, scene, scene.options.max_depth);
                                    if (luminance(throughput) > 0)
                                    {
                                        Z += Rn.M;
                                    }
                                    else
                                    {
                                        if (luminance(Rn.pathTree.L) > 0)
                                        {
                                            printf("");
                                        }
                                    }
                                }
                            }
                        }
                        if (R.w_sum > 0)
                        {
                            R.W = R.w_sum / (Z * luminance(R.pathTree.P_hat));
                            if (Z == 0)
                            {
                                printf("Error: Z is zero!\n");
                            }
                        }

                        pathReservoirsSwap[y * w + x] = R;

                        if (iter < scene.options.iterations - 1)
                        {
                            continue;
                        }

                        // Hacky: exclude NaNs in the rendering.
                        if (isfinite(R.pathTree.P_hat) && !isnan(R.pathTree.P_hat))
                        {
                            Spectrum d = make_zero_spectrum();
                            if (luminance(R.pathTree.P_hat) > 0)
                            {
                                d = R.pathTree.P_hat * R.W;
                                //printfVec3(d);
                                //printf(" | ");
                                //printfVec3(R.path.L);
                                //printf("\n");
                            }
                            if (isfinite(d) && !isnan(d))
                            {
                                img(x, y) += d;
                                img_cnt(x, y) += 1;
                            }
                            else
                            {
                                printf("Error: d is not finite");
                            }
                        }
                        else
                        {
                            printf("Error: L is not finite");
                        }
                    }
                }
                reporter.update(1);
            }, Vector2i(num_tiles_x, num_tiles_y));

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
                            pathReservoirs[y * w + x] = pathReservoirsSwap[y * w + x];
                        }
                    }
                }, Vector2i(num_tiles_x, num_tiles_y));
    }
#endif
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
                    if (img_cnt(x, y) > 0)
                    {
                        img(x, y) /= img_cnt(x, y);
                    }
                    else
                    {
                        img(x, y) = Vector3(0, 0, 0);
                    }
                }
            }
        }, Vector2i(num_tiles_x, num_tiles_y));
    reporter.done();

    printf("MSE: %lf\n", MSE / countShift);

#if DUMP_PT_PATHS
    exportTrainingData(path_train, path_val, h, w, spp);
#endif
    return img;
}