#pragma once
#include <transform.h>
#include "path_tracing_helper.h"
#include "lightpaths.h"

#define RAW_PT_OUTPUT 0
#define DUMP_PT_PATHS 0
bool gdpt_sample_vol_scattering_homogenized(int medium_id,
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

void gdpt_random_walk(LightPath& path, const Scene& scene, const Ray& initRay, const RayDifferential& rayDiff,
    ReplayableSampler& sampler,
    Spectrum beta, Real pdf,
    int maxDepth, TransportDirection direction, const LightPathTree* halfVectorCopyTree)
{
    if (maxDepth == 0) return;

    int bounces = 0;
    // Declare variables for forward and reverse probability densities
    Real pdfFwd = pdf, pdfRev = 0;
    Real pdfSampleNext = pdf;

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

            scatter = gdpt_sample_vol_scattering_homogenized(currentMedium, currentRay, sampler, tMax,
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
                    pdfSampleNext = pdfFwd;
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
            vertex = CreateSurface(vertex, beta, throughput, pdfFwd, path.Vertices.back(), currentMedium, pdfSampleNext);
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
            pdfSampleNext = pdfFwd;
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
                vertex.geometric_normal = Vector3(0, 0, 0);
                vertex.shading_frame.n = Vector3(0, 0, 0);
                vertex.position = currentRay.org + currentRay.dir * 99999.0;
                vertex.light_id = -1;
                vertex.beta = beta;
                vertex.throughput = throughput;
                vertex.pdfSampleNext = pdfSampleNext;
                vertex.componentType = ComponentType::None;
                path.Vertices.push_back(vertex);
            }
            break;
        }

    }
}

LightPath gdpt_gen_camera_subpath(const Scene& scene,
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
    gdpt_random_walk(path, scene, ray, ray_diff, sampler, cameraVertex.beta, pdf_dir, maxDepth - 1,
        TransportDirection::TO_LIGHT, halfVectorCopyTree);
    return path;
}

LightPathTree gdpt_gen_camera_nee_pathtree(const Scene& scene,
    int x, int y, int h, int w,
    ReplayableSampler& sampler,
    int maxDepth, bool replay = false, const LightPathTree* halfVectorCopyTree = nullptr)
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
    LightPath path = gdpt_gen_camera_subpath(scene, screen_pos, sampler, maxDepth, halfVectorCopyTree);

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
            light_vertex.pdfSampleNext = pdf_nee / G;

            curNode.L_Nee = L_nee;
            curNode.NeeVertex = light_vertex;
        }

        pathtree.Vertices.push_back(curNode);
    }
    pathtree.Sampler = sampler;
    return pathtree;
}

bool gdpt_reeval_lightPathTree(const Scene& scene, LightPathTree& pathTree, const LightPathTree& basePath, TransportDirection direction = TransportDirection::TO_LIGHT)
{
    Spectrum beta = make_const_spectrum(1);
    Spectrum throughput = make_const_spectrum(1);
    for (int i = 1; i < pathTree.Vertices.size(); i++)
    {
        PathVertex& start = pathTree.Vertices[i - 1].Vertex;
        PathVertex& end = pathTree.Vertices[i].Vertex;

        //Vector3 orig = start.position;
        //Ray shadow_ray{ orig, end.position - orig,
        //    get_shadow_epsilon(scene),
        //    (1 - get_shadow_epsilon(scene)) };
        //std::optional<PathVertex> vertex_ = intersect(scene, shadow_ray, RayDifferential{ Real(0), Real(0) });

        //if (vertex_)
        //{
        //    return false;
        //}
        const Material& mat = scene.materials[end.material_id];
        if (i < pathTree.Vertices.size() - 1)
        {
            PathVertex& next = pathTree.Vertices[i + 1].Vertex;
            const PathVertex& base_vertex = basePath.Vertices[i + 1].Vertex;
            // Let's do the hemispherical sampling next.
            Vector3 win = normalize(start.position - end.position);
            Vector3 wout = normalize(next.position - end.position);
            Real pdf_bsdf = pdf_sample_bsdf(mat, win, wout, end, scene.texture_pool, direction);
            Spectrum f_bsdf = eval(mat, win, wout, end, scene.texture_pool, direction);

            if (pdf_bsdf == 0)
            {
                return false;
            }

            beta *= f_bsdf / base_vertex.pdfSampleNext;
            throughput *= f_bsdf;

            next.beta = beta;
            next.throughput = throughput;
        }

        if (pathTree.Vertices[i].NeeVertex.vertexType != VertexType::None)
        {
            PathVertex& next = pathTree.Vertices[i].NeeVertex;
            Vector3 win = normalize(start.position - end.position);
            Vector3 wout = normalize(next.position - end.position);
            Spectrum f_bsdf = eval(mat, win, wout, end, scene.texture_pool);

            Real G = max(dot(-wout, next.geometric_normal), Real(0)) /
                distance_squared(next.position, end.position);

            const Light& light = scene.lights[next.light_id];
            PointAndNormal point_on_light = { next.position, next.geometric_normal };
            Real pdf_pos_t;
            Real pdf_dir;
            pdf_point_direction_on_light(light, point_on_light, end.position, scene, &pdf_pos_t, &pdf_dir);

            beta *= f_bsdf / pathTree.Vertices[i].NeeVertex.pdfSampleNext;// pdf_pos_t * G;
            throughput *= f_bsdf;
        }
    }
    return true;

}

void gdpt_eval_nee_pathtree(LightPathTree& tree, const Scene& scene, bool selectLength, const std::vector<Real>& jacobians, int selectedLength = -1)
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
                    Real J = jacobians.empty() ? 1 : jacobians[i];
                    L += vertex.beta * emission(vertex, dir_light, scene) * w * J;
                    P_hat += vertex.throughput * emission(vertex, dir_light, scene) * w * J;
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
                mat, win, wout, vertex, scene.texture_pool) * G;
            Real pdf_nee = nee_vertex.pdfFwd;
            Real w = (pdf_nee * pdf_nee) / (pdf_nee * pdf_nee + p_bsdf * p_bsdf);

            if (!selectLength || i == selectedLength - 2)
            {
                Real J = jacobians.empty() ? 1 : jacobians[i + 1];
                L += nee_vertex.beta * node.L_Nee * w * J;
                P_hat += nee_vertex.throughput * node.L_Nee * w * J;
            }
        }
    }

    tree.L = L;
    tree.P_hat = P_hat;
}

void gdpt_eval_nee_pathtree2(const LightPathTree& basePath, LightPathTree& offsetPath, const Scene& scene, bool selectLength, const std::vector<Real>& jacobians, int selectedLength = -1)
{
    Spectrum L = make_zero_spectrum();
    Spectrum P_hat = make_zero_spectrum();

    for (int i = 1; i < offsetPath.Vertices.size(); i++)
    {
        const LightPathTreeNode& node = offsetPath.Vertices[i];
        const PathVertex& vertex = node.Vertex;
        const PathVertex& vertex_base = basePath.Vertices[i].Vertex;
        const PathVertex& nee_vertex = node.NeeVertex;

        const PathVertex& prev_vertex = offsetPath.Vertices[i - 1].Vertex;

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
                    Real J = jacobians.empty() ? 1 : jacobians[i];
                    L += vertex.beta * emission(vertex, dir_light, scene) * w * J;
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
                mat, win, wout, vertex, scene.texture_pool) * G;
            Real pdf_nee = nee_vertex.pdfFwd;
            Real w = (pdf_nee * pdf_nee) / (pdf_nee * pdf_nee + p_bsdf * p_bsdf);

            if (!selectLength || i == selectedLength - 2)
            {
                Real J = jacobians.empty() ? 1 : jacobians[i + 1];
                L += nee_vertex.beta * node.L_Nee * w * J;
                P_hat += nee_vertex.throughput * node.L_Nee * w;
            }
        }
    }

    offsetPath.L = L;
    offsetPath.P_hat = P_hat;
}

bool tryVertexReconnect(const Scene& scene, const LightPathTree& basePath, LightPathTree& offsetPathTree, std::vector<Real>& jacobians)
{
    if (offsetPathTree.Vertices.size() < 3 || basePath.Vertices.size() < 3)
    {
        return false;
    }
    Real J = 1;

    PathVertex xi = basePath.Vertices[1].Vertex;
    PathVertex xi1 = basePath.Vertices[2].Vertex;
    PathVertex yi = offsetPathTree.Vertices[1].Vertex;

    Real A = std::abs(dot(xi1.geometric_normal, normalize(xi1.position - yi.position)) / dot(xi1.geometric_normal, normalize(xi1.position - xi.position)));
    Real B = length_squared(xi1.position - xi.position) / length_squared(xi1.position - yi.position);

    if (length(xi1.position - xi.position) < 10 || length(xi1.position - yi.position) < 10)
    {
        return false;
    }

    if (length(xi1.geometric_normal) > 0)
    {
        J /= A * B;
    }


    //cameraPath.Vertices[1].pdfFwd = 1;

    // Test if connectible
    Vector3 orig = yi.position;
    Ray shadow_ray{ orig, xi1.position - orig,
        get_shadow_epsilon(scene),
        (1 - get_shadow_epsilon(scene)) };
    std::optional<PathVertex> vertex_ = intersect(scene, shadow_ray, RayDifferential{ Real(0), Real(0) });

    if (vertex_)
    {
        return false;
    }


    for (int s = 2; s < basePath.Vertices.size(); s++)
    {
        if (s >= offsetPathTree.Vertices.size())
        {
            offsetPathTree.Vertices.push_back(basePath.Vertices[s]);
        }
        else
        {
            offsetPathTree.Vertices[s] = basePath.Vertices[s];
        }
    }

    while (offsetPathTree.Vertices.size() > basePath.Vertices.size())
    {
        offsetPathTree.Vertices.pop_back();
    }

    jacobians.clear();
    for (int i = 0; i < 2; i++)
    {
        jacobians.push_back(1);
    }
    for (int i = 2; i < scene.options.max_depth; i++)
    {
        jacobians.push_back(1 / J);
    }
    return true;
}

LightPathTree gdpt_eval_shift_mapping_path(const Scene& scene,
    int x, int y,
    ReplayableSampler& sampler, const LightPathTree& basePath)
{
    int w = scene.camera.width, h = scene.camera.height;
    LightPathTree offsetPathTree = gdpt_gen_camera_nee_pathtree(scene, x, y, w, h, sampler, scene.options.max_depth, true, nullptr);

    std::vector<Real> jacobian;
    if (scene.options.shiftMapping == ShiftMappingType::VertexReconnect)
    {
        LightPathTree offsetPathTreeCopy = offsetPathTree;
        if (tryVertexReconnect(scene, basePath, offsetPathTreeCopy, jacobian))
        {
            offsetPathTree = offsetPathTreeCopy;
            gdpt_reeval_lightPathTree(scene, offsetPathTree, basePath);
        }
        gdpt_eval_nee_pathtree2(basePath, offsetPathTree, scene, false, jacobian);
    }
    else if (scene.options.shiftMapping == ShiftMappingType::RandomReplay)
    {
        gdpt_eval_nee_pathtree(offsetPathTree, scene, false, jacobian);
    }

    //Real J = 1;
    //for (int i = 1; i < std::min(tree.Vertices.size(), Rn.pathTree.Vertices.size()) - 1; i++)
    //{
    //    const PathVertex& prev = tree.Vertices[i - 1].Vertex;
    //    const PathVertex& cur = tree.Vertices[i].Vertex;
    //    const PathVertex& nxt = tree.Vertices[i + 1].Vertex;

    //    const PathVertex& prev2 = Rn.pathTree.Vertices[i - 1].Vertex;
    //    const PathVertex& cur2 = Rn.pathTree.Vertices[i].Vertex;
    //    const PathVertex& nxt2 = Rn.pathTree.Vertices[i + 1].Vertex;

    //    Vector3 wout = normalize(nxt.position - cur.position);
    //    Vector3 Hy = normalize(prev.position - cur.position) + wout;
    //    Hy = normalize(Hy);

    //    Vector3 wout2 = normalize(nxt2.position - cur2.position);
    //    Vector3 Hx = normalize(prev2.position - cur2.position) + wout2;
    //    Hx = normalize(Hx);

    //    J *= dot(Hy, wout) / dot(Hx, wout2);
    //}

    Spectrum throughput;
    Spectrum L = offsetPathTree.L;
    if (isnan(L) || !isfinite(L))
    {
        printf("%lf %lf %lf\n", L.x, L.y, L.z);
    }
    return offsetPathTree;
}


Image3 do_gdpt(const Scene& scene)
{
    Image3 gt_img = imread3("box_glossy_1024b.exr");


    int w = scene.camera.width, h = scene.camera.height;
    Image3 img(w, h);
    Image3 img_copy(w, h);
    Image3 img_Gx(w, h);
    Image3 img_Gy(w, h);
    Image3 img_GxGT(w, h);
    Image3 img_GyGT(w, h);
    std::mutex img_mutex;

    constexpr int tile_size = 16;
    int num_tiles_x = (w + tile_size - 1) / tile_size;
    int num_tiles_y = (h + tile_size - 1) / tile_size;

    Real MSEGT = 0;

    parallel_for([&](const Vector2i& tile)
        {
            int x0 = tile[0] * tile_size;
            int x1 = min(x0 + tile_size, w);
            int y0 = tile[1] * tile_size;
            int y1 = min(y0 + tile_size, h);

            for (int y = y0; y < y1; y++)
            {
                for (int x = x0; x < x1; x++)
                {
                    Spectrum Gx_l = make_zero_spectrum();
                    Spectrum Gx_r = make_zero_spectrum();
                    Spectrum Gy_u = make_zero_spectrum();
                    Spectrum Gy_d = make_zero_spectrum();
                    if (x - 1 >= 0)
                    {
                        Gx_l += 0.5 * (gt_img(x, y) - gt_img(x - 1, y));
                    }
                    if (x + 1 < w)
                    {
                        Gx_r += 0.5 * (gt_img(x + 1, y) - gt_img(x, y));
                    }
                    if (y - 1 >= 0)
                    {
                        Gy_u += 0.5 * (gt_img(x, y) - gt_img(x, y - 1));
                    }
                    if (y + 1 < h)
                    {
                        Gy_d += 0.5 * (gt_img(x, y + 1) - gt_img(x, y));
                    }

                    {
                        std::lock_guard<std::mutex> guard(img_mutex);
                        if (x - 1 >= 0)
                        {
                            img_GxGT(x - 1, y) = Gx_l;
                        }
                        if (x + 1 < w)
                        {
                            img_GxGT(x, y) = Gx_r;
                        }
                        if (y - 1 >= 0)
                        {
                            img_GyGT(x, y - 1) = Gy_u;
                        }
                        if (y + 1 < h)
                        {
                            img_GyGT(x, y) = Gy_d;
                        }
                    }
                }
            }
        }, Vector2i(num_tiles_x, num_tiles_y));


    Real MSE = 0;
    uint64_t countShift = 0;

    ProgressReporter reporter(num_tiles_x * num_tiles_y);
    parallel_for([&](const Vector2i& tile)
        {
            // Use a different rng stream for each thread.
            pcg32_state rng = init_pcg32(tile[1] * num_tiles_x + tile[0]);
            int x0 = tile[0] * tile_size;
            int x1 = min(x0 + tile_size, w);
            int y0 = tile[1] * tile_size;
            int y1 = min(y0 + tile_size, h);

            ReplayableSampler sampler(rng);
            for (int y = y0; y < y1; y++)
            {
                for (int x = x0; x < x1; x++)
                {
                    Spectrum radiance = make_zero_spectrum();
                    int spp = scene.options.samples_per_pixel;
                    Spectrum Gx_l = make_zero_spectrum();
                    Spectrum Gx_r = make_zero_spectrum();
                    Spectrum Gy_u = make_zero_spectrum();
                    Spectrum Gy_d = make_zero_spectrum();
                    for (int s = 0; s < spp; s++)
                    {
                        LightPathTree cameraPathTree = gdpt_gen_camera_nee_pathtree(scene, x, y, w, h, sampler, scene.options.max_depth);
                        gdpt_eval_nee_pathtree(cameraPathTree, scene,false, std::vector<Real>());
                        Spectrum throughput;
                        Spectrum L = cameraPathTree.L;

                        ReplayableSampler samplerCopy = sampler;
                        if (isfinite(L))
                        {
                            // Hacky: exclude NaNs in the rendering.
                            radiance += L;
                        }
                        else
                        {
                            printf("Error: L is not finite");
                        }

                        if (x - 1 >= 0)
                        {
                            LightPathTree offsetPath = gdpt_eval_shift_mapping_path(scene, x - 1, y, samplerCopy, cameraPathTree);
                            Spectrum oldL = cameraPathTree.L;
                            {
                                std::lock_guard<std::mutex> lock(img_mutex);
                                Real d = (luminance(offsetPath.L - oldL));
                                MSE += d * d;
                                countShift++;
                            }
                            Gx_l += 0.5 * (L - offsetPath.L);
                        }
                        if (x + 1 < w)
                        {
                            LightPathTree offsetPath = gdpt_eval_shift_mapping_path(scene, x + 1, y, samplerCopy, cameraPathTree);
                            Spectrum oldL = cameraPathTree.L;
                            {
                                std::lock_guard<std::mutex> lock(img_mutex);
                                Real d = (luminance(offsetPath.L - oldL));
                                MSE += d * d;
                                countShift++;
                            }
                            Gx_r += 0.5 * (offsetPath.L - L);
                        }
                        if (y - 1 >= 0)
                        {
                            LightPathTree offsetPath = gdpt_eval_shift_mapping_path(scene, x, y - 1, samplerCopy, cameraPathTree);
                            Spectrum oldL = cameraPathTree.L;
                            {
                                std::lock_guard<std::mutex> lock(img_mutex);
                                Real d = (luminance(offsetPath.L - oldL));
                                MSE += d * d;
                                countShift++;
                            }
                            Gy_u += 0.5 * (L - offsetPath.L);
                        }
                        if (y + 1 < h)
                        {
                            LightPathTree offsetPath = gdpt_eval_shift_mapping_path(scene, x, y + 1, samplerCopy, cameraPathTree);
                            Spectrum oldL = cameraPathTree.L;
                            {
                                std::lock_guard<std::mutex> lock(img_mutex);
                                Real d = (luminance(offsetPath.L - oldL));
                                MSE += d * d;
                                countShift++;
                            }
                            Gy_d += 0.5 * (offsetPath.L - L);
                        }
                    }
                    {
                        std::lock_guard<std::mutex> guard(img_mutex);
                        img(x, y) = radiance;
                        if (x - 1 >= 0)
                        {
                            img_Gx(x - 1, y) += Gx_l;
                        }
                        if (x + 1 < w)
                        {
                            img_Gx(x, y) += Gx_r;
                        }
                        if (y - 1 >= 0)
                        {
                            img_Gy(x, y - 1) += Gy_u;
                        }
                        if (y + 1 < h)
                        {
                            img_Gy(x, y) += Gy_d;
                        }
                    }
                }
            }
            reporter.update(1);
        }, Vector2i(num_tiles_x, num_tiles_y));

    int spp = scene.options.samples_per_pixel;
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            img(x, y) = (img(x, y) / (Real)spp);
            img_Gx(x, y) = (img_Gx(x, y) / (Real)spp);
            img_Gy(x, y) = (img_Gy(x, y) / (Real)spp);

            MSEGT += luminance((img_Gx(x, y) - img_GxGT(x, y)) * (img_Gx(x, y) - img_GxGT(x, y)));
            MSEGT += luminance((img_Gy(x, y) - img_GyGT(x, y)) * (img_Gy(x, y) - img_GyGT(x, y)));
        }
    }

    MSEGT /= w * h;

    for (int I = 0; I < 100; I++)
    {
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                Spectrum v = img(x, y);
                if (x - 1 >= 0)
                {
                    v += img(x - 1, y) + img_GxGT(x - 1, y);
                }
                if (x + 1 < w)
                {
                    v += img(x + 1, y) + img_GxGT(x, y);
                }
                if (y - 1 >= 0)
                {
                    v += img(x, y - 1) + img_GyGT(x, y - 1);
                }
                if (y + 1 < h)
                {
                    v += img(x, y + 1) + img_GyGT(x, y);
                }
                img_copy(x, y) = v * 0.25;
            }
        }
    }
    reporter.done();
    printf("MSE: %lf\n", MSE / countShift);
    printf("MSE to GT: %lf\n", MSEGT);
#if RAW_PT_OUTPUT
    return img;
#else 
    return img_copy;
#endif
}
//
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