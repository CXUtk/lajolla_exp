#include "../microfacet.h"
#include <pcg.h>

Spectrum eval_op::operator()(const DisneyMetal &bsdf) const {
    if (dot(vertex.geometric_normal, dir_in) < 0 ||
            dot(vertex.geometric_normal, dir_out) < 0) {
        // No light below the surface
        return make_zero_spectrum();
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }    
    if (dot(frame.n, dir_out) < 0)
    {
        return make_zero_spectrum();
    }

    // Homework 1: implement this!
    Spectrum base_color = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real roughness = eval(
        bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real anisotropic = eval(
        bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real aspect = std::sqrt(1 - 0.9 * anisotropic);
    Real alphaX = std::max(0.0001, roughness * roughness / aspect);
    Real alphaY = std::max(0.0001, roughness * roughness * aspect);


    Vector3 H = normalize(dir_in + dir_out);
    Real NdotH = dot(frame.n, H);
    if (NdotH <= 0)
    {
        return make_zero_spectrum();
    }
    Real HdotV = std::max(Real(0), dot(H, dir_in));
    Real NdotV = std::max(Real(0), dot(frame.n, dir_in));
    Real NdotL = std::max(Real(0), dot(frame.n, dir_out));
    Vector3 wi = to_local(frame, dir_in), wo = to_local(frame, dir_out);

    Real D = D_GGX_aniso(to_local(frame, H), alphaX, alphaY);
    Real G = smith_masking_gtr2_aniso(wi, alphaX, alphaY) * smith_masking_gtr2_aniso(wo, alphaX, alphaY);
    return (D * G * schlick_fresnel(base_color, HdotV)) / (4 * NdotV);
}

Real pdf_sample_bsdf_op::operator()(const DisneyMetal &bsdf) const {
    if (dot(vertex.geometric_normal, dir_in) < 0)
    {
        return 0;
    }
    //if (dot(vertex.geometric_normal, dir_in) < 0 ||
    //        dot(vertex.geometric_normal, dir_out) < 0) {
    //    // No light below the surface
    //    return 0;
    //}

    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }

    // Homework 1: implement this!
    Real roughness = eval(
        bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real anisotropic = eval(
        bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real aspect = 1;//std::sqrt(1 - 0.9 * anisotropic);
    Real alphaX = std::max(0.0001, roughness * roughness / aspect);
    Real alphaY = std::max(0.0001, roughness * roughness * aspect);

    Vector3 H = normalize(dir_in + dir_out);
    Real NdotH = dot(frame.n, H);
    //if (NdotH <= 0)
    //{
    //    return 0;
    //}
    Real D = D_GGX_aniso(to_local(frame, H), alphaX, alphaX);

    Vector3 wi = to_local(frame, dir_in), wo = to_local(frame, dir_out);
    Real Gwin = smith_masking_gtr2_aniso(wi, alphaX, alphaX);
    Real NdotV = std::abs(dot(frame.n, dir_in));
    Real HdotV = std::max(dot(H, dir_in), 0.0);
    //return (D * Gwin * HdotV) / (NdotV);
    return D * std::max(NdotH, 0.0);
}

std::optional<BSDFSampleRecord>
sample_bsdf_op::operator()(const DisneyMetal& bsdf) const
{
    if (dot(vertex.geometric_normal, dir_in) < 0)
    {
        // No light below the surface
        return {};

    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0)
    {
        frame = -frame;
    }
    // Homework 1: implement this!
    Real roughness = eval(
        bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real anisotropic = eval(
        bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real aspect = 1;// std::sqrt(1 - 0.9 * anisotropic);
    Real alphaX = std::max(0.0001, roughness * roughness / aspect);
    Real alphaY = std::max(0.0001, roughness * roughness * aspect);

    //static pcg32_state rng;
    //static bool first = true;
    //if (first)
    //{
    //    init_pcg32((uint64_t)(rnd_param_uv.x * 1e18) * 143243432LL + (uint64_t)(rnd_param_uv.y * 1e18));
    //    first = false;
    //}


    Vector3 wout;
    Vector2 rnd_param = rnd_param_uv;
    Vector3 wi = to_local(frame, dir_in);
    Vector3 H = sample_DcosTheta(wi, alphaX, rnd_param);
    wout = reflect(-wi, H);

    return BSDFSampleRecord{
        to_world(frame, wout),
        Real(0) /* eta */, roughness /* roughness */, eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool), (roughness < 0.5 ? ComponentType::Specular : ComponentType::Diffuse) };
}

TextureSpectrum get_texture_op::operator()(const DisneyMetal &bsdf) const {
    return bsdf.base_color;
}
