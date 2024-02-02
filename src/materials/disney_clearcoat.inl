#include "../microfacet.h"

Spectrum eval_op::operator()(const DisneyClearcoat& bsdf) const
{
    if (dot(vertex.geometric_normal, dir_in) < 0 ||
        dot(vertex.geometric_normal, dir_out) < 0)
    {
        // No light below the surface
        return make_zero_spectrum();
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0)
    {
        frame = -frame;
    }

    // Homework 1: implement this!
    Real clearcoat_gloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real alphaG = (1 - clearcoat_gloss) * 0.1 + clearcoat_gloss * 1e-3;
    Real alphaG2 = alphaG * alphaG;
    constexpr Real R0 = 0.04;


    Vector3 H = normalize(dir_in + dir_out);
    Real NdotH = std::max(Real(0), dot(frame.n, H));
    Real HdotL = std::max(Real(0), dot(H, dir_out));
    Real NdotV = std::max(Real(0), dot(frame.n, dir_in));

    Real D = (alphaG2 - 1) / (c_PI * std::log(alphaG2) * (1 + (alphaG2 - 1) * NdotH * NdotH));
    Vector3 wi = to_local(frame, dir_in), wo = to_local(frame, dir_out);
    Real G = smith_masking_gtr2_aniso(wi, 0.25, 0.25) * smith_masking_gtr2_aniso(wo, 0.25, 0.25);
    Spectrum F = schlick_fresnel(Spectrum(R0, R0, R0), HdotL);
    return D * G * F / (4 * NdotV);
}

Real pdf_sample_bsdf_op::operator()(const DisneyClearcoat& bsdf) const
{
    if (dot(vertex.geometric_normal, dir_in) < 0)
    {
        // No light below the surface
        return 0;
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0)
    {
        frame = -frame;
    }
    // Homework 1: implement this!
    Vector3 H = normalize(dir_in + dir_out);
    Real NdotH = dot(frame.n, H);
    if (NdotH <= 0)
    {
        return 0;
    }
    Real clearcoat_gloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real alphaG = (1 - clearcoat_gloss) * 0.1 + clearcoat_gloss * 1e-3;
    Real alphaG2 = alphaG * alphaG;
    Real D = (alphaG2 - 1) / (c_PI * std::log(alphaG2) * (1 + (alphaG2 - 1) * NdotH * NdotH));

    Vector3 wi = to_local(frame, dir_in), wo = to_local(frame, dir_out);
    Real NdotV = std::abs(dot(frame.n, dir_in));
    return D * NdotH / (4 * std::abs(dot(H, dir_out)));
}

std::optional<BSDFSampleRecord>
        sample_bsdf_op::operator()(const DisneyClearcoat &bsdf) const {
    if (dot(vertex.geometric_normal, dir_in) < 0) {
        // No light below the surface
        return {};
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }

    // Homework 1: implement this!
    Real clearcoat_gloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real alpha = (1 - clearcoat_gloss) * 0.1 + clearcoat_gloss * 1e-3;
    Vector3 wi = to_local(frame, dir_in);
    Vector3 H = sample_DcosTheta(wi, alpha, rnd_param_uv);
    Vector3 wout = reflect(-wi, H);

    return BSDFSampleRecord{
        to_world(frame, wout),
        Real(0) /* eta */, Real(1) /* roughness */ };
}

TextureSpectrum get_texture_op::operator()(const DisneyClearcoat &bsdf) const {
    return make_constant_spectrum_texture(make_zero_spectrum());
}
