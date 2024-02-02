#include "../microfacet.h"

inline Real square(Real x) { return x * x; }

Spectrum eval_op::operator()(const DisneyGlass &bsdf) const {
    bool reflect = dot(vertex.geometric_normal, dir_in) *
                   dot(vertex.geometric_normal, dir_out) > 0;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    // Homework 1: implement this!

    Spectrum base_color = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real roughness = eval(
        bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real anisotropic = eval(
        bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;

    Vector3 H;
    if (reflect)
    {
        H = normalize(dir_in + dir_out);
    }
    else
    {
        // "Generalized half-vector" from Walter et al.
        // See "Microfacet Models for Refraction through Rough Surfaces"
        H = normalize(dir_in + dir_out * eta);

        if (length_squared(dir_in + dir_out * eta) < Real(1e-3))
        {
            H = normalize(cross(cross(dir_out, frame.n), dir_out));
        }
    }

    // Flip half-vector if it's below surface
    if (dot(H, frame.n) < 0)
    {
        H = -H;
    }
    Real aspect = std::sqrt(1 - 0.9 * anisotropic);
    Real alphaX = std::max(0.0001, roughness * roughness / aspect);
    Real alphaY = std::max(0.0001, roughness * roughness * aspect);

    Real HdotV = dot(H, dir_in);
    Real HdotL = dot(H, dir_out);
    // We should use fabs because reflection could happen in the back face
    Real NdotH = std::fabs(dot(frame.n, H));
    Real NdotV = std::fabs(dot(frame.n, dir_in));

    
    Real F =  fresnel_dielectric(std::abs(HdotV), eta);
    //Real F = schlick_fresnel(fresnel_R0(eta), HdotV);
    //if (eta < 1)
    //{
    //    F = schlick_fresnel(fresnel_R0(eta), HdotV, eta);
    //}
    Vector3 wi = to_local(frame, dir_in), wo = to_local(frame, dir_out);
    Real D = D_GGX_aniso(to_local(frame, H), alphaX, alphaY);
    Real G = smith_masking_gtr2_aniso(wi, alphaX, alphaY) * smith_masking_gtr2_aniso(wo, alphaX, alphaY);
    if (reflect)
    {
        return base_color * (D * G * F) / (4 * NdotV);
    }
    else
    {
        return sqrt(base_color) * (D * G * (1 - F) * std::abs(HdotV * HdotL)) / (NdotV * square(HdotV + eta * HdotL));
    }
}

Real pdf_sample_bsdf_op::operator()(const DisneyGlass &bsdf) const {
    bool reflect = dot(vertex.geometric_normal, dir_in) *
                   dot(vertex.geometric_normal, dir_out) > 0;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    // Homework 1: implement this!
    // If we are going into the surface, then we use normal eta
    // (internal/external), otherwise we use external/internal.
    Real eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;
    assert(eta > 0);

    Vector3 half_vector;
    if (reflect)
    {
        half_vector = normalize(dir_in + dir_out);
    }
    else
    {
        // "Generalized half-vector" from Walter et al.
        // See "Microfacet Models for Refraction through Rough Surfaces"
        half_vector = normalize(dir_in + dir_out * eta);
    }

    // Flip half-vector if it's below surface
    if (dot(half_vector, frame.n) < 0)
    {
        half_vector = -half_vector;
    }

    Real roughness = eval(
        bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real anisotropic = eval(
        bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    // Clamp roughness to avoid numerical issues.
    Real aspect = std::sqrt(1 - 0.9 * anisotropic);
    Real alphaX = std::max(0.0001, roughness * roughness / aspect);
    Real alphaY = std::max(0.0001, roughness * roughness * aspect);

    // We sample the visible normals, also we use F to determine
    // whether to sample reflection or refraction
    // so PDF ~ F * D * G_in for reflection, PDF ~ (1 - F) * D * G_in for refraction.
    Real h_dot_in = dot(half_vector, dir_in);
    Real F = fresnel_dielectric(h_dot_in, eta);
    //Real F = schlick_fresnel(fresnel_R0(eta), h_dot_in);
    //if (eta < 1)
    //{
    //    F = schlick_fresnel(fresnel_R0(eta), h_dot_in, eta);
    //}
    Real D = D_GGX_aniso(to_local(frame, half_vector), alphaX, alphaY);
    Real G_in = smith_masking_gtr2_aniso(to_local(frame, dir_in), alphaX, alphaY);
    if (reflect)
    {
        return (F * D * G_in) / (4 * fabs(dot(frame.n, dir_in)));
    }
    else
    {
        Real h_dot_out = dot(half_vector, dir_out);
        Real sqrt_denom = h_dot_in + eta * h_dot_out;
        Real dh_dout = eta * eta * h_dot_out / (sqrt_denom * sqrt_denom);
        return (1 - F) * D * G_in * fabs(dh_dout * h_dot_in / dot(frame.n, dir_in));
    }
}

std::optional<BSDFSampleRecord>
        sample_bsdf_op::operator()(const DisneyGlass &bsdf) const {
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    // Homework 1: implement this!

    Real roughness = eval(
        bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    // Clamp roughness to avoid numerical issues.
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    Real anisotropic = eval(
        bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real aspect = std::sqrt(1 - 0.9 * anisotropic);
    Real alphaX = std::max(0.0001, roughness * roughness / aspect);
    Real alphaY = std::max(0.0001, roughness * roughness * aspect);

    // Sample a micro normal and transform it to world space -- this is our half-vector.
    Real alpha = roughness * roughness;
    Real eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;
    Vector3 local_dir_in = to_local(frame, dir_in);
    Vector3 local_micro_normal =
        sample_visible_normals_aniso(local_dir_in, alphaX, alphaY, rnd_param_uv);

    Vector3 half_vector = to_world(frame, local_micro_normal);
    // Flip half-vector if it's below surface
    if (dot(half_vector, frame.n) < 0)
    {
        half_vector = -half_vector;
    }

    // Now we need to decide whether to reflect or refract.
    // We do this using the Fresnel term.
    Real h_dot_in = dot(half_vector, dir_in);
    Real F = fresnel_dielectric(h_dot_in, eta);
    //Real F = schlick_fresnel(fresnel_R0(eta), h_dot_in);
    //if (eta < 1)
    //{
    //    F = schlick_fresnel(fresnel_R0(eta), h_dot_in, eta);
    //}

    if (rnd_param_w < F)
    {
        // Reflection
        Vector3 reflected = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
        // set eta to 0 since we are not transmitting
        return BSDFSampleRecord{ reflected, Real(0) /* eta */, roughness };
    }
    else
    {
        // Refraction
        // https://en.wikipedia.org/wiki/Snell%27s_law#Vector_form
        // (note that our eta is eta2 / eta1, and l = -dir_in)
        Real h_dot_out_sq = 1 - (1 - h_dot_in * h_dot_in) / (eta * eta);
        if (h_dot_out_sq <= 0)
        {
            // Total internal reflection
            // This shouldn't really happen, as F will be 1 in this case.
            return {};
        }
        // flip half_vector if needed
        if (h_dot_in < 0)
        {
            half_vector = -half_vector;
        }
        Real h_dot_out = sqrt(h_dot_out_sq);
        Vector3 refracted = -dir_in / eta + (fabs(h_dot_in) / eta - h_dot_out) * half_vector;
        return BSDFSampleRecord{ refracted, eta, roughness };
    }
}

TextureSpectrum get_texture_op::operator()(const DisneyGlass &bsdf) const {
    return bsdf.base_color;
}
