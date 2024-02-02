#include "../microfacet.h"
#include "../table_dist.h"

Spectrum eval_op::operator()(const DisneyBSDF& bsdf) const
{
    bool reflect = dot(vertex.geometric_normal, dir_in) *
        dot(vertex.geometric_normal, dir_out) > 0;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0)
    {
        frame = -frame;
    }
    // Homework 1: implement this!
    Spectrum base_color = eval(
        bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real roughness = eval(
        bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular_transmission = eval(
        bsdf.specular_transmission, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real metallic = eval(
        bsdf.metallic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat = eval(
        bsdf.clearcoat, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real sheen = eval(
        bsdf.sheen, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular = eval(
        bsdf.specular, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular_tint = eval(
        bsdf.specular_tint, vertex.uv, vertex.uv_screen_size, texture_pool);

    Spectrum f_Diffuse = make_zero_spectrum();
    DisneyDiffuse diffuse_bsdf{};
    diffuse_bsdf.base_color = bsdf.base_color;
    diffuse_bsdf.roughness = bsdf.roughness;
    diffuse_bsdf.subsurface = bsdf.subsurface;

    Spectrum f_Clearcoat = make_zero_spectrum();
    DisneyClearcoat clearcoat_bsdf{};
    clearcoat_bsdf.clearcoat_gloss = bsdf.clearcoat_gloss;

    Spectrum f_Sheen = make_zero_spectrum();
    DisneySheen sheen_bsdf{};
    sheen_bsdf.base_color = bsdf.base_color;
    sheen_bsdf.sheen_tint = bsdf.sheen_tint;

    Spectrum f_Glass = make_zero_spectrum();
    DisneyGlass glass_bsdf{};
    glass_bsdf.base_color = bsdf.base_color;
    glass_bsdf.roughness = bsdf.roughness;
    glass_bsdf.anisotropic = bsdf.anisotropic;
    glass_bsdf.eta = bsdf.eta;

    Spectrum f_Metal = make_zero_spectrum();

    // Metal is calculated seperately
    Real anisotropic = eval(
        bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real aspect = std::sqrt(1 - 0.9 * anisotropic);
    Real alphaX = std::max(0.0001, roughness * roughness / aspect);
    Real alphaY = std::max(0.0001, roughness * roughness * aspect);
    Vector3 H = normalize(dir_in + dir_out);

    Real NdotH = dot(frame.n, H);

    Real HdotL = std::abs(dot(H, dir_out));
    Real NdotV = std::abs(dot(frame.n, dir_in));
    Real D = D_GGX_aniso(to_local(frame, H), alphaX, alphaY);
    Vector3 wi = to_local(frame, dir_in), wo = to_local(frame, dir_out);
    Real G = smith_masking_gtr2_aniso(wi, alphaX, alphaY) * smith_masking_gtr2_aniso(wo, alphaX, alphaY);

    Real l = luminance(base_color);
    Spectrum tint = Spectrum(Real(1), Real(1), Real(1));
    if (l > 0)
    {
        tint = base_color / l;
    }
    Real eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;
    Spectrum Ks = (1 - specular_tint) + specular_tint * tint;
    Spectrum C0 = specular * fresnel_R0(eta) * (1 - metallic) * Ks + metallic * base_color;
    Spectrum F = schlick_fresnel(C0, HdotL);


    f_Glass = this->operator()(glass_bsdf);

    if (dot(vertex.geometric_normal, dir_in) > 0)
    {
        //if (dot(frame.n, dir_out) * dot(vertex.geometric_normal, dir_out) < 0)
        //{
        //    printf("NdotO = %lf, GdotO = %lf, F = %lf\n", dot(frame.n, dir_out), dot(vertex.geometric_normal, dir_out),
        //        D * G * F.x / (4 * NdotV));
        //}
        if (dot(vertex.geometric_normal, dir_out) > 0)
        {
            f_Metal = (D * G * F) / (4 * NdotV);
            f_Diffuse = this->operator()(diffuse_bsdf);
            f_Clearcoat = this->operator()(clearcoat_bsdf);
            f_Sheen = this->operator()(sheen_bsdf);
        }

        //if (wo.z < 0)
        //{
        //    printf("Going down: %lf\n", f_Metal.x);
        //}
    }


    return (1 - specular_transmission) * (1 - metallic) * f_Diffuse
        + (1 - metallic) * sheen * f_Sheen
        + 0.25 * clearcoat * f_Clearcoat
        + (1 - metallic) * specular_transmission * f_Glass
        + (1 - specular_transmission * (1 - metallic)) * f_Metal;
}

Real pdf_sample_bsdf_op::operator()(const DisneyBSDF &bsdf) const {
    bool reflect = dot(vertex.geometric_normal, dir_in) *
                   dot(vertex.geometric_normal, dir_out) > 0;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    // Homework 1: implement this!

    Real roughness = eval(
        bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular_transmission = eval(
        bsdf.specular_transmission, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real metallic = eval(
        bsdf.metallic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat = eval(
        bsdf.clearcoat, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real sheen = eval(
        bsdf.sheen, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular = eval(
        bsdf.specular, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular_tint = eval(
        bsdf.specular_tint, vertex.uv, vertex.uv_screen_size, texture_pool);

    DisneyDiffuse diffuse{};
    diffuse.base_color = bsdf.base_color;
    diffuse.roughness = bsdf.roughness;
    diffuse.subsurface = bsdf.subsurface;

    DisneyMetal metal_bsdf{};
    metal_bsdf.base_color = bsdf.base_color;
    metal_bsdf.roughness = bsdf.roughness;
    metal_bsdf.anisotropic = bsdf.anisotropic;

    DisneyGlass glass_bsdf{};
    glass_bsdf.base_color = bsdf.base_color;
    glass_bsdf.roughness = bsdf.roughness;
    glass_bsdf.anisotropic = bsdf.anisotropic;
    glass_bsdf.eta = bsdf.eta;

    DisneyClearcoat clearcoat_bsdf{};
    clearcoat_bsdf.clearcoat_gloss = bsdf.clearcoat_gloss;

    Real weights[4] = {
        (1 - metallic) * (1 - specular_transmission),
        (1 - specular_transmission * (1 - metallic)),
        (1 - metallic) * specular_transmission,
        0.25 * clearcoat
    };

    Real wSum = weights[0] + weights[1] + weights[2] + weights[3];
    for (int i = 0; i < 4; i++)
    {
        weights[i] /= wSum;
    }

    Real pdfs[4] = {
        this->operator()(diffuse),
        this->operator()(metal_bsdf),
        this->operator()(glass_bsdf),
        this->operator()(clearcoat_bsdf)
    };

    if (dot(vertex.geometric_normal, dir_in) <= 0)
    {
        return pdfs[2];
    }

    Real pdfTotal = 0;
    Real pdfWeight = 0;
    for (int i = 0; i < 4; i++)
    {
        pdfTotal += pdfs[i] * weights[i];
    }
    // printf("Pdf: %lf\n", pdfTotal);


    return pdfTotal;
}

std::optional<BSDFSampleRecord>
        sample_bsdf_op::operator()(const DisneyBSDF &bsdf) const {
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    // Homework 1: implement this!
    Real roughness = eval(
        bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular_transmission = eval(
        bsdf.specular_transmission, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real metallic = eval(
        bsdf.metallic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat = eval(
        bsdf.clearcoat, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real sheen = eval(
        bsdf.sheen, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular = eval(
        bsdf.specular, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular_tint = eval(
        bsdf.specular_tint, vertex.uv, vertex.uv_screen_size, texture_pool);

    Real weights[4] = {
            (1 - metallic) * (1 - specular_transmission),
            (1 - specular_transmission * (1 - metallic)),
            (1 - metallic) * specular_transmission,
            0.25 * clearcoat
    };

    // printf("Weights %lf %lf %lf %lf\n", weights[0], weights[1], weights[2], weights[3]);

    Real weightPrefix[5];
    weightPrefix[0] = 0;
    for (int i = 1; i < 5; i++)
    {
        weightPrefix[i] = weightPrefix[i - 1] + weights[i - 1];
    }
    for (int i = 0; i < 5; i++)
    {
        weightPrefix[i] /= weightPrefix[4];

    }

    int bsdf_selected = -1;
    Real rescaleW = 0; 
    for (int i = 0; i < 5; i++)
    {
        if (rnd_param_w < weightPrefix[i])
        {
            bsdf_selected = i - 1;
            rescaleW = (rnd_param_w - weightPrefix[i - 1]) / (weightPrefix[i] - weightPrefix[i - 1]);
            break;
        }
    }


    DisneyDiffuse diffuse{};
    diffuse.base_color = bsdf.base_color;
    diffuse.roughness = bsdf.roughness;
    diffuse.subsurface = bsdf.subsurface;

    DisneyMetal metal_bsdf{};
    metal_bsdf.base_color = bsdf.base_color;
    metal_bsdf.roughness = bsdf.roughness;
    metal_bsdf.anisotropic = bsdf.anisotropic;

    DisneyGlass glass_bsdf{};
    glass_bsdf.base_color = bsdf.base_color;
    glass_bsdf.roughness = bsdf.roughness;
    glass_bsdf.anisotropic = bsdf.anisotropic;
    glass_bsdf.eta = bsdf.eta;

    DisneyClearcoat clearcoat_bsdf{};
    clearcoat_bsdf.clearcoat_gloss = bsdf.clearcoat_gloss;

    //TableDist1D bsdf_dist = make_table_dist_1d({ weights[0], weights[1] , weights[2], weights[3] });
    //int bsdf_seleccted = sample(bsdf_dist, rnd_param_w);

    if (dot(vertex.geometric_normal, dir_in) <= 0)
    {
        return this->operator()(glass_bsdf);
    }

    switch (bsdf_selected)
    {
    case 0:
        return this->operator()(diffuse);
    case 1:
        return this->operator()(metal_bsdf);
    case 2: {
        Real* rp = const_cast<Real*>(&rnd_param_w);
        *rp = rescaleW;
        return this->operator()(glass_bsdf);
    }
    case 3:
        return this->operator()(clearcoat_bsdf);
    default:
        printf("ERROR on BSDF\n");
        return this->operator()(diffuse);
    }
}

TextureSpectrum get_texture_op::operator()(const DisneyBSDF &bsdf) const {
    return bsdf.base_color;
}
