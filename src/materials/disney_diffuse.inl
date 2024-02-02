Real F_D(Real Fd90, const Vector3& n, const Vector3& w)
{
    Real NdotW = std::abs(dot(n, w));
    return 1 + (Fd90 - 1) * std::pow(1 - NdotW, 5);
}

Spectrum eval_op::operator()(const DisneyDiffuse &bsdf) const {
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

    if (dot(vertex.geometric_normal, dir_out) <= 0)
    {
        return make_zero_spectrum();
    }

    // Homework 1: implement this!
    Spectrum base_color = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real roughness = eval(
        bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real subScattering = eval(
        bsdf.subsurface, vertex.uv, vertex.uv_screen_size, texture_pool);
    Spectrum baseColor = base_color / c_PI;

    Vector3 H = normalize(dir_in + dir_out);
    Real HdotO = dot(H, dir_out);
    Real Fd90 = 0.5 + 2 * roughness * HdotO * HdotO;


    Real Fss90 = roughness * HdotO * HdotO;
    Real NdotL = std::max(Real(0), dot(frame.n, dir_out));
    Real NdotV = std::max(Real(0), dot(frame.n, dir_in));

    Spectrum f_BaseDiffuse = baseColor * F_D(Fd90, frame.n, dir_in) * F_D(Fd90, frame.n, dir_out) * NdotL;
    Spectrum f_SS = 1.25 * baseColor * 
        (F_D(Fss90, frame.n, dir_in) * F_D(Fss90, frame.n, dir_out)
            * (1.0 / (NdotV + NdotL) - 0.5) + 0.5) * NdotL;

    return lerp(f_BaseDiffuse, f_SS, subScattering);
}

Real pdf_sample_bsdf_op::operator()(const DisneyDiffuse &bsdf) const {
    if (dot(vertex.geometric_normal, dir_in) < 0 ||
        dot(vertex.geometric_normal, dir_out) < 0)
    {
        // No light below the surface
        return Real(0);
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0)
    {
        frame = -frame;
    }

    // For Lambertian, we importance sample the cosine hemisphere domain.
    return fmax(dot(frame.n, dir_out), Real(0)) / c_PI;
}

std::optional<BSDFSampleRecord> sample_bsdf_op::operator()(const DisneyDiffuse &bsdf) const {
    // For Lambertian, we importance sample the cosine hemisphere domain.
    if (dot(vertex.geometric_normal, dir_in) < 0) {
        // Incoming direction is below the surface.
        return {};
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }    
    Real roughness = eval(
        bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);

    return BSDFSampleRecord{
        to_world(frame, sample_cos_hemisphere(rnd_param_uv)),
        Real(0) /* eta */, roughness /* roughness */};
}

TextureSpectrum get_texture_op::operator()(const DisneyDiffuse &bsdf) const {
    return bsdf.base_color;
}
