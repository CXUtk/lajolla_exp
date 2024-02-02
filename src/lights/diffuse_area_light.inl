inline Vector3 sample_cos_hemisphere(const Vector2& rnd_param)
{
    Real phi = c_TWOPI * rnd_param[0];
    Real tmp = sqrt(std::clamp(1 - rnd_param[1], Real(0), Real(1)));
    return Vector3{
        cos(phi) * tmp, sin(phi) * tmp,
        sqrt(std::clamp(rnd_param[1], Real(0), Real(1)))
    };
}

Real light_power_op::operator()(const DiffuseAreaLight &light) const {
    return luminance(light.intensity) * surface_area(scene.shapes[light.shape_id]) * c_PI;
}

PointAndNormal sample_point_on_light_op::operator()(const DiffuseAreaLight &light) const {
    const Shape &shape = scene.shapes[light.shape_id];
    return sample_point_on_shape(shape, ref_point, rnd_param_uv, rnd_param_w);
}


std::optional<LightLeSampleRecord> sample_point_direction_on_light_op::operator()(const DiffuseAreaLight& light) const
{
    const Shape& shape = scene.shapes[light.shape_id];
    PointAndNormal p = sample_point_on_shape(shape, ref_point, rnd_param_uv, rnd_param_w);
    Real pdf_pos = pdf_point_on_shape(scene.shapes[light.shape_id], p, ref_point);
    Frame frame(p.normal);
    Vector3 dirlocal = sample_cos_hemisphere(rnd_param_xy);
    Vector3 dir = normalize(to_world(frame, dirlocal));
    Real pdf_dir = max(dirlocal.z, Real(0)) / c_PI;

    Spectrum L = make_zero_spectrum();
    if (dot(p.normal, dir) > 0)
    {
        L = light.intensity;
    }
    return LightLeSampleRecord{ p.position, p.normal, dir, L, pdf_pos, pdf_dir };
}

int get_medium_interface_light_op::operator()(const DiffuseAreaLight& light) const
{
    const Shape& shape = scene.shapes[light.shape_id];
    if (dot(direction, point_on_light.normal) >= 0)
    {
        return get_exterior_medium_id(shape);
    }
    return get_interior_medium_id(shape);
}


Real pdf_point_on_light_op::operator()(const DiffuseAreaLight &light) const {
    return pdf_point_on_shape(
        scene.shapes[light.shape_id], point_on_light, ref_point);
}

void pdf_point_direction_on_light_op::operator()(const DiffuseAreaLight& light) const
{
    *pdf_pos = pdf_point_on_shape(
        scene.shapes[light.shape_id], point_on_light, ref_point);
    Frame frame(point_on_light.normal);
    Vector3 dir = normalize(ref_point - point_on_light.position);
    *pdf_dir = max(dot(point_on_light.normal, dir), Real(0)) / c_PI;
}

Spectrum emission_op::operator()(const DiffuseAreaLight &light) const {
    if (dot(point_on_light.normal, view_dir) <= 0) {
        return make_zero_spectrum();
    }
    return light.intensity;
}

void init_sampling_dist_op::operator()(DiffuseAreaLight &light) const {
}
