#include "camera.h"
#include "lajolla.h"
#include "transform.h"

#include <cmath>

Camera::Camera(const Matrix4x4 &cam_to_world,
               Real fov,
               int width, int height,
               const Filter &filter,
               int medium_id)
    : cam_to_world(cam_to_world),
      world_to_cam(inverse(cam_to_world)),
      width(width), height(height),
      filter(filter), medium_id(medium_id) {
    Real aspect = (Real)width / (Real)height;
    cam_to_sample = scale(Vector3(-Real(0.5), -Real(0.5) * aspect, Real(1.0))) *
                    translate(Vector3(-Real(1.0), -Real(1.0) / aspect, Real(0.0))) *
                    perspective(fov);
    sample_to_cam = inverse(cam_to_sample);
    Vector3 pMin = xform_point(sample_to_cam, Vector3(0, 0, 0));
    Vector3 pMax = xform_point(sample_to_cam, Vector3(1, 1, 0));
    //pMin = normalize(xform_vector(camera.cam_to_world, pMin));
    //pMax = normalize(xform_vector(camera.cam_to_world, pMax));
    pMin /= pMin.z;
    pMax /= pMax.z;
    A = std::abs((pMax.x - pMin.x) * (pMax.y - pMin.y));
}

Ray sample_primary(const Camera &camera,
                   const Vector2 &screen_pos) {
    // screen_pos' domain is [0, 1]^2
    Vector2 pixel_pos{screen_pos.x * camera.width, screen_pos.y * camera.height};

    // Importance sample from the pixel filter (see filter.h for more explanation).
    // We first extract the subpixel offset.
    Real dx = pixel_pos.x - floor(pixel_pos.x);
    Real dy = pixel_pos.y - floor(pixel_pos.y);
    // dx/dy are uniform variables in [0, 1],
    // so let's remap them using importance sampling.
    Vector2 offset = sample(camera.filter, Vector2{dx, dy});
    // Filters are placed at pixel centers.
    Vector2 remapped_pos{
      (floor(pixel_pos.x) + Real(0.5) + offset.x) / camera.width,
      (floor(pixel_pos.y) + Real(0.5) + offset.y) / camera.height};

    Vector3 pt = xform_point(camera.sample_to_cam,
        Vector3(remapped_pos[0], remapped_pos[1], Real(0)));
    Vector3 dir = normalize(pt);
    return Ray{xform_point(camera.cam_to_world, Vector3{0, 0, 0}),
               // the last normalize might not be necessary
               normalize(xform_vector(camera.cam_to_world, dir)),
               Real(0), infinity<Real>()};
}

Spectrum eval_importance(const Camera& camera, const Vector3& dir, Vector2* rasterPos)
{
    Vector3 cam_dir = normalize(xform_vector(camera.world_to_cam, dir));
    Vector3 cam_space = xform_vector(camera.cam_to_sample, cam_dir);
    *rasterPos = Vector2(cam_space.x / cam_space.z, cam_space.y / cam_space.z);

    Real cosTheta = cam_dir.z;
    if (cosTheta <= 0) return make_zero_spectrum();

    if (rasterPos->x < 0 || rasterPos->x > 1 || rasterPos->y < 0 || rasterPos->y > 1)
    {
        return make_zero_spectrum();
    }

    Real lensArea = 1;
    Real cos2Theta = cosTheta * cosTheta;
    return make_const_spectrum(1.0 / (camera.A * lensArea * cos2Theta * cos2Theta));
}

void pdf_importance(const Camera& camera, const Vector3& dir, Real* pdf_pos, Real* pdf_dir)
{
    Vector3 cam_dir = normalize(xform_vector(camera.world_to_cam, dir));
    Vector3 cam_space = xform_vector(camera.cam_to_sample, cam_dir);
    Vector2 rasterPos = Vector2(cam_space.x / cam_space.z, cam_space.y / cam_space.z);

    Real cosTheta = cam_dir.z;
    if (cosTheta <= 0)
    {
        *pdf_pos = *pdf_dir = 0;
        return;
    }

    if (rasterPos.x < 0 || rasterPos.x > 1 || rasterPos.y < 0 || rasterPos.y > 1)
    {
        *pdf_pos = *pdf_dir = 0;
        return;
    }

    Real lensArea = 1;
    *pdf_pos = 1.0 / lensArea;
    *pdf_dir = 1.0 / (camera.A * cosTheta * cosTheta * cosTheta);
}

Spectrum sample_importance(const Camera& camera, const Vector3& pos, const Vector3& normal, Real* pdf, Vector2* rasterPos)
{
    Vector3 diff = pos - xform_point(camera.cam_to_world,
        Vector3{ 0, 0, 0 });
    Vector3 dir = normalize(diff);
    Vector3 cam_dir = normalize(xform_vector(camera.world_to_cam, dir));

    Real cosTheta = cam_dir.z;
    if (cosTheta <= 0)
    {
        *pdf = 0;
        return make_zero_spectrum();
    }
    *pdf = length_squared(diff) / (cosTheta);
    return eval_importance(camera, dir, rasterPos);
}
