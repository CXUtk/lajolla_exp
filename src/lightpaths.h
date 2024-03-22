#pragma once
#include "path_tracing_helper.h"
#include "sampler.h"

enum class SampleMethod
{
    BSDF,
    NEE
};
/// An "LightPath" represents a list of vertices of a light path
/// We store the information we need for computing any sort of path contribution & sampling density.
struct LightPath
{
    Vector2                     ScreenPos;
    ReplayableSampler           Sampler;
    std::vector<PathVertex>     Vertices;
    Spectrum                    L;
    Spectrum                    P_hat;
    SampleMethod                SampleMethod = SampleMethod::BSDF;
    Real                        Pdf1 = 0;
    Real                        Pdf2 = 0;
};


struct LightPathTreeNode
{
    PathVertex  Vertex;
    PathVertex  NeeVertex;
    Spectrum    L_Nee;
    Real        Pdf_Nee = 0.0;
};

/// An "LightPathTree" represents a tree of vertices of a light path with NEE
/// We store the information we need for computing any sort of path contribution & sampling density.
struct LightPathTree
{
    ReplayableSampler                   Sampler;
    std::vector<LightPathTreeNode>      Vertices;
    Spectrum                            L;
    Spectrum                            P_hat;
    Real                                J;
};


struct LightPathSerializationInfo
{
    std::vector<Real>   Sequence;
    Spectrum            Throughput;
};

Spectrum path_sample_phase_function(int medium_id, const Ray& ray, const Scene& scene,
    ReplayableSampler& sampler, Vector3* wout, Real* pdf, Real* pdfRev)
{
    const Medium& medium = scene.media[medium_id];
    Spectrum sigma_s = get_sigma_s(medium, ray.org);
    PhaseFunction phaseFunction = get_phase_function(medium);

    Vector2 phase_rnd_param = Vector2(sampler.next_double(), sampler.next_double());

    Vector3 win = -ray.dir;
    std::optional<Vector3> next_dir = sample_phase_function(phaseFunction, win, phase_rnd_param);
    if (next_dir)
    {
        *wout = *next_dir;
        *pdf = pdf_sample_phase(phaseFunction, win, *next_dir);
        *pdfRev = pdf_sample_phase(phaseFunction, *next_dir, win);
        return eval(phaseFunction, win, *next_dir);
    }
    *pdf = 0;
    return make_zero_spectrum();
}