#pragma once
#include <vector>
#include <algorithm>
#include <memory>
#include "path_tracing_helper.h"

struct primary_sample
{
    void backup()
    {
        valueBackup = value;
        modifyBackup = lastModificationIteration;
    }
    void restore()
    {
        value = valueBackup;
        lastModificationIteration = modifyBackup;
    }
    Real value = 0;
    int64_t lastModificationIteration = 0;
    Real valueBackup = 0;
    int64_t modifyBackup = 0;
};

inline Real ErfInv(Real x)
{
    Real w, p;
    x = std::clamp(x, -.99999, .99999);
    w = -std::log((1 - x) * (1 + x));
    if (w < 5)
    {
        w = w - 2.5f;
        p = 2.81022636e-08f;
        p = 3.43273939e-07f + p * w;
        p = -3.5233877e-06f + p * w;
        p = -4.39150654e-06f + p * w;
        p = 0.00021858087f + p * w;
        p = -0.00125372503f + p * w;
        p = -0.00417768164f + p * w;
        p = 0.246640727f + p * w;
        p = 1.50140941f + p * w;
    }
    else
    {
        w = std::sqrt(w) - 3;
        p = -0.000200214257f;
        p = 0.000100950558f + p * w;
        p = 0.00134934322f + p * w;
        p = -0.00367342844f + p * w;
        p = 0.00573950773f + p * w;
        p = -0.0076224613f + p * w;
        p = 0.00943887047f + p * w;
        p = 1.00167406f + p * w;
        p = 2.83297682f + p * w;
    }
    return p * x;
}

struct MLTSampler
{
    MLTSampler(int mutationsPerPixel, int rngSequenceIndex, Real sigma,
        Real largeStepProbability, int streamCount)
        : sigma(sigma), largeStepProbability(largeStepProbability), streamCount(streamCount)
    {
        rng = init_pcg32(rngSequenceIndex);
    }

    Real next_double()
    {
        int index = next_index();
        ensure_ready(index);
        return X[index].value;
    }

    int next_index()
    {
        int id = streamIndex + streamCount * sampleIndex;
        sampleIndex++;
        return id;
    }

    void start_iteration()
    {
        currentIteration++;
        largeStep = (next_pcg32_real<Real>(rng) < largeStepProbability);
    }

    void accept()
    {
        if (largeStep)
        {
            lastLargeStepIteration = currentIteration;
        }
    }
    void reject()
    {
        for (auto& x : X)
        {
            if (x.lastModificationIteration == currentIteration)
            {
                x.restore();
            }
        }
        --currentIteration;
    }

    void ensure_ready(int index)
    {
        if (index >= X.size())
        {
            X.resize(index + 1);
        }
        primary_sample& xi = X[index];
        if (xi.lastModificationIteration < lastLargeStepIteration)
        {
            xi.value = next_pcg32_real<Real>(rng);
            xi.lastModificationIteration = lastLargeStepIteration;
        }

        xi.backup();
        if (largeStep)
        {
            xi.value = next_pcg32_real<Real>(rng);
        }
        else
        {
            int64_t nSmall = currentIteration - xi.lastModificationIteration;
            Real normalSample = 1.414213562 * ErfInv(2 * next_pcg32_real<Real>(rng) - 1);

            Real effSigma = sigma * std::sqrt((Real)nSmall);
            xi.value += normalSample * effSigma;
            xi.value -= std::floor(xi.value);
        }
        xi.lastModificationIteration = currentIteration;
    }

    void start_stream(int stream)
    {
        streamIndex = stream;
        sampleIndex = 0;
    }

    int sampleIndex = 0, streamIndex = 0, streamCount;
    int currentIteration = 0;
    int lastLargeStepIteration = 0;
    Real largeStepProbability = 0;

    std::vector<primary_sample> X;
    pcg32_state rng;
    Real sigma = 0;
    bool largeStep = true;
};


struct ReplayableSampler
{
    ReplayableSampler()
    {
        rng = init_pcg32(0);
    }
    ReplayableSampler(int rngSequenceIndex)
    {
        rng = init_pcg32(rngSequenceIndex);
    }

    ReplayableSampler(pcg32_state rngState)
    {
        rng = rngState;
    }

    Real next_double()
    {
        int index = next_index();
        ensure_ready(index);
        return X[index];
    }

    int next_index()
    {
        return sampleIndex++;
    }

    void replay()
    {
        sampleIndex = 0;
    }

    void start_iteration()
    {
        sampleIndex = 0;
        X.clear();
    }

    void ensure_ready(int index)
    {
        if (index >= X.size())
        {
            X.resize(index + 1);
            Real& xi = X[index];
            xi = next_pcg32_real<Real>(rng);
        }
    }

    std::vector<Real> exportRandSeq() const
    {
        return X;
    }

    int sampleIndex = 0;

    std::vector<Real> X;
    pcg32_state rng;
};
