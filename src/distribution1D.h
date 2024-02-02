#pragma once
#include "lajolla.h"
#include "vector.h"
#include <vector>

struct Distribution1D
{
	Distribution1D(const std::vector<Real>& pmf)
	{
		this->pmf = pmf;
		int n = pmf.size();
		this->cdf.resize(pmf.size() + 1);
		// Compute integral of step function at $x_i$
		cdf[0] = 0;
		for (int i = 1; i < n + 1; ++i) cdf[i] = cdf[i - 1] + pmf[i - 1] / n;

		// Transform step function integral into CDF
		funcInt = cdf[n];
		if (funcInt == 0)
		{
			for (int i = 1; i < n + 1; ++i) cdf[i] = i / Real(n);
		}
		else
		{
			for (int i = 1; i < n + 1; ++i) cdf[i] /= funcInt;
		}


		//for (int i = 1; i < this->pmf.size(); i++)
		//{
		//	this->pmf[i] = pmf[i - 1];
		//	Real prev = this->cdf[i - 1];
		//	this->cdf[i] = prev + this->pmf[i];
		//}
		//Real total = this->cdf.back();
		//funcInt = total / (this->pmf.size() - 1);
		//if (total > 0)
		//{
		//	for (int i = 0; i < (int)this->pmf.size(); i++)
		//	{
		//		this->pmf[i] /= total;
		//		this->cdf[i] /= total;
		//	}
		//}
		//else
		//{
		//	for (int i = 0; i < (int)pmf.size(); i++)
		//	{
		//		this->pmf[i] = Real(1) / Real(this->pmf.size() - 1);
		//		this->cdf[i] = Real(i) / Real(this->pmf.size() - 1);
		//	}
		//	this->cdf.back() = 1;
		//}
	}

	int sample(Real* pdf, Real rnd_param) const
	{
		int n = pmf.size();
		const auto it = std::upper_bound(this->cdf.begin(), this->cdf.end(), rnd_param);
		int ans = it - this->cdf.begin() - 1;
		*pdf = this->pmf[ans];
		return ans;
	}

	Real funcInt;
	std::vector<Real> pmf;
	std::vector<Real> cdf;
};