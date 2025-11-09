#pragma once
#include "arguments.hpp"
#include "dataset.hpp"
#include <fstream>
#include <cmath>
#include <vector>

class SearchMethod {
public:
    explicit SearchMethod(const Arguments& a, const int b, const std::vector<VectorData> c) : Args(a), Dim(b), Data(c) {}
    virtual void buildIndex() = 0;
    virtual void search(const std::vector<VectorData> &queries, std::ofstream& out) = 0;

protected:
    Arguments Args;
    int Dim = 0;
    std::vector<VectorData> Data;

    inline double l2(const std::vector<float> &a, const std::vector<float> &b)
    {
        double s = 0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            double d = a[i] - b[i];
            s += d * d;
        }
        return std::sqrt(s);
    }
};
