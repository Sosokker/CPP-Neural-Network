#ifndef ACCURACY_HPP
#define ACCURACY_HPP

#include <vector>
#include <numeric>
#include <cmath>

class Accuracy {
private:
    double accumulated_sum;
    int accumulated_count;

public:
    Accuracy();
    double calculate(const std::vector<int>& predictions, const std::vector<int>& y);
    double calculate_accumulated();
    void new_pass();
    virtual int compare(int prediction, int ground_truth);
};

class Accuracy_Categorical : public Accuracy {
public:
    int compare(int prediction, int ground_truth) override;
};

class Accuracy_Regression : public Accuracy {
private:
    double precision;

public:
    Accuracy_Regression();
    void init(const std::vector<double>& y, bool reinit = false);
    int compare(int prediction, int ground_truth) override;
};

#endif // ACCURACY_HPP
