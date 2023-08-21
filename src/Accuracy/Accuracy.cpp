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
    Accuracy() : accumulated_sum(0.0), accumulated_count(0) {}

    double calculate(const std::vector<int>& predictions, const std::vector<int>& y) {
        std::vector<int> comparisons(predictions.size());
        for (size_t i = 0; i < predictions.size(); ++i) {
            comparisons[i] = compare(predictions[i], y[i]);
        }

        double accuracy = static_cast<double>(std::accumulate(comparisons.begin(), comparisons.end(), 0)) / predictions.size();

        accumulated_sum += std::accumulate(comparisons.begin(), comparisons.end(), 0);
        accumulated_count += predictions.size();

        return accuracy;
    }

    double calculate_accumulated() {
        double accuracy = static_cast<double>(accumulated_sum) / accumulated_count;
        return accuracy;
    }

    void new_pass() {
        accumulated_sum = 0.0;
        accumulated_count = 0;
    }

    virtual int compare(int prediction, int ground_truth) {
        return prediction == ground_truth ? 1 : 0;
    }
};

class Accuracy_Categorical : public Accuracy {
public:
    int compare(int prediction, int ground_truth) override {
        return prediction == ground_truth ? 1 : 0;
    }
};

class Accuracy_Regression : public Accuracy {
private:
    double precision;

public:
    Accuracy_Regression() : precision(0.0) {}

    void init(const std::vector<double>& y, bool reinit = false) {
        if (precision == 0.0 || reinit) {
            precision = std::sqrt(std::accumulate(y.begin(), y.end(), 0.0, [](double acc, double val) {
                return acc + (val * val);
            })) / 250.0;
        }
    }

    int compare(int prediction, int ground_truth) override {
        return std::abs(prediction - ground_truth) < precision ? 1 : 0;
    }
};

#endif // ACCURACY_HPP
