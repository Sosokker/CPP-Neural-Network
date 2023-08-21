#ifndef ACTIVATIONFUNCTION_HPP
#define ACTIVATIONFUNCTION_HPP

#include <vector>

class Activation_ReLU {
public:
    void forward(std::vector<double>& inputs, bool training);
    void backward(std::vector<double>& dvalues);
    std::vector<double> predictions(const std::vector<double>& outputs);
private:
    std::vector<double> inputs;
    std::vector<double> output;
    std::vector<double> dinputs;
};

class Activation_Softmax {
public:
    void forward(std::vector<double>& inputs, bool training);
    void backward(std::vector<double>& dvalues);
    std::vector<double> predictions(const std::vector<double>& outputs);
private:
    std::vector<double> inputs;
    std::vector<double> output;
    std::vector<double> dinputs;
};

class Activation_Sigmoid {
public:
    void forward(std::vector<double>& inputs, bool training);
    void backward(std::vector<double>& dvalues);
    std::vector<double> predictions(const std::vector<double>& outputs);
private:
    std::vector<double> inputs;
    std::vector<double> output;
    std::vector<double> dinputs;
};

class Activation_Linear {
public:
    void forward(std::vector<double>& inputs, bool training);
    void backward(std::vector<double>& dvalues);
    std::vector<double> predictions(const std::vector<double>& outputs);
private:
    std::vector<double> inputs;
    std::vector<double> output;
    std::vector<double> dinputs;
};

#endif // ACTIVATIONFUNCTION_HPP
