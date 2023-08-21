    #pragma once
    #include <vector>
    #include <utility>

    class Layer {
    public:
        virtual void forward(const std::vector<double>& inputs, bool training) = 0;
        virtual void backward(std::vector<double>& dvalues) = 0;
        virtual std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> get_parameters() const = 0;
        virtual void set_parameters(const std::vector<std::vector<double>>& weights, const std::vector<std::vector<double>>& biases) = 0;
    };

    class Layer_Dense : public Layer {
    private:
        double weight_regularizer_l1, weight_regularizer_l2,
            bias_regularizer_l1, bias_regularizer_l2;
        std::vector<std::vector<double>> weights, biases;
        std::vector<double> inputs;

    public:
        Layer_Dense(int n_inputs, int n_neurons,
                    double weight_regularizer_l1,
                    double weight_regularizer_l2,
                    double bias_regularizer_l1,
                    double bias_regularizer_l2);

        void forward(const std::vector<double>& inputs, bool training) override;
        void backward(std::vector<double>& dvalues) override;
        std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> get_parameters() const override;
        void set_parameters(const std::vector<std::vector<double>>& weights, const std::vector<std::vector<double>>& biases) override;
    };

    class Layer_Dropout : public Layer {
    private:
        double rate;

    public:
        Layer_Dropout(double rate);

        void forward(const std::vector<double>& inputs, bool training) override;
        void backward(std::vector<double>& dvalues) override;
    };


    class Layer_Input {
    private:
        std::vector<double> output;

    public:
        void forward(const std::vector<double>& inputs, bool training);
    };

    // Add other Layer classes as needed
