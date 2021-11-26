using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace neural_networks_kubsu.NeuralNetwork.ActivationFunction.SoftMaxActivationFunction
{
    public class SoftMaxActivationFunction: IActivationFunction
    {
        public double Activate(double value)
        {
            return 0;
        }

        public double Derivative(double value)
        {
            return value * (1 - value);
        }
    }
}
