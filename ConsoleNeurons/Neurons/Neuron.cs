using System;
using System.Collections.Generic;
using System.Linq;

namespace ConsoleNeurons
{
    public class Neuron
    {
        public List<double> Weights { get; }
        public List<double> Inputs { get; }
        public double Output { get; private set; }
        public double Delta { get; private set; }
        public NeuronType NType { get; }
        public Neuron(int count, NeuronType neuronType = NeuronType.Normal)
        {
            NType = neuronType;
            Weights = new List<double>();
            Inputs = new List<double>();
            InitLists(count);
        }

        private void InitLists(int count)
        {
            Random ran = new Random();
            for (int i = 0; i < count; i++)
            {
                if (NType == NeuronType.Input)
                    Weights.Add(1);
                else
                    Weights.Add(ran.NextDouble());
                Inputs.Add(0);
            }
        }

        public double FeedForward(List<double> inputs)
        {
            if (inputs.Count != Weights.Count)
                throw new Exception("Amount of neurons doesn't match up with initial value");

            Inputs.Clear();
            Inputs.AddRange(inputs);

            var sum = inputs.Zip(Weights, (i, w) => i * w).Sum();
            Output = NType == NeuronType.Input ? sum : Sigmoid(sum);
            return Output;
        }

        public void SetWeights(params double[] weights)
        {
            if (weights.Length != Weights.Count)
                throw new Exception("Amount of neurons doesn't match up with initial value");
            for (int i = 0; i < Weights.Count; i++)
            {
                Weights[i] = weights[i];
            }
        }
        public void Learn(double error, double rate)
        {
            if (NType == NeuronType.Input)
                return;

            Delta = error * SigmoidDx(Output);
            for (int i = 0; i < Weights.Count; i++)
            {
                var newWeight = Weights[i] - Inputs[i] * Delta * rate;
                Weights[i] = newWeight;
            }
        }

        private static double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Pow(Math.E, -x));
        }
        private static double SigmoidDx(double x)
        {
            var sigmoid = Sigmoid(x);
            return sigmoid / (1 - sigmoid);
        }

        public override string ToString()
        {
            return Output.ToString();
        }

    }
}
