using System;
using System.Collections.Generic;
using System.Linq;

namespace ConsoleNeurons
{
    public class NeuralNetwork
    {
        public List<Layer> Layers { get; }
        public Topology Topology { get; }
        public NeuralNetwork(Topology topology)
        {
            Layers = new List<Layer>();
            Topology = topology;
            CreateLayer(NeuronType.Input, Topology.InputCount);

            foreach (var count in Topology.HiddenLayers)
                CreateLayer(NeuronType.Normal, count);

            CreateLayer(NeuronType.Output, Topology.OutputCount);
        }

        public double Learn(List<Tuple<double, double[]>> dataSet, int epoch)
        {
            var error = 0.0;
            for (int i = 0; i < epoch; i++)
            {
                foreach (var data in dataSet)
                {
                    error += BackPropagation(data.Item1, data.Item2);
                }
            }
            return error / epoch;
        }

        private double BackPropagation(double expected, params double[] inputs)
        {
            var actual = FeedForward(inputs).Output;
            var diff = actual - expected;

            Layers.Last().Neurons.ForEach(i => i.Learn(diff, Topology.LearningRate));
            for (int j = Layers.Count - 2; j >= 0; j--)
            {
                var layer = Layers[j];
                var previousLayer = Layers[j + 1];
                for (int i = 0; i < layer.NeuronsCount; i++)
                {
                    var neuron = layer.Neurons[i];
                    for (int k = 0; k < previousLayer.NeuronsCount; k++)
                    {
                        var previousNeuron = previousLayer.Neurons[k];
                        var error = previousNeuron.Weights[i] * previousNeuron.Delta;
                        neuron.Learn(error, Topology.LearningRate);
                    }
                }
            }
            return diff * diff;
        }

        public Neuron FeedForward(params double[] inputSignals)
        {
            SendSignalsToInput(inputSignals);
            FeedForwardLayersAfterInput();

            return Layers.Last().Neurons.OrderByDescending(i => i.Output).First();
        }

        private void FeedForwardLayersAfterInput()
        {
            for (int i = 1; i < Layers.Count; i++)
            {
                var previousLayerSignals = Layers[i - 1].GetSignals().ToList();
                Layers[i].Neurons.ForEach(l => l.FeedForward(previousLayerSignals));
            }
        }

        private void SendSignalsToInput(params double[] inputSignals)
        {
            for (int i = 0; i < inputSignals.Length; i++)
            {
                var signal = new List<double>() { inputSignals[i] };
                Layers[0].Neurons[i].FeedForward(signal);
            }
        }

        private void CreateLayer(NeuronType type, int count)
        {
            var neurons = new List<Neuron>();
            for (int i = 0; i < count; i++)
            {
                neurons.Add(new Neuron(Layers.LastOrDefault()?.NeuronsCount ?? 1, type));
            }
            Layers.Add(new Layer(neurons, type));
        }
    }
}
