using System;
using System.Collections.Generic;
using System.Linq;

namespace ConsoleNeurons
{
    public class Layer
    {
        public List<Neuron> Neurons { get; }
        public int NeuronsCount => Neurons?.Count ?? 0;
        public NeuronType Ntype { get; }
        public Layer(List<Neuron> neurons, NeuronType type = NeuronType.Normal)
        {
            if (neurons.Any(q => q.NType != type))
                throw new Exception("Neuron types in collection does not match initial type");
            Neurons = neurons;
            Ntype = type;
        }
        
        public List<double> GetSignals()
        {
            return Neurons.Select(i => i.Output).ToList();
        }
        public override string ToString()
        {
            return Ntype.ToString();
        }
    }
}
