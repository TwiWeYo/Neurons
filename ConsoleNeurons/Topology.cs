using System.Collections.Generic;
using System.Linq;

namespace ConsoleNeurons
{
    public class Topology
    {
        public Topology(int inputCount, int outputCount, double learningRate, params int[] layers)
        {
            InputCount = inputCount;
            OutputCount = outputCount;
            LearningRate = learningRate;
            HiddenLayers = layers.ToList();
        }

        public int InputCount { get; }
        public int OutputCount { get; }
        public double LearningRate { get; }
        public List<int> HiddenLayers { get; }
    }
}
