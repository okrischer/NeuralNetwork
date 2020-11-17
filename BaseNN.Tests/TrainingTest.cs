using System;
using Xunit;
using System.Numerics;

namespace BaseNN.Tests
{
    public class NeuronTrainingTest
    {
        // create vectors from floats
        private static Vector<float> CreateVector(float p, float q = 0)
        {
            float[] weights = new float[8];
            weights[0] = p;
            weights[1] = q;
            return new Vector<float>(weights);
        }

        [Theory]
        [InlineData(1,0)]
        [InlineData(0,1)]
        public void TrainedAndNeuronYieldsCorrectNegativeResults (float v0, float v1)
        {
            // create and train the neuron
            Neuron And = new Neuron(2, CreateVector(-1,-1));
            for (int i = 0; i < 5; i++)
            {
                And.Train(CreateVector(v0,v1), 0);
            }
            // test the neuron
            Console.WriteLine($"AND-negative: Bias: {And.Bias}, Weights: {And.Weights.ToString()}");
            int result = And.Feed(CreateVector(v0, v1));
            // check the result
            Assert.Equal(0, result);
        }
        [Theory]
        [InlineData(1,1)]
        public void TrainedAndNeuronYieldsCorrectPositiveResults (float v0, float v1)
        {
            // create and train the neuron
            Neuron And = new Neuron(2, CreateVector(-1,-1));
            for (int i = 0; i < 5; i++)
            {
                And.Train(CreateVector(v0, v1), 1);
            }
            // test the neuron
            Console.WriteLine($"AND-positive: Bias: {And.Bias}, Weights: {And.Weights.ToString()}");
            int result = And.Feed(CreateVector(v0, v1));
            // check the result
            Assert.Equal(1, result);
        }

        [Theory]
        [InlineData(1,0)]
        [InlineData(0,1)]
        [InlineData(1,1)]
        [InlineData(0,0)]
        public void TrainedOrNeuronYieldsCorrectResults (float v0, float v1)
        {
            // create and train the neuron
            Neuron Or = new Neuron(2, CreateVector(-1,-1));
            for (int i = 0; i < 35; i++)
            {
                Or.Train(CreateVector(0,0), 0);
                Or.Train(CreateVector(1,1), 1);
                Or.Train(CreateVector(1,0), 1);
                Or.Train(CreateVector(0,1), 1);
            }
            // test the neuron
            Console.WriteLine($"OR-Test: Bias: {Or.Bias}, Weights: {Or.Weights.ToString()}");
            int result = Or.Feed(CreateVector(v0, v1));
            // check the result
            if (v0 == 0 & v1 == 0)
            {
                Assert.Equal(0, result);
            }
            else 
            {
                Assert.Equal(1, result);
            }
        }
    }
}
