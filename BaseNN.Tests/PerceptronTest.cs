using System;
using Xunit;
using BaseNN;

namespace BaseNN.Tests
{
    public class PerceptronTest
    {
        [Fact]
        public void simpleTest()
        {
            double[] weights = {1,2};
            double bias = -2;
            double[] input = {5,2};
            var p = new Neuron(bias, weights);
            var result = p.Feed(input);
            Assert.Equal(1, result);
        }
    }
}
