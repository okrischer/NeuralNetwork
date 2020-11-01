using System;
using Xunit;
using BaseNN;

namespace BaseNN.Tests
{
    public class PerceptronTest
    {
        // bias and weights for simple tests
        double bias = -2;
        double[] weights = {1,2};

        [Fact]
        public void ShouldThrowException()
        {
            double[] input = {2};
            var perceptron = new Neuron(bias, weights);
            try
            {
                var result = perceptron.Feed(input);
            }
            catch (OperationCanceledException oce)
            {
                Assert.Equal("the number of input values does not match the number of weights"
                , oce.Message);
            }
        }
        [Theory]
        [InlineData(2,1)]
        [InlineData(1,2)]
        [InlineData(1,1)]
        public void SimpleTestShouldYield1(double i1, double i2)
        {
            double[] input = {i1, i2};
            var perceptron = new Neuron(bias, weights);
            var result = perceptron.Feed(input);
            Assert.Equal(1, result);
        }
        [Theory]
        [InlineData(0,1)]
        [InlineData(1,0)]
        [InlineData(0,0)]
        public void SimpleTestShouldYield0(double i1, double i2)
        {
            double[] input = {i1, i2};
            var perceptron = new Neuron(bias, weights);
            var result = perceptron.Feed(input);
            Assert.Equal(0, result);
        }
    }
}
