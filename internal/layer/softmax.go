package layer

import (
	"math"

	"github.com/varrrro/gonn/internal/functions"
	"gonum.org/v1/gonum/mat"
)

// SoftmaxLayer for multi-class classification.
type SoftmaxLayer struct {
	inputSize  int
	outputSize int
	input      mat.Vector
	output     mat.Vector
	weights    mat.Matrix
	biases     mat.Vector
	deltas     mat.Vector
}

// CreateSoftmaxLayer with the given size.
func CreateSoftmaxLayer(nInput, nOutput int, weights, biases []float64) *SoftmaxLayer {
	return &SoftmaxLayer{
		inputSize:  nInput,
		outputSize: nOutput,
		weights:    mat.NewDense(nOutput, nInput, weights),
		biases:     mat.NewVecDense(nOutput, biases),
	}
}

// GetOutput of the layer.
func (l *SoftmaxLayer) GetOutput() mat.Vector {
	return l.output
}

// GetWeights of the layer.
func (l *SoftmaxLayer) GetWeights() mat.Matrix {
	return l.weights
}

// GetDeltas of the layer.
func (l *SoftmaxLayer) GetDeltas() mat.Vector {
	return l.deltas
}

// FeedForward an input through the layer.
func (l *SoftmaxLayer) FeedForward(x mat.Vector) {
	l.input = x

	z := mat.NewVecDense(l.outputSize, nil)
	z.MulVec(l.weights, l.input)
	z.AddVec(z, l.biases)

	c := mat.Max(z)
	sum := 0.0
	for i := 0; i < l.outputSize; i++ {
		sum += math.Exp(z.AtVec(i) - c)
	}

	y := mat.NewVecDense(l.outputSize, nil)
	for i := 0; i < l.outputSize; i++ {
		value := functions.Softmax(c, z.AtVec(i), sum)
		y.SetVec(i, value)
	}

	l.output = y
}

// CalculateDeltas of the layer with the given target.
func (l *SoftmaxLayer) CalculateDeltas(t mat.Vector) {
	d := mat.NewVecDense(l.outputSize, nil)
	d.SubVec(l.output, t)

	l.deltas = d
}

// CalculateHiddenDeltas for the layer with the values from the next layer.
//
// Not implemented in an output layer.
func (l *SoftmaxLayer) CalculateHiddenDeltas(nextDeltas mat.Vector, nextWeights mat.Matrix) {}

// UpdateWeights and biases of the layer with the given Eta.
func (l *SoftmaxLayer) UpdateWeights(eta float64) {
	newWeights := mat.NewDense(l.outputSize, l.inputSize, nil)
	newBiases := mat.NewVecDense(l.outputSize, nil)

	for i := 0; i < l.outputSize; i++ {
		for j := 0; j < l.inputSize; j++ {
			weightIncrement := -1.0 * eta * l.deltas.AtVec(i) * l.input.AtVec(j)
			newWeights.Set(i, j, l.weights.At(i, j)+weightIncrement)
		}

		biasIncrement := -1.0 * eta * l.deltas.AtVec(i)
		newBiases.SetVec(i, l.biases.AtVec(i)+biasIncrement)
	}

	l.weights = newWeights
	l.biases = newBiases
}
