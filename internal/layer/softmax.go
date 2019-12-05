package layer

import (
	"math"

	"github.com/varrrro/gonn/internal/functions"
	"github.com/varrrro/gonn/internal/util"
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
	//weightIncrements mat.Matrix
	//biasIncrements   mat.Vector
}

// CreateSoftmaxLayer with the given size.
func CreateSoftmaxLayer(nInput, nOutput int) *SoftmaxLayer {
	return &SoftmaxLayer{
		inputSize:  nInput,
		outputSize: nOutput,
		weights:    mat.NewDense(nOutput, nInput, util.InitializeRandom(nInput*nOutput)),
		biases:     mat.NewVecDense(nOutput, util.InitializeRandom(nOutput)),
		//weightIncrements: mat.NewDense(nOutput, nInput, util.InitializeZeroes(nInput*nOutput)),
		//biasIncrements:   mat.NewVecDense(nOutput, util.InitializeZeroes(nOutput)),
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
func (l *SoftmaxLayer) UpdateWeights(eta, mu float64) {
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

/*
// DoMomentumStep with the given Mu.
func (l *SoftmaxLayer) DoMomentumStep(mu float64) {
	newWeights := mat.NewDense(l.outputSize, l.inputSize, nil)
	newBiases := mat.NewVecDense(l.outputSize, nil)

	newWeightIncrements := mat.NewDense(l.outputSize, l.inputSize, nil)
	newBiasIncrements := mat.NewVecDense(l.outputSize, nil)

	for i := 0; i < l.outputSize; i++ {
		for j := 0; j < l.inputSize; j++ {
			weightIncrement := (mu * l.weightIncrements.At(i, j))
			newWeights.Set(i, j, l.weights.At(i, j)+weightIncrement)
			newWeightIncrements.Set(i, j, weightIncrement)
		}

		biasIncrement := (mu * l.biasIncrements.AtVec(i))
		newBiases.SetVec(i, l.biases.AtVec(i)+biasIncrement)
		newBiasIncrements.SetVec(i, biasIncrement)
	}

	l.weights = newWeights
	l.biases = newBiases

	l.weightIncrements = newWeightIncrements
	l.biasIncrements = newBiasIncrements
}

// DoCorrectionStep with the given Eta.
func (l *SoftmaxLayer) DoCorrectionStep(eta float64) {
	newWeights := mat.NewDense(l.outputSize, l.inputSize, nil)
	newBiases := mat.NewVecDense(l.outputSize, nil)

	newWeightIncrements := mat.NewDense(l.outputSize, l.inputSize, nil)
	newBiasIncrements := mat.NewVecDense(l.outputSize, nil)

	for i := 0; i < l.outputSize; i++ {
		for j := 0; j < l.inputSize; j++ {
			weightIncrement := -1.0 * eta * l.deltas.AtVec(i) * l.input.AtVec(j)
			newWeights.Set(i, j, l.weights.At(i, j)+weightIncrement)
			newWeightIncrements.Set(i, j, l.weightIncrements.At(i, j)+weightIncrement)
		}

		biasIncrement := - -1.0 * eta * l.deltas.AtVec(i)
		newBiases.SetVec(i, l.biases.AtVec(i)+biasIncrement)
		newBiasIncrements.SetVec(i, l.biasIncrements.AtVec(i)+biasIncrement)
	}

	l.weights = newWeights
	l.biases = newBiases

	l.weightIncrements = newWeightIncrements
	l.biasIncrements = newBiasIncrements
}
*/
