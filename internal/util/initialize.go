package util

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// InitializeZeroes filled array.
func InitializeZeroes(n int) []float64 {
	array := make([]float64, n)

	for i := 0; i < n; i++ {
		array[i] = 0.0
	}

	return array
}

// InitializeWeights array with values [-1, 1].
func InitializeWeights(nInput, nOutput int) []float64 {
	weights := make([]float64, nInput*nOutput)

	for i := 0; i < len(weights); i++ {
		weights[i] = (rand.Float64() * 2) - 1
	}

	return weights
}

// InitializeBiases array with values [-1, 1].
func InitializeBiases(n int) []float64 {
	biases := make([]float64, n)

	for i := 0; i < n; i++ {
		biases[i] = (rand.Float64() * 2) - 1
	}

	return biases
}

// InitializeWeightDeltas with zero values.
func InitializeWeightDeltas(nInput, nOutput int) []float64 {
	deltas := make([]float64, nInput*nOutput)

	for i := 0; i < len(deltas); i++ {
		deltas[i] = 0.0
	}

	return deltas
}

// InitializeTarget vector with 1.0 in the wanted index and 0.0 in the rest.
func InitializeTarget(size, wanted int) mat.Vector {
	target := mat.NewVecDense(size, nil)

	for i := 0; i < size; i++ {
		if i == wanted {
			target.SetVec(i, 1.0)
		} else {
			target.SetVec(i, 0.0)
		}
	}

	return target
}
