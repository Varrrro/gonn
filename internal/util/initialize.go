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

// InitializeRandom values in [-1, 1].
func InitializeRandom(n int) []float64 {
	array := make([]float64, n)

	for i := 0; i < n; i++ {
		array[i] = (rand.Float64() * 2) - 1
	}

	return array
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
