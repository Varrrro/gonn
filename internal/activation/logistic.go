package activation

import "math"

// Logistic activation function.
func Logistic(z float64) float64 {
	return 1 / (1 + math.Exp(z*-1))
}
