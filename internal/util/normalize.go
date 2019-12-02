package util

import (
	"gonum.org/v1/gonum/mat"
)

// NormalizeImages to [0, 1].
func NormalizeImages(imgs *[][]byte) []mat.Vector {
	normImgs := make([]mat.Vector, len(*imgs))

	for i, img := range *imgs {
		normImgs[i] = normalize(&img)
	}

	return normImgs
}

func normalize(raw *[]byte) mat.Vector {
	rawData := *raw
	n := len(rawData)
	data := mat.NewVecDense(n, nil)

	for i := 0; i < n; i++ {
		data.SetVec(i, float64(rawData[i])/255.0)
	}

	return data
}
