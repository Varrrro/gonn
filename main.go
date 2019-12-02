package main

import (
	"log"

	"github.com/varrrro/gonn/internal/solution"
	"github.com/varrrro/gonn/internal/util"
)

const (
	mnistURL    = "http://yann.lecun.com/exdb/mnist/"
	localPath   = "data/mnist/"
	trainImages = "train-images-idx3-ubyte.gz"
	trainLabels = "train-labels-idx1-ubyte.gz"
	testImages  = "t10k-images-idx3-ubyte.gz"
	testLabels  = "t10k-labels-idx1-ubyte.gz"
)

func main() {
	/*
		err := util.DownloadMNIST(mnistURL, localPath, trainImages, trainLabels, testImages, testLabels)
		if err != nil {
			log.Println(err.Error())
			return
		}
	*/

	rawTrainImgs, trainLbls, err := util.ReadData(localPath, trainImages, trainLabels)
	if err != nil {
		log.Println(err.Error())
		return
	}

	rawTestImgs, testLbls, err := util.ReadData(localPath, testImages, testLabels)
	if err != nil {
		log.Println(err.Error())
		return
	}

	trainImgs := util.NormalizeImages(&rawTrainImgs)
	testImgs := util.NormalizeImages(&rawTestImgs)

	solution.InitMultilayer(&trainImgs, &testImgs, &trainLbls, &testLbls)
}
