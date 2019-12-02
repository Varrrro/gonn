package util

import (
	"compress/gzip"
	"encoding/binary"
	"errors"
	"log"
	"os"
)

// ReadData from MNIST files into memory.
func ReadData(localPath, imgsFilename, labelsFilename string) (imgs [][]byte, labels []int, err error) {
	nimgs, imgs, err := readImages(localPath + imgsFilename)
	if err != nil {
		log.Printf("Error while reading " + imgsFilename + " images")
		return nil, nil, err
	}

	nlabels, labels, err := readLabels(localPath + labelsFilename)
	if err != nil {
		log.Printf("Error while reading " + labelsFilename + " labels")
		return nil, nil, err
	}

	if nimgs != nlabels {
		return nil, nil, errors.New("The number of images and labels read isn't the same")
	}

	return imgs, labels, nil
}

func readImages(filepath string) (size int, images [][]byte, err error) {
	file, err := os.Open(filepath)
	if err != nil {
		return 0, nil, err
	}
	defer file.Close()

	gz, err := gzip.NewReader(file)
	if err != nil {
		return 0, nil, err
	}
	defer gz.Close()

	var (
		magic    int32
		n        int32
		nrows    int32
		ncolumns int32
	)

	if err = binary.Read(gz, binary.BigEndian, &magic); err != nil {
		return 0, nil, err
	}
	if magic != 2051 {
		return 0, nil, os.ErrInvalid
	}
	if err = binary.Read(gz, binary.BigEndian, &n); err != nil {
		return 0, nil, err
	}
	if err = binary.Read(gz, binary.BigEndian, &nrows); err != nil {
		return 0, nil, err
	}
	if err = binary.Read(gz, binary.BigEndian, &ncolumns); err != nil {
		return 0, nil, err
	}

	log.Printf("Reading %d %dx%d images", n, nrows, ncolumns)

	images = make([][]byte, n)
	m := nrows * ncolumns

	for i := 0; i < int(n); i++ {
		images[i] = make([]byte, m)

		for j := 0; j < int(m); j++ {
			if err = binary.Read(gz, binary.BigEndian, &images[i][j]); err != nil {
				return 0, nil, err
			}
		}
	}

	return int(n), images, nil
}

func readLabels(filepath string) (size int, labels []int, err error) {
	file, err := os.Open(filepath)
	if err != nil {
		return 0, nil, err
	}
	defer file.Close()

	gz, err := gzip.NewReader(file)
	if err != nil {
		return 0, nil, err
	}
	defer gz.Close()

	var (
		magic int32
		n     int32
	)

	if err = binary.Read(gz, binary.BigEndian, &magic); err != nil {
		return 0, nil, err
	}
	if magic != 2049 {
		return 0, nil, os.ErrInvalid
	}
	if err = binary.Read(gz, binary.BigEndian, &n); err != nil {
		return 0, nil, err
	}

	log.Printf("Reading %d labels", n)

	var l uint8
	labels = make([]int, n)

	for i := 0; i < int(n); i++ {
		if err = binary.Read(gz, binary.BigEndian, &l); err != nil {
			return 0, nil, err
		}

		labels[i] = int(l)
	}

	return int(n), labels, nil
}
