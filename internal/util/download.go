package util

import (
	"io"
	"log"
	"net/http"
	"os"
)

// DownloadMNIST image and label sets to from the given URL.
func DownloadMNIST(url, localPath, trainingImages, trainingLabels, testImages, testLabels string) (err error) {
	log.Println("Downloading " + trainingImages)
	err = downloadFile(url+trainingImages, localPath+trainingImages)
	if err != nil {
		log.Println("Error while downloading training images")
		return err
	}
	log.Println("Downloaded " + trainingImages + " correctly")

	log.Println("Downloading " + trainingLabels)
	err = downloadFile(url+trainingLabels, localPath+trainingLabels)
	if err != nil {
		log.Println("Error while downloading training labels")
		return err
	}
	log.Println("Downloaded " + trainingLabels + " correctly")

	log.Println("Downloading " + testImages)
	err = downloadFile(url+testImages, localPath+testImages)
	if err != nil {
		log.Println("Error while downloading test images")
		return err
	}
	log.Println("Downloaded " + testImages + " correctly")

	log.Println("Downloading " + testLabels)
	err = downloadFile(url+testLabels, localPath+testLabels)
	if err != nil {
		log.Println("Error while downloading test labels")
		return err
	}
	log.Println("Downloaded " + testLabels + " correctly")

	return err
}

func downloadFile(url string, filepath string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	out, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	return err
}
